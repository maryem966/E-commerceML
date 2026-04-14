import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
import os
import joblib

from utils import load_data, save_data, save_model, logger

# ============================================================
# COLONNES À SUPPRIMER AVANT TOUTE MODÉLISATION
# ============================================================
# - CustomerID : identifiant unique, sans valeur prédictive.
# - NewsletterSubscribed : constante (toujours "Yes"), apporte 0 information.
# - ChurnRiskCategory : DÉRIVÉE DIRECTEMENT DE LA CIBLE → DATA LEAKAGE MAJEUR.
#   Si on la garde, le modèle apprend à partir de la réponse → accuracy = 1.0 artificielle !
# - ChurnRisk : même raison que ChurnRiskCategory.
# - AccountStatus : corrélée à la cible (compte inactif = client parti) → DATA LEAKAGE.
# IMPORTANT : 'CustomerType' et 'RFMSegment' sont aussi du data leakage !
# CustomerType='Perdu' → 100% Churn=1, 'Hyperactif/Régulier' → 100% Churn=0
# RFMSegment='Dormants' → 100% Churn=1, 'Champions' → 100% Churn=0
# LoyaltyLevel='Ancien' → 100% Churn=0, 'Etabli' → 91% Churn=0
# FavoriteSeason='Automne' → 98% Churn=0 (suspisieusement corrélé)
# Ces colonnes sont CONSTRUITES À PARTIR de la même logique que la cible → LEAKAGE.
LEAKAGE_COLS = [
    'CustomerID',
    'NewsletterSubscribed',
    'ChurnRiskCategory',
    'ChurnRisk',
    'AccountStatus',
    'CustomerType',   # Leakage : Perdu=100% churn, Régulier=0% churn
    'RFMSegment',     # Leakage : Dormants=100% churn, Champions=0% churn
    'LoyaltyLevel',   # Leakage : Ancien=100% fidèle (dérivé de l'ancienneté=churn signal)
    'FavoriteSeason', # Leakage : Automne=98% fidèle (dérivé des patterns d'achat)
    'Recency',        # Leakage : Churn = 1 if Recency > 90 else 0
]


def handle_outliers(df):
    """Détecte et corrige les valeurs aberrantes dans les colonnes clés."""
    logger.info("Handling outliers...")
    
    if 'SupportTicketsCount' in df.columns:
        # Valeurs aberrantes : -1 ou 999 (valeur codée par défaut).
        # On plafonne au 99e percentile pour conserver les vrais cas élevés.
        upper_limit = df['SupportTicketsCount'].quantile(0.99)
        df.loc[df['SupportTicketsCount'] > upper_limit, 'SupportTicketsCount'] = upper_limit
        df.loc[df['SupportTicketsCount'] < 0, 'SupportTicketsCount'] = 0
        
    if 'SatisfactionScore' in df.columns:
        # Score valide : 1 à 5. Les valeurs -1 ou 99 sont des erreurs de saisie.
        # On remplace par la médiane des valeurs valides.
        valid_scores = df.loc[(df['SatisfactionScore'] >= 1) & (df['SatisfactionScore'] <= 5), 'SatisfactionScore']
        median_score = valid_scores.median() if len(valid_scores) > 0 else 3.0
        df.loc[(df['SatisfactionScore'] > 5) | (df['SatisfactionScore'] < 1), 'SatisfactionScore'] = median_score
        
    return df


def feature_engineering(df):
    """Crée de nouvelles features et parse les dates."""
    logger.info("Performing feature engineering...")
    
    # --- Parsing de la date d'inscription ---
    if 'RegistrationDate' in df.columns:
        reg_date = pd.to_datetime(df['RegistrationDate'], dayfirst=True, errors='coerce')
        df['RegYear']    = reg_date.dt.year.fillna(-1).astype(int)
        df['RegMonth']   = reg_date.dt.month.fillna(-1).astype(int)
        df['RegDay']     = reg_date.dt.day.fillna(-1).astype(int)
        df['RegWeekday'] = reg_date.dt.weekday.fillna(-1).astype(int)
        df = df.drop(columns=['RegistrationDate'])
        
    # --- Feature sur l'IP ---
    if 'LastLoginIP' in df.columns:
        def is_private_ip(ip):
            try:
                return 1 if str(ip).startswith(('192.168.', '10.', '172.')) else 0
            except:
                return 0
        df['Is_Private_IP'] = df['LastLoginIP'].apply(is_private_ip)
        df = df.drop(columns=['LastLoginIP'])
        
    # --- Features comportementales dérivées ---
    if 'MonetaryTotal' in df.columns and 'Frequency' in df.columns:
        df['AvgBasketValue'] = np.where(
            df['Frequency'] > 0,
            df['MonetaryTotal'] / df['Frequency'],
            0
        )
    
    # --- Suppression des colonnes à data leakage ---
    # Ces colonnes sont directement dérivées de la cible 'Churn' ou identifiants inutiles.
    for col in LEAKAGE_COLS:
        if col in df.columns:
            df = df.drop(columns=[col])
            logger.info(f"Dropped leakage/useless column: {col}")
        
    return df


def drop_high_correlation(X, threshold=0.85):
    """Supprime les features numériques trop corrélées entre elles (multicolinéarité)."""
    logger.info("Dropping highly correlated numerical features...")
    numeric_df = X.select_dtypes(include=[np.number])
    corr_matrix = numeric_df.corr().abs()
    
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    
    logger.info(f"Dropped {len(to_drop)} features due to correlation > {threshold}: {to_drop}")
    X = X.drop(columns=to_drop)
    return X, to_drop


def create_pipeline(X_train):
    """Construit le pipeline de prétraitement (imputation + scaling + encodage)."""
    numeric_features    = X_train.select_dtypes(include=['int64', 'float64', 'int32']).columns.tolist()
    categorical_features = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
    
    logger.info(f"Building pipeline: {len(numeric_features)} numerical, {len(categorical_features)} categorical features.")
    
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler',  StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot',  OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer,    numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )
        
    # PCA : réduit la dimensionnalité en conservant 95% de la variance
    return Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('pca', PCA(n_components=0.95))
    ]), numeric_features, categorical_features


def run_preprocessing(data_path=None):
    """Pipeline principal de prétraitement : charge, nettoie, transforme et sauvegarde les données."""
    
    if data_path:
        df = load_data(data_path)
    else:
        df = load_data()
    
    # Étape 1 : Nettoyage des outliers
    df = handle_outliers(df)
    
    # Étape 2 : Feature Engineering (inclut la suppression des colonnes leakage)
    df = feature_engineering(df)
    
    # Sauvegarde du dataset traité avant le split
    os.makedirs(os.path.join('data', 'processed'), exist_ok=True)
    df.to_csv(os.path.join('data', 'processed', 'processed_data.csv'), index=False)
    
    # Vérification que 'Churn' est bien présent dans le dataset
    if 'Churn' not in df.columns:
        raise ValueError("La colonne 'Churn' est introuvable après le feature engineering !")
    
    # Étape 3 : Séparation X / y
    y = df['Churn']
    X = df.drop(columns=['Churn'])
    
    # Étape 4 : Suppression des features trop corrélées
    X, dropped_cols = drop_high_correlation(X, threshold=0.85)
    
    # Étape 5 : Split Train/Test stratifié (80/20)
    logger.info("Splitting dataset 80/20 with stratification...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Étape 6 : Construction et ajustement du pipeline de prétraitement
    pipeline, num_cols, cat_cols = create_pipeline(X_train)
    
    logger.info("Fitting preprocessing pipeline on training data...")
    X_train_processed = pipeline.fit_transform(X_train)
    X_test_processed  = pipeline.transform(X_test)
    
    # Étape 7 : Rééquilibrage des classes avec SMOTE (uniquement sur le Train)
    logger.info("Applying SMOTE to rebalance classes...")
    try:
        from imblearn.over_sampling import SMOTE
        smote = SMOTE(random_state=42)
        X_train_processed, y_train = smote.fit_resample(X_train_processed, y_train)
        logger.info("SMOTE successfully applied.")
    except ImportError:
        logger.warning("imbalanced-learn not installed. Skipping SMOTE. Run: pip install imbalanced-learn")
        
    logger.info(f"Processed shapes: Train={X_train_processed.shape}, Test={X_test_processed.shape}")
    
    # Étape 8 : Sauvegarde des splits
    os.makedirs(os.path.join('data', 'train_test'), exist_ok=True)
    pd.DataFrame(X_train_processed).to_csv(os.path.join('data', 'train_test', 'X_train.csv'), index=False)
    pd.DataFrame(X_test_processed).to_csv(os.path.join('data', 'train_test', 'X_test.csv'), index=False)
    y_train.to_csv(os.path.join('data', 'train_test', 'y_train.csv'), index=False)
    y_test.to_csv(os.path.join('data', 'train_test', 'y_test.csv'), index=False)
    
    # Étape 9 : Sauvegarde du pipeline et des métadonnées
    os.makedirs('models', exist_ok=True)
    save_model(pipeline, os.path.join('models', 'preprocessor_pipeline.pkl'))
    save_model(
        {'dropped_cols': dropped_cols, 'num_cols': num_cols, 'cat_cols': cat_cols},
        os.path.join('models', 'features_meta.pkl')
    )
    
    logger.info("Preprocessing complete. Dataset ready for training.")


if __name__ == '__main__':
    run_preprocessing()
