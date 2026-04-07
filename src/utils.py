import pandas as pd
import joblib
import logging
import os

# -----------------------------
# Configuration du logging
# -----------------------------
# Permet d'afficher des messages d'information ou d'erreur pendant l'exécution
# level=logging.INFO → on affiche les messages d'information et plus graves
# format → affiche l'heure, le niveau du message et le texte
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)  # Création d'un objet logger spécifique au module

# -----------------------------
# Définition du chemin par défaut pour les données brutes
# -----------------------------
# Utilisation d'un chemin relatif pour charger le fichier CSV depuis le dossier 'data/raw'
DEFAULT_RAW_DATA_PATH = os.path.join(
    "data", "raw", "retail_customers_COMPLETE_CATEGORICAL - retail_customers_COMPLETE_CATEGORICAL.csv"
)

# -----------------------------
# Fonction pour charger les données
# -----------------------------
def load_data(filepath=DEFAULT_RAW_DATA_PATH):
    """
    Charge le dataset CSV depuis le chemin fourni.
    
    Paramètre:
        filepath (str) : chemin du fichier CSV à charger
        
    Retour:
        df (pandas.DataFrame) : dataframe contenant les données chargées
        
    Fonctionnement:
        - Affiche un message indiquant le fichier en cours de chargement
        - Tente de lire le CSV avec pandas
        - Si succès, affiche la taille du dataset et retourne le DataFrame
        - Si erreur, log l'erreur et lève une exception
    """
    logger.info(f"Loading data from {filepath}")
    try:
        df = pd.read_csv(filepath)
        logger.info(f"Successfully loaded data with shape {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

# -----------------------------
# Fonction pour sauvegarder un DataFrame
# -----------------------------
def save_data(df, filepath):
    """
    Sauvegarde un DataFrame dans un fichier CSV.
    
    Paramètre:
        df (pandas.DataFrame) : dataframe à sauvegarder
        filepath (str) : chemin du fichier CSV de sortie
        
    Fonctionnement:
        - Crée le dossier parent si nécessaire (os.makedirs)
        - Sauvegarde le dataframe sans l'index
        - Log l'opération réussie ou les erreurs
    """
    logger.info(f"Saving data to {filepath}")
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)  # Crée les dossiers si nécessaire
        df.to_csv(filepath, index=False)  # Sauvegarde le dataframe
        logger.info(f"Successfully saved data.")
    except Exception as e:
        logger.error(f"Error saving data: {e}")
        raise

# -----------------------------
# Fonction pour sauvegarder un modèle ou un transformateur
# -----------------------------
def save_model(model, filepath):
    """
    Sauvegarde un modèle entraîné ou un transformateur avec joblib.
    
    Paramètre:
        model : modèle ou transformateur scikit-learn à sauvegarder
        filepath (str) : chemin du fichier de sortie (.joblib ou .pkl)
        
    Fonctionnement:
        - Crée le dossier parent si nécessaire
        - Utilise joblib.dump pour sauvegarder l'objet
        - Log l'opération réussie ou les erreurs
    """
    logger.info(f"Saving model to {filepath}")
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(model, filepath)
        logger.info("Successfully saved model.")
    except Exception as e:
        logger.error(f"Error saving model: {e}")
        raise

# -----------------------------
# Fonction pour charger un modèle ou un transformateur
# -----------------------------
def load_model(filepath):
    """
    Charge un modèle entraîné ou un transformateur depuis un fichier.
    
    Paramètre:
        filepath (str) : chemin du fichier à charger
        
    Retour:
        model : objet Python chargé (modèle ou transformateur)
        
    Fonctionnement:
        - Utilise joblib.load pour charger l'objet
        - Log l'opération réussie ou les erreurs
    """
    logger.info(f"Loading model from {filepath}")
    try:
        model = joblib.load(filepath)
        logger.info("Successfully loaded model.")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise