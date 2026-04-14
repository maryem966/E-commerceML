import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)

from utils import save_model, logger


# ==============================
# Load data
# ==============================
def load_processed_data():
    logger.info("Loading train/test splits...")
    X_train = pd.read_csv(os.path.join('data', 'train_test', 'X_train.csv'))
    X_test = pd.read_csv(os.path.join('data', 'train_test', 'X_test.csv'))
    y_train = pd.read_csv(os.path.join('data', 'train_test', 'y_train.csv')).values.ravel()
    y_test = pd.read_csv(os.path.join('data', 'train_test', 'y_test.csv')).values.ravel()
    return X_train, X_test, y_train, y_test


# ==============================
# Create reports folder
# ==============================
def create_reports_dir():
    os.makedirs('reports', exist_ok=True)


# ==============================
# Confusion Matrix Plot
# ==============================
def save_confusion_matrix(name, y_test, preds):
    cm = confusion_matrix(y_test, preds)

    plt.figure()
    sns.heatmap(cm, annot=True, fmt='d')
    plt.title(f'Confusion Matrix - {name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')

    filepath = os.path.join('reports', f'{name}_confusion_matrix.png')
    plt.savefig(filepath)
    plt.close()

    logger.info(f"Confusion matrix saved to {filepath}")


# ==============================
# Feature Importance
# ==============================
def save_feature_importance(name, model, feature_names):
    plt.figure()

    if hasattr(model, "feature_importances_"):
        # Random Forest
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        # Logistic Regression
        importances = np.abs(model.coef_[0])
    else:
        logger.warning(f"No feature importance available for {name}")
        return

    indices = np.argsort(importances)[::-1][:20]  # Top 20 features

    plt.bar(range(len(indices)), importances[indices])
    plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation=90)
    plt.title(f'Feature Importance - {name}')

    filepath = os.path.join('reports', f'{name}_feature_importance.png')
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()

    logger.info(f"Feature importance saved to {filepath}")


# ==============================
# Evaluation
# ==============================
def evaluate_model(name, model, X_test, y_test):
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else [0]*len(y_test)

    metrics = {
        'Accuracy': accuracy_score(y_test, preds),
        'Precision': precision_score(y_test, preds, zero_division=0),
        'Recall': recall_score(y_test, preds, zero_division=0),
        'F1-Score': f1_score(y_test, preds, zero_division=0),
        'ROC-AUC': roc_auc_score(y_test, probs) if sum(probs) > 0 else 0
    }

    logger.info(f"--- {name} Metrics ---")
    for k, v in metrics.items():
        logger.info(f"{k}: {v:.4f}")

    # Save reports
    save_confusion_matrix(name, y_test, preds)
    save_feature_importance(name, model, X_test.columns)

    return metrics, preds


# ==============================
# Training
# ==============================
def train_and_evaluate():
    create_reports_dir()
    X_train, X_test, y_train, y_test = load_processed_data()

    # 1. Logistic Regression
    logger.info("Training Logistic Regression...")
    lr_params = {'C': [0.1, 1.0, 10.0], 'class_weight': ['balanced']}
    lr_grid = GridSearchCV(LogisticRegression(max_iter=1000), lr_params, cv=5, scoring='f1')
    lr_grid.fit(X_train, y_train)

    logger.info(f"Best LR Params: {lr_grid.best_params_}")
    lr_metrics, _ = evaluate_model("LogisticRegression", lr_grid.best_estimator_, X_test, y_test)

    # 2. Random Forest
    logger.info("Training Random Forest...")
    rf_params = {
        'n_estimators': [50, 100],
        'max_depth': [10, 20, None],
        'class_weight': ['balanced']
    }

    rf_grid = GridSearchCV(RandomForestClassifier(random_state=42), rf_params, cv=5, scoring='f1')
    rf_grid.fit(X_train, y_train)

    logger.info(f"Best RF Params: {rf_grid.best_params_}")
    rf_metrics, _ = evaluate_model("RandomForest", rf_grid.best_estimator_, X_test, y_test)

    # Best model selection
    if rf_metrics['F1-Score'] > lr_metrics['F1-Score']:
        best_model = rf_grid.best_estimator_
        best_name = "RandomForest"
    else:
        best_model = lr_grid.best_estimator_
        best_name = "LogisticRegression"

    logger.info(f"Best overall model is {best_name}. Saving to disk...")
    save_model(best_model, os.path.join('models', 'best_model.pkl'))


if __name__ == '__main__':
    train_and_evaluate()