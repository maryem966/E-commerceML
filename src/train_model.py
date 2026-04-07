import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from utils import save_model, logger

def load_processed_data():
    logger.info("Loading train/test splits...")
    X_train = pd.read_csv(os.path.join('data', 'train_test', 'X_train.csv'))
    X_test = pd.read_csv(os.path.join('data', 'train_test', 'X_test.csv'))
    y_train = pd.read_csv(os.path.join('data', 'train_test', 'y_train.csv')).values.ravel()
    y_test = pd.read_csv(os.path.join('data', 'train_test', 'y_test.csv')).values.ravel()
    return X_train, X_test, y_train, y_test

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
    return metrics, preds

def train_and_evaluate():
    X_train, X_test, y_train, y_test = load_processed_data()
    
    # 1. Logistic Regression
    logger.info("Training Logistic Regression...")
    lr_params = {'C': [0.1, 1.0, 10.0], 'class_weight': ['balanced']}
    lr_grid = GridSearchCV(LogisticRegression(max_iter=1000), lr_params, cv=5, scoring='f1')
    lr_grid.fit(X_train, y_train)
    logger.info(f"Best LR Params: {lr_grid.best_params_}")
    lr_metrics, _ = evaluate_model("Logistic Regression", lr_grid.best_estimator_, X_test, y_test)
    
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
    rf_metrics, _ = evaluate_model("Random Forest", rf_grid.best_estimator_, X_test, y_test)
    
    # Select the very best Model
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
