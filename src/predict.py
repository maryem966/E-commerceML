import os
import pandas as pd
from utils import load_model, logger
from preprocessing import handle_outliers, feature_engineering

class ChurnPredictor:
    def __init__(self):
        self.preprocessor = load_model(os.path.join('models', 'preprocessor_pipeline.pkl'))
        self.meta = load_model(os.path.join('models', 'features_meta.pkl'))
        self.model = load_model(os.path.join('models', 'best_model.pkl'))
        
    def prepare_input(self, raw_df):
        df_clean = handle_outliers(raw_df)
        df_clean = feature_engineering(df_clean)
        
        # Ensure we have all the columns required by preprocessor
        expected_num = self.meta['num_cols']
        expected_cat = self.meta['cat_cols']
        
        # Add missing categorical as 'Unknown' / missing num as 0 if they don't exist
        for col in expected_num:
            if col not in df_clean.columns:
                df_clean[col] = 0
                
        for col in expected_cat:
            if col not in df_clean.columns:
                df_clean[col] = 'Unknown'
                
        # Drop highly correlated ones that were removed in training
        dropped = self.meta['dropped_cols']
        df_clean = df_clean.drop(columns=[c for c in dropped if c in df_clean.columns])
        
        # Ordering columns based on original training definition is not strictly required since ColumnTransformer grabs by name,
        # but to be safe, we just pass the dataframe with all the needed names.
        return df_clean

    def predict(self, raw_df):
        logger.info("Preparing data for prediction...")
        prepared_df = self.prepare_input(raw_df)
        
        logger.info("Transforming via pipeline...")
        X_processed = self.preprocessor.transform(prepared_df)
        
        logger.info("Making predictions...")
        preds = self.model.predict(X_processed)
        probs = self.model.predict_proba(X_processed)[:, 1] if hasattr(self.model, "predict_proba") else [0]*len(preds)
        
        return preds, probs

if __name__ == "__main__":
    # Test batch prediction with existing raw file
    from utils import load_data
    df_raw = load_data().head(10) # 10 samples
    predictor = ChurnPredictor()
    predictions, probabilities = predictor.predict(df_raw)
    
    for i in range(len(predictions)):
        print(f"Customer {i+1} -> Churn: {predictions[i]} | Prob: {probabilities[i]:.4f}")
