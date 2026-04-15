"""
Model training module for NYC Taxi Trip Duration prediction.
Handles model training and serialization.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import json
import yaml
from typing import Tuple, Dict, Any


class ModelTrainer:
    """Train and manage NYC taxi trip duration model."""
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        """Initialize trainer with configuration."""
        self.config = self._load_config(config_path)
        self.processed_path = Path(self.config["data"]["processed_path"])
        self.model_dir = Path(self.config["model"]["model_dir"])
        self.model = None
        self.feature_columns = None  # Store feature column names and order
        
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def load_processed_data(self, filename: str) -> pd.DataFrame:
        """Load processed data from CSV."""
        file_path = self.processed_path / filename
        print(f"Loading data from {file_path}")
        return pd.read_csv(file_path)
    
    def prepare_features(self, df: pd.DataFrame, is_train: bool = True) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare features and target for training."""
        df = df.copy()
        
        # Convert numeric columns to float explicitly to avoid type mismatches
        for col in df.columns:
            if col not in ["pickup_datetime", "trip_duration", "id"]:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Columns to drop (include datetime columns)
        drop_cols = self.config["preprocessing"]["drop_columns"] + ["pickup_datetime"]
        drop_cols = [col for col in drop_cols if col in df.columns]
        
        # Also drop datetime-related objects and non-numeric columns
        for col in df.columns:
            if df[col].dtype == "object":
                drop_cols.append(col)
        
        drop_cols = list(set(drop_cols))  # Remove duplicates
        
        # Select features - only numeric, non-empty columns
        feature_cols = [col for col in df.columns
                       if col not in drop_cols and col != "trip_duration"
                       and df[col].dtype != "object"
                       and df[col].notna().any()]  # drop fully-NaN columns

        # Sort features to ensure consistent ordering
        feature_cols = sorted(feature_cols)
        
        X = df[feature_cols].astype(float)
        
        if is_train:
            # Store feature columns for later use in predictions
            self.feature_columns = feature_cols
            y = df["trip_duration"]
            print(f"Features shape: {X.shape}")
            print(f"Target shape: {y.shape}")
            print(f"Feature columns: {feature_cols}")
            return X, y
        else:
            print(f"Features shape: {X.shape}")
            # Reorder columns to match training data
            if self.feature_columns is not None:
                print(f"Reordering columns from {list(X.columns)} to {self.feature_columns}")
                X = X[self.feature_columns]
            else:
                print(f"⚠️  feature_columns not set, keeping original order: {list(X.columns)}")
            return X, None
    
    def train(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Train the Random Forest model."""
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.config["preprocessing"]["test_size"],
            random_state=self.config["preprocessing"]["random_state"]
        )
        
        print(f"\nTraining set size: {X_train.shape}")
        print(f"Test set size: {X_test.shape}")
        
        # Initialize and train model
        model_config = self.config["training"]
        self.model = RandomForestRegressor(
            n_estimators=model_config["n_estimators"],
            max_depth=model_config["max_depth"],
            min_samples_split=model_config["min_samples_split"],
            min_samples_leaf=model_config["min_samples_leaf"],
            random_state=model_config["random_state"],
            n_jobs=-1
        )
        
        print("\nTraining Random Forest model...")
        self.model.fit(X_train, y_train)
        print("Training completed!")
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        
        metrics = {
            "mse": mean_squared_error(y_test, y_pred),
            "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
            "mae": mean_absolute_error(y_test, y_pred),
            "r2": r2_score(y_test, y_pred)
        }
        
        print(f"\nModel Performance:")
        print(f"  MSE:  {metrics['mse']:.4f}")
        print(f"  RMSE: {metrics['rmse']:.4f}")
        print(f"  MAE:  {metrics['mae']:.4f}")
        print(f"  R²:   {metrics['r2']:.4f}")
        
        return metrics
    
    def save_model(self, model_name: str = None) -> None:
        """Save trained model to disk."""
        if self.model is None:
            raise ValueError("No model to save. Train a model first.")
        
        if model_name is None:
            model_name = self.config["model"]["model_name"]
        
        model_path = self.model_dir / model_name
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        joblib.dump(self.model, model_path)
        print(f"Model saved to {model_path}")
        
        # Also save feature columns metadata
        metadata_path = self.model_dir / "model_metadata.json"
        metadata = {"feature_columns": self.feature_columns}
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f)
        print(f"Model metadata saved to {metadata_path}")
    
    def load_model(self, model_name: str = None):
        """Load trained model from disk."""
        if model_name is None:
            model_name = self.config["model"]["model_name"]
        
        model_path = self.model_dir / model_name
        self.model = joblib.load(model_path)
        print(f"Model loaded from {model_path}")
        
        # Also try to load feature columns metadata
        metadata_path = self.model_dir / "model_metadata.json"
        print(f"Looking for metadata at: {metadata_path} (exists: {metadata_path.exists()})")
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                self.feature_columns = metadata.get("feature_columns")
            print(f"✓ Model metadata loaded ({len(self.feature_columns)} features)")
        else:
            print(f"⚠️  Metadata not found at {metadata_path}")
        
        return self.model
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions with the trained model."""
        if self.model is None:
            raise ValueError("No model loaded. Train or load a model first.")
        
        # Ensure columns are in the correct order
        if self.feature_columns is not None and isinstance(X, pd.DataFrame):
            X = X[self.feature_columns]
            # Convert to numpy array to avoid sklearn column name issues
            X = X.values
        
        return self.model.predict(X)


def main():
    """Run complete training pipeline."""
    trainer = ModelTrainer()
    
    # Load processed data
    train_df = trainer.load_processed_data("train_processed.csv")
    
    # Prepare features
    X, y = trainer.prepare_features(train_df, is_train=True)
    
    # Train model
    metrics = trainer.train(X, y)
    
    # Save model
    trainer.save_model()
    
    print("\nTraining pipeline completed successfully!")


if __name__ == "__main__":
    main()
