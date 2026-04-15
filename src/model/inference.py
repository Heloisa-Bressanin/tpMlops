"""
Inference module for NYC Taxi Trip Duration prediction.
Handles loading the model and making predictions.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List
import yaml
from src.data.preprocessing import DataPreprocessor
from src.model.training import ModelTrainer


class TaxiDurationPredictor:
    """Make predictions for taxi trip duration."""
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        """Initialize predictor with configuration."""
        self.config = self._load_config(config_path)
        self.preprocessor = DataPreprocessor(config_path)
        self.trainer = ModelTrainer(config_path)
        self.model = None
        
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def load_model(self):
        """Load the trained model."""
        self.model = self.trainer.load_model()
        return self.model
    
    def predict_from_raw(self, data: pd.DataFrame) -> np.ndarray:
        """
        Make predictions from raw data.
        Preprocesses the data and returns predictions.
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        # Preprocess
        processed_data = self.preprocessor.extract_datetime_features(data)
        processed_data = self.preprocessor.engineer_features(processed_data)

        # Add columns that were present during training but are not generated
        # by the preprocessing pipeline (they were NaN at training time too).
        expected_features = self.trainer.feature_columns or []
        for col in expected_features:
            if col not in processed_data.columns:
                processed_data[col] = np.nan

        # Prepare features
        X, _ = self.trainer.prepare_features(processed_data, is_train=False)

        # Predict
        predictions = self.trainer.predict(X)
        return predictions
    
    def predict_single(self, sample: dict) -> float:
        """
        Make prediction for a single sample.
        
        Args:
            sample: Dictionary with required features
                   {
                       'pickup_datetime': '2016-03-14 17:24:55',
                       'pickup_latitude': 40.7614,
                       'pickup_longitude': -73.9776,
                       'dropoff_latitude': 40.7489,
                       'dropoff_longitude': -73.9680,
                       'passenger_count': 1
                   }
        
        Returns:
            Predicted trip duration in seconds
        """
        df = pd.DataFrame([sample])
        predictions = self.predict_from_raw(df)
        return predictions[0]
    
    def predict_batch(self, samples: List[dict]) -> np.ndarray:
        """
        Make predictions for multiple samples.
        
        Args:
            samples: List of dictionaries with required features
        
        Returns:
            Array of predicted trip durations in seconds
        """
        df = pd.DataFrame(samples)
        predictions = self.predict_from_raw(df)
        return predictions


def validate_inference():
    """Validate inference with sample predictions."""
    print("=" * 60)
    print("NYC TAXI TRIP DURATION - INFERENCE VALIDATION")
    print("=" * 60)
    
    predictor = TaxiDurationPredictor()
    
    # Load model
    print("\nLoading model...")
    predictor.load_model()
    print("✓ Model loaded successfully")
    
    # Sample predictions
    print("\nMaking sample predictions...")
    
    samples = [
        {
            'pickup_datetime': '2016-03-14 17:24:55',
            'pickup_latitude': 40.7614,
            'pickup_longitude': -73.9776,
            'dropoff_latitude': 40.7489,
            'dropoff_longitude': -73.9680,
            'passenger_count': 1
        },
        {
            'pickup_datetime': '2016-03-14 10:15:30',
            'pickup_latitude': 40.7505,
            'pickup_longitude': -73.9972,
            'dropoff_latitude': 40.7614,
            'dropoff_longitude': -73.9776,
            'passenger_count': 2
        },
        {
            'pickup_datetime': '2016-03-14 22:50:00',
            'pickup_latitude': 40.6501,
            'pickup_longitude': -73.9496,
            'dropoff_latitude': 40.7589,
            'dropoff_longitude': -73.9851,
            'passenger_count': 3
        }
    ]
    
    predictions = predictor.predict_batch(samples)
    
    print("\nSample Predictions:")
    print("-" * 60)
    for i, (sample, pred) in enumerate(zip(samples, predictions), 1):
        print(f"\nSample {i}:")
        print(f"  Pickup:   {sample['pickup_datetime']}")
        print(f"  Location: ({sample['pickup_latitude']:.4f}, {sample['pickup_longitude']:.4f})")
        print(f"  Dropoff:  ({sample['dropoff_latitude']:.4f}, {sample['dropoff_longitude']:.4f})")
        print(f"  Passengers: {sample['passenger_count']}")
        print(f"  Predicted Duration: {pred:.0f} seconds ({pred/60:.1f} minutes)")
    
    print("\n" + "=" * 60)
    print("✓ Inference validation completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    validate_inference()
