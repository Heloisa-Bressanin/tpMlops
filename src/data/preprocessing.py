"""
Data preprocessing module for NYC Taxi Trip Duration prediction.
Handles data loading, cleaning, and feature engineering.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple
import yaml


class DataPreprocessor:
    """Preprocess NYC taxi data."""
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        """Initialize preprocessor with configuration."""
        self.config = self._load_config(config_path)
        self.raw_path = Path(self.config["data"].get("raw_path", "data/raw"))  # Optional fallback
        self.processed_path = Path(self.config["data"]["processed_path"])
        
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def load_data(self, filename: str) -> pd.DataFrame:
        """Load CSV data from raw folder."""
        file_path = self.raw_path / filename
        print(f"Loading data from {file_path}")
        return pd.read_csv(file_path)
    
    def extract_datetime_features(self, df: pd.DataFrame, date_column: str = "pickup_datetime") -> pd.DataFrame:
        """Extract datetime features from pickup_datetime."""
        df = df.copy()
        df[date_column] = pd.to_datetime(df[date_column])
        
        df['year'] = df[date_column].dt.year
        df['month'] = df[date_column].dt.month
        df['day'] = df[date_column].dt.day
        df['hour'] = df[date_column].dt.hour
        df['minute'] = df[date_column].dt.minute
        df['day_of_week'] = df[date_column].dt.dayofweek
        
        return df
    
    def calculate_distance(self, lat1, lon1, lat2, lon2):
        """Calculate haversine distance in kilometers."""
        R = 6371  # Earth radius in km
        
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        
        return R * c
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create new features from existing ones."""
        df = df.copy()
        
        # Distance features
        df['distance'] = self.calculate_distance(
            df['pickup_latitude'],
            df['pickup_longitude'],
            df['dropoff_latitude'],
            df['dropoff_longitude']
        )
        
        # Manhattan distance (simplified)
        df['manhattan_distance'] = (
            np.abs(df['pickup_latitude'] - df['dropoff_latitude']) +
            np.abs(df['pickup_longitude'] - df['dropoff_longitude'])
        )
        
        # Direction features
        df['direction_ns'] = df['dropoff_latitude'] - df['pickup_latitude']
        df['direction_ew'] = df['dropoff_longitude'] - df['pickup_longitude']
        
        return df
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean data by removing invalid entries."""
        df = df.copy()
        
        # Remove rows with trip_duration <= 0
        if 'trip_duration' in df.columns:
            df = df[df['trip_duration'] > 0]
        
        # Remove rows with invalid coordinates
        df = df[
            (df['pickup_latitude'] >= 40.5) & (df['pickup_latitude'] <= 40.9) &
            (df['dropoff_latitude'] >= 40.5) & (df['dropoff_latitude'] <= 40.9) &
            (df['pickup_longitude'] >= -74.3) & (df['pickup_longitude'] <= -73.7) &
            (df['dropoff_longitude'] >= -74.3) & (df['dropoff_longitude'] <= -73.7)
        ]
        
        # Remove rows with passenger count <= 0
        if 'passenger_count' in df.columns:
            df = df[df['passenger_count'] > 0]
        
        return df
    
    def preprocess(self, filename: str, is_train: bool = True) -> pd.DataFrame:
        """Complete preprocessing pipeline."""
        # Load data
        df = self.load_data(filename)
        print(f"Loaded data shape: {df.shape}")
        
        # Extract datetime features
        df = self.extract_datetime_features(df)
        print("Extracted datetime features")
        
        # Clean data
        if is_train:
            df = self.clean_data(df)
            print(f"Cleaned data shape: {df.shape}")
        
        # Engineer features
        df = self.engineer_features(df)
        print("Engineered features")
        
        return df
    
    def save_processed_data(self, df: pd.DataFrame, filename: str) -> None:
        """Save processed data to CSV."""
        output_path = self.processed_path / filename
        df.to_csv(output_path, index=False)
        print(f"Saved processed data to {output_path}")


def main():
    """Run preprocessing pipeline."""
    preprocessor = DataPreprocessor()
    
    # Preprocess training data
    train_df = preprocessor.preprocess("train.csv", is_train=True)
    preprocessor.save_processed_data(train_df, "train_processed.csv")
    
    # Preprocess test data
    test_df = preprocessor.preprocess("test.csv", is_train=False)
    preprocessor.save_processed_data(test_df, "test_processed.csv")
    
    print("\nPreprocessing completed successfully!")


if __name__ == "__main__":
    main()
