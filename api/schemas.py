"""
Pydantic schemas for API request/response validation
"""

from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


class TripPredictionRequest(BaseModel):
    """Schema for trip duration prediction request"""

    model_config = {
        "json_schema_extra": {
            "example": {
                "pickup_datetime": "2016-03-14 17:24:55",
                "pickup_latitude": 40.7614,
                "pickup_longitude": -73.9776,
                "dropoff_latitude": 40.7489,
                "dropoff_longitude": -73.9680,
                "passenger_count": 1,
                "vendor_id": 1,
            }
        }
    }

    pickup_datetime: str = Field(..., description="Pickup datetime in format YYYY-MM-DD HH:MM:SS")
    pickup_latitude: float = Field(..., description="Pickup latitude", ge=40.5, le=40.9)
    pickup_longitude: float = Field(..., description="Pickup longitude", ge=-74.3, le=-73.7)
    dropoff_latitude: float = Field(..., description="Dropoff latitude", ge=40.5, le=40.9)
    dropoff_longitude: float = Field(..., description="Dropoff longitude", ge=-74.3, le=-73.7)
    passenger_count: int = Field(..., description="Number of passengers", ge=1)
    vendor_id: int = Field(..., description="Vendor ID", ge=1, le=2)


class TripPredictionResponse(BaseModel):
    """Schema for trip duration prediction response"""

    model_config = {
        "json_schema_extra": {
            "example": {
                "prediction_id": 1,
                "predicted_duration_seconds": 900.5,
                "predicted_duration_minutes": 15.0,
                "input_data": {
                    "pickup_datetime": "2016-03-14 17:24:55",
                    "pickup_latitude": 40.7614,
                    "pickup_longitude": -73.9776,
                    "dropoff_latitude": 40.7489,
                    "dropoff_longitude": -73.9680,
                    "passenger_count": 1,
                    "vendor_id": 1,
                },
                "timestamp": "2024-04-15T10:30:00",
            }
        }
    }

    prediction_id: int = Field(..., description="Unique prediction ID")
    predicted_duration_seconds: float = Field(..., description="Predicted trip duration in seconds")
    predicted_duration_minutes: float = Field(..., description="Predicted trip duration in minutes")
    input_data: TripPredictionRequest = Field(..., description="Input data used for prediction")
    timestamp: datetime = Field(..., description="Prediction timestamp")


class PredictionStats(BaseModel):
    """Schema for prediction statistics"""
    
    total_predictions: int = Field(..., description="Total number of predictions made")
    average_duration_seconds: float = Field(..., description="Average predicted duration")
    min_duration_seconds: float = Field(..., description="Minimum predicted duration")
    max_duration_seconds: float = Field(..., description="Maximum predicted duration")
