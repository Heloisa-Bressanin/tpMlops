"""
FastAPI application for NYC Taxi Trip Duration prediction
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from sqlalchemy import func
from pathlib import Path
import sys
import pandas as pd
from datetime import datetime

# Add parent directory to path to import src modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model.inference import TaxiDurationPredictor
from api.schemas import TripPredictionRequest, TripPredictionResponse, PredictionStats
from api.database import get_db, Prediction, engine

# Global model instance
predictor = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup, clean up on shutdown."""
    global predictor
    project_root = Path(__file__).parent.parent
    try:
        predictor = TaxiDurationPredictor(config_path=str(project_root / 'configs/config.yaml'))
        predictor.trainer.processed_path = project_root / 'data/processed'
        predictor.trainer.model_dir = project_root / 'models'
        predictor.load_model()
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        raise
    yield


# Initialize FastAPI app
app = FastAPI(
    title="NYC Taxi Trip Duration Prediction API",
    description="API for predicting NYC taxi trip duration using a trained ML model",
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["Health"])
async def root():
    """Root endpoint - health check"""
    return {
        "message": "NYC Taxi Trip Duration Prediction API",
        "status": "running",
        "docs": "/docs"
    }


@app.post("/predict", response_model=TripPredictionResponse, tags=["Predictions"])
async def predict_trip_duration(
    request: TripPredictionRequest,
    db: Session = Depends(get_db)
):
    """
    Predict trip duration for a single trip
    
    **Input parameters:**
    - pickup_datetime: Pickup time (YYYY-MM-DD HH:MM:SS format)
    - pickup_latitude: Pickup latitude (40.5-40.9)
    - pickup_longitude: Pickup longitude (-74.3 to -73.7)
    - dropoff_latitude: Dropoff latitude (40.5-40.9)
    - dropoff_longitude: Dropoff longitude (-74.3 to -73.7)
    - passenger_count: Number of passengers (≥1)
    - vendor_id: Vendor ID (1 or 2)
    
    **Returns:**
    - prediction_id: Database ID of the prediction
    - predicted_duration_seconds: Predicted duration in seconds
    - predicted_duration_minutes: Predicted duration in minutes
    - input_data: Echoed input data
    - timestamp: Prediction timestamp
    """
    global predictor
    
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        predictions = predictor.predict_batch([{
            'pickup_datetime': request.pickup_datetime,
            'pickup_latitude': request.pickup_latitude,
            'pickup_longitude': request.pickup_longitude,
            'dropoff_latitude': request.dropoff_latitude,
            'dropoff_longitude': request.dropoff_longitude,
            'passenger_count': request.passenger_count,
            'vendor_id': request.vendor_id,
            'trip_duration': 0
        }])
        
        predicted_duration_seconds = float(predictions[0])
        predicted_duration_minutes = predicted_duration_seconds / 60
        
        # Store in database
        db_prediction = Prediction(
            timestamp=datetime.utcnow(),
            pickup_datetime=request.pickup_datetime,
            pickup_latitude=request.pickup_latitude,
            pickup_longitude=request.pickup_longitude,
            dropoff_latitude=request.dropoff_latitude,
            dropoff_longitude=request.dropoff_longitude,
            passenger_count=request.passenger_count,
            vendor_id=request.vendor_id,
            predicted_duration_seconds=predicted_duration_seconds
        )
        db.add(db_prediction)
        db.commit()
        db.refresh(db_prediction)
        
        return TripPredictionResponse(
            prediction_id=db_prediction.id,
            predicted_duration_seconds=predicted_duration_seconds,
            predicted_duration_minutes=predicted_duration_minutes,
            input_data=request,
            timestamp=db_prediction.timestamp
        )
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")


@app.post("/predict-batch", tags=["Predictions"])
async def predict_batch(
    requests: list[TripPredictionRequest],
    db: Session = Depends(get_db)
):
    """
    Predict trip duration for multiple trips
    
    Returns a list of predictions
    """
    global predictor
    
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert requests to list of dicts
        samples = []
        for req in requests:
            samples.append({
                'pickup_datetime': req.pickup_datetime,
                'pickup_latitude': req.pickup_latitude,
                'pickup_longitude': req.pickup_longitude,
                'dropoff_latitude': req.dropoff_latitude,
                'dropoff_longitude': req.dropoff_longitude,
                'passenger_count': req.passenger_count,
                'vendor_id': req.vendor_id,
                'trip_duration': 0
            })
        
        # Make predictions
        predictions = predictor.predict_batch(samples)
        
        # Store all predictions in database
        results = []
        for req, pred_duration in zip(requests, predictions):
            db_prediction = Prediction(
                timestamp=datetime.utcnow(),
                pickup_datetime=req.pickup_datetime,
                pickup_latitude=req.pickup_latitude,
                pickup_longitude=req.pickup_longitude,
                dropoff_latitude=req.dropoff_latitude,
                dropoff_longitude=req.dropoff_longitude,
                passenger_count=req.passenger_count,
                vendor_id=req.vendor_id,
                predicted_duration_seconds=float(pred_duration)
            )
            db.add(db_prediction)
            results.append({
                "predicted_duration_seconds": float(pred_duration),
                "predicted_duration_minutes": float(pred_duration) / 60
            })
        
        db.commit()
        return results
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Batch prediction failed: {str(e)}")


@app.get("/stats", response_model=PredictionStats, tags=["Statistics"])
async def get_prediction_stats(db: Session = Depends(get_db)):
    """
    Get statistics about all predictions made
    
    Returns:
    - total_predictions: Total number of predictions
    - average_duration_seconds: Average predicted duration
    - min_duration_seconds: Minimum predicted duration
    - max_duration_seconds: Maximum predicted duration
    """
    try:
        total = db.query(func.count(Prediction.id)).scalar() or 0
        
        if total == 0:
            return PredictionStats(
                total_predictions=0,
                average_duration_seconds=0.0,
                min_duration_seconds=0.0,
                max_duration_seconds=0.0
            )
        
        avg = db.query(func.avg(Prediction.predicted_duration_seconds)).scalar() or 0
        min_val = db.query(func.min(Prediction.predicted_duration_seconds)).scalar() or 0
        max_val = db.query(func.max(Prediction.predicted_duration_seconds)).scalar() or 0
        
        return PredictionStats(
            total_predictions=total,
            average_duration_seconds=float(avg),
            min_duration_seconds=float(min_val),
            max_duration_seconds=float(max_val)
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")


@app.get("/history", tags=["Statistics"])
async def get_prediction_history(
    limit: int = 10,
    db: Session = Depends(get_db)
):
    """
    Get recent prediction history
    
    Parameters:
    - limit: Maximum number of recent predictions to return (default: 10)
    
    Returns list of recent predictions
    """
    try:
        predictions = db.query(Prediction)\
            .order_by(Prediction.timestamp.desc())\
            .limit(limit)\
            .all()
        
        return [
            {
                "id": p.id,
                "timestamp": p.timestamp,
                "pickup_datetime": p.pickup_datetime,
                "pickup_location": (p.pickup_latitude, p.pickup_longitude),
                "dropoff_location": (p.dropoff_latitude, p.dropoff_longitude),
                "passenger_count": p.passenger_count,
                "predicted_duration_minutes": p.predicted_duration_seconds / 60
            }
            for p in predictions
        ]
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get history: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
