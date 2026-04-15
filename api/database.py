"""
Database configuration and models for storing predictions
"""

from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime
from sqlalchemy.orm import DeclarativeBase, sessionmaker
from datetime import datetime
from pathlib import Path

# Database setup - store in project root
db_path = Path(__file__).parent.parent / "predictions.db"
DATABASE_URL = f"sqlite:///{db_path}"

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},
    echo=False
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

class Base(DeclarativeBase):
    pass


class Prediction(Base):
    """Database model for storing predictions"""
    
    __tablename__ = "predictions"
    
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    
    # Input data
    pickup_datetime = Column(String, index=True)
    pickup_latitude = Column(Float)
    pickup_longitude = Column(Float)
    dropoff_latitude = Column(Float)
    dropoff_longitude = Column(Float)
    passenger_count = Column(Integer)
    vendor_id = Column(Integer)
    
    # Prediction result
    predicted_duration_seconds = Column(Float, index=True)


# Create tables
Base.metadata.create_all(bind=engine)


def get_db():
    """Get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
