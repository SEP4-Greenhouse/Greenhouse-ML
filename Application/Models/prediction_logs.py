# app/Models/prediction_log.py

from sqlalchemy import Column, Integer, String, Float, DateTime
from Application.database import Base

# SQLAlchemy model representing a prediction log entry in the database
class PredictionLog(Base):
    __tablename__ = "prediction_logs"  # Table name in the database

    id = Column(Integer, primary_key=True, index=True)  # Auto-incrementing ID
    timestamp = Column(DateTime)  # Timestamp when prediction was made
    sensorType = Column(String)  # Type of sensor (e.g., humidity, temperature)
    value = Column(Float)  # Sensor value recorded
    status = Column(String)  # Status result (e.g., normal, warning)
    suggestion = Column(String)  # Suggested action based on prediction
    trendAnalysis = Column(String, nullable=True)  # Optional trend analysis description
