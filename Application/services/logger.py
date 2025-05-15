# Application/Services/logger.py

from Application.Models.prediction_log import PredictionLog
from Application.schema.predict import SensorData, PredictionResult
from sqlalchemy.orm import Session

def log_prediction(sensor: SensorData, result: PredictionResult, db: Session):
    """
    Logs a prediction result to the database using SQLAlchemy ORM.

    Parameters:
    - sensor (SensorData): The live sensor data used for prediction.
    - result (PredictionResult): The outcome of the prediction logic.
    - db (Session): The SQLAlchemy database session (injected from FastAPI Depends).

    This function creates a new PredictionLog database entry and commits it.
    """
    log = PredictionLog(
        timestamp=result.timestamp,
        sensorType=sensor.sensorType,
        value=sensor.value,
        status=result.status,
        suggestion=result.suggestion,
        trendAnalysis=result.trendAnalysis
    )

    # Add the log to the current session and persist to database
    db.add(log)
    db.commit()
