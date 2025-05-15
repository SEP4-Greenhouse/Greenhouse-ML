# Application/Api/predictions.py

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from Application.Models.prediction_log import PredictionLog
from Application.database import get_db

router = APIRouter()

@router.get("/predictions")
def get_predictions(db: Session = Depends(get_db)):
    """
    Retrieve all prediction entries from the database.

    This endpoint returns a list of all prediction logs,
    sorted by timestamp in descending order (latest first).

    Parameters:
    - db: SQLAlchemy session, injected via FastAPI's dependency system.

    Returns:
    - A list of dictionaries, each representing a prediction log.
    """
    # Query all prediction entries and sort by newest first
    predictions = db.query(PredictionLog).order_by(PredictionLog.timestamp.desc()).all()

    # Format each entry as a dictionary for API response
    return [
        {
            "timestamp": p.timestamp,
            "sensorType": p.sensorType,
            "value": p.value,
            "status": p.status,
            "suggestion": p.suggestion,
            "trendAnalysis": p.trendAnalysis,
        }
        for p in predictions
    ]
