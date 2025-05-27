from fastapi import APIRouter, HTTPException, Request
import logging
from datetime import datetime
from Application.Dtos.predict import PredictionRequestDto, PredictionResponseDto
from Application.services.ml_model_services import analyze_prediction

# Set up proper logging
logger = logging.getLogger(__name__)

# Initialize FastAPI router with versioning
router = APIRouter(prefix="/api/ml", tags=["ML"])

@router.post("/predict", response_model=PredictionResponseDto)
async def predict(payload: PredictionRequestDto, request: Request):
    """
    Endpoint to predict hours until the next watering is needed based on sensor readings and plant information.

    Args:
        payload (PredictionRequestDto): The request body containing sensor readings and plant growth stage information.
        request (Request): The HTTP request object, used to extract client information.

    Returns:
        PredictionResponseDto: An object containing prediction time and hours until next watering.

    Raises:
        HTTPException: If the prediction fails or an error occurs during processing.
    """
    try:
        # Log the incoming request
        client_ip = request.client.host if request.client else "unknown"
        logger.info(f"Received prediction request from {client_ip} for plant stage: {payload.plantGrowthStage}")

        # Process the prediction
        result = await analyze_prediction(payload)

        # Check if the prediction was successful and log model version for diagnostics
        logger.info(f"Successful prediction: {result.HoursUntilNextWatering:.2f} hours using model {getattr(result, 'modelVersion', 'unknown')}")
        
        # Return response with proper field names matching C# conventions
        return PredictionResponseDto(
            PredictionTime=result.PredictionTime,
            HoursUntilNextWatering=result.HoursUntilNextWatering
        )

    except Exception as e:
        # Log the error properly
        logger.error(f"Error processing prediction: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="An error occurred while processing the prediction"
        )