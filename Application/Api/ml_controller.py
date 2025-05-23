from fastapi import APIRouter, HTTPException, Request
from Application.Dtos.predict import PredictionRequestDto, PredictionResultDto 
from Application.services.ml_model_services import analyze_prediction

# Initialize FastAPI router with versioning
router = APIRouter(prefix="/api/v1/ml", tags=["ML"])

@router.post("/predict", response_model=PredictionResultDto)
async def predict(payload: PredictionRequestDto, request: Request):
    """
    Endpoint to predict hours until the next watering is needed based on sensor readings and plant information.

    Args:
        payload (PredictionRequestDto): The request body containing sensor readings and plant growth stage information.
        request (Request): The HTTP request object, used to extract client information.

    Returns:
        PredictionResultDto: An object containing the prediction result, including the timestamp, predicted hours until watering, and model version.

    Raises:
        HTTPException: If the prediction fails or an error occurs during processing.

    Logs:
        - Logs the client IP and plant growth stage for each request.
        - Logs success or failure of the prediction process.
        - Logs errors encountered during processing.
    """
    try:
        # Log the incoming request
        client_ip = request.client.host if request.client else "unknown"
        print(f"[ML_API] Received prediction request from {client_ip} for plant stage: {payload.plantGrowthStage}")

        # Process the prediction
        result = await analyze_prediction(payload)

        # Check if the prediction was successful
        if result.predictedHoursUntilWatering < 0:
            print(f"[ML_API] Prediction failed for plant stage: {payload.plantGrowthStage}")
            raise HTTPException(
                status_code=500,
                detail="Prediction failed. Check server logs for details."
            )

        print(f"[ML_API] Successful prediction: {result.predictedHoursUntilWatering:.2f} hours using model {getattr(result, 'modelVersion', 'unknown')}")
        return result

    except Exception as e:
        # Log the error (in production, use proper logging)
        print(f"[ML_CONTROLLER] Error processing prediction: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="An error occurred while processing the prediction"
        )
