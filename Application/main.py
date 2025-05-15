from fastapi import FastAPI
from Application.Api import predict, sensor, predictions  # Import API route modules
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Access configuration variables
PORT = os.getenv("PORT", "8000")  # Default to port 8000 if not specified
DEBUG = os.getenv("DEBUG", "False").lower() == "true"

# Initialize FastAPI application with debug mode if enabled
app = FastAPI(debug=DEBUG)

# Register API routers
# Handles POST /predict (ML analysis based on sensor + history)
Application.include_router(predict.router)

# Handles POST /sensor (live sensor data without history)
Application.include_router(sensor.router)

# Handles GET /predictions (returns logs from predictions.csv)
Application.include_router(predictions.router)
