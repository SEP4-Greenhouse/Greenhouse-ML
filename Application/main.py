# Application/main.py

from fastapi import FastAPI
from Application.Api import predict, sensor  # Import API route modules
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Access configuration variables
PORT = os.getenv("PORT", "8000")  # Default port 8000
DEBUG = os.getenv("DEBUG", "False").lower() == "true"

# Initialize FastAPI app
app = FastAPI(debug=DEBUG)

# Register route modules
app.include_router(predict.router)      # Handles POST /predict
app.include_router(sensor.router)       # Handles POST /sensor

