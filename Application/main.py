import os
import glob
from fastapi import FastAPI
from contextlib import asynccontextmanager
from Application.api.ml_controller import router as ml_router
from fastapi.middleware.cors import CORSMiddleware


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Call model availability check during startup
    check_model_availability()
    print("✅ [STARTUP] Prediction service is starting...")
    yield
    print("🛑 [SHUTDOWN] App is shutting down...")

app = FastAPI(
    title="Greenhouse ML API",
    description="🚀 Predict optimal watering time based on sensor data",
    version="1.0.0",
    lifespan=lifespan
)

# At app startup (e.g., in main.py)
def check_model_availability():
    """Check if ML model is available at startup."""
    model_files = glob.glob(os.path.join("Application/trained_models", "reg_model_*.pkl"))
    if not model_files:
        print("[STARTUP WARNING] No ML model files found. Predictions will use fallback logic.")
    else:
        print(f"[STARTUP] Found {len(model_files)} ML model(s). Using: {os.path.basename(max(model_files, key=os.path.getctime))}")

# Add CORS middleware for API access from other services
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed service URLs
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register the ML prediction router
app.include_router(ml_router)

# Simple health check endpoint for Docker/k8s
@app.get("/health", tags=["Health"])
async def health_check():
    return {"status": "ok"}