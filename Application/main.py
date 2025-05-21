from fastapi import FastAPI
from contextlib import asynccontextmanager
from Application.api.ml_service import router as ml_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[STARTUP] Starting prediction scheduler...")
    yield
    print("[SHUTDOWN] App is closing...")

app = FastAPI(lifespan=lifespan)

app.include_router(ml_router, prefix="/api/ml")
