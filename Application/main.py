from fastapi import FastAPI
from contextlib import asynccontextmanager
from Application.Api.ml_service import router as ml_router
from Application.services.scheduler import start_scheduler

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[STARTUP] Starting prediction scheduler...")
    start_scheduler()
    yield
    print("[SHUTDOWN] App is closing...")

app = FastAPI(lifespan=lifespan)

app.include_router(ml_router, prefix="/api/ml")
