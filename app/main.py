from fastapi import FastAPI
from app.api.predict import router as predict_router

app = FastAPI()
app.include_router(predict_router)
