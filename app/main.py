from fastapi import FastAPI
from app.Api import predict

app = FastAPI()

app.include_router(predict.router)
