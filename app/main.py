from fastapi import FastAPI
from app.routes import image_api

app = FastAPI(title="Flower Classifier API")

app.include_router(image_api.router, prefix="/api")
