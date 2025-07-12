from pydantic import BaseModel, HttpUrl
from typing import Optional

class ImagePredictionResponse(BaseModel):
    class_: str
    confidence: float

class ImageURLRequest(BaseModel):
    image_url: HttpUrl
