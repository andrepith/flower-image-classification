from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from app.schemas.image_prediction import ImagePredictionResponse, ImageURLRequest
from app.utils.predict_image import predict_flower, predict_from_url

router = APIRouter()

@router.post("/predict-image-file", response_model=ImagePredictionResponse)
async def predict_image_file(file: UploadFile = File(...)):
    if not file.filename.lower().endswith((".jpg", ".jpeg", ".png")):
        raise HTTPException(status_code=400, detail="Only image files are supported")
    
    content = await file.read()
    result = predict_flower(content)

    return ImagePredictionResponse(class_=result["class"], confidence=result["confidence"])


@router.post("/predict-image-url", response_model=ImagePredictionResponse)
async def predict_image_url(payload: ImageURLRequest):
    try:
        result = predict_from_url(payload.image_url)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error fetching image: {str(e)}")

    return ImagePredictionResponse(class_=result["class"], confidence=result["confidence"])
