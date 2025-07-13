import tensorflow as tf
import numpy as np
from PIL import Image, UnidentifiedImageError
import io
import requests

MODEL_PATH = "app/models/flower_cnn_model.h5"
IMG_SIZE = (128, 128)

model = tf.keras.models.load_model(MODEL_PATH)
class_names = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']

def preprocess_image(image: Image.Image) -> np.ndarray:
    image = image.convert("RGB").resize(IMG_SIZE)
    image_array = np.expand_dims(np.array(image) / 255.0, axis=0)
    return image_array

def predict_flower(file: bytes) -> dict:
    image = Image.open(io.BytesIO(file))
    image_array = preprocess_image(image)
    return _predict_from_array(image_array)

def predict_from_url(url: str) -> dict:
    response = requests.get(url, timeout=10)

    # ✅ Check if URL returned successfully
    response.raise_for_status()

    # ✅ Validate content-type header
    content_type = response.headers.get("Content-Type", "")
    if not content_type.startswith("image/"):
        raise ValueError(f"URL does not point to a valid image (got Content-Type: {content_type})")

    # ✅ Try to open as image (final sanity check)
    try:
        image = Image.open(io.BytesIO(response.content))
    except UnidentifiedImageError:
        raise ValueError("The URL content is not a valid image.")

    image_array = preprocess_image(image)
    return _predict_from_array(image_array)

def _predict_from_array(image_array: np.ndarray) -> dict:
    predictions = model.predict(image_array)[0]
    predicted_index = np.argmax(predictions)
    return {
        "class": class_names[predicted_index],
        "confidence": float(predictions[predicted_index])
    }
