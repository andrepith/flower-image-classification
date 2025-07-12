import tensorflow as tf
import numpy as np
from PIL import Image
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
    response.raise_for_status()
    image = Image.open(io.BytesIO(response.content))
    image_array = preprocess_image(image)
    return _predict_from_array(image_array)

def _predict_from_array(image_array: np.ndarray) -> dict:
    predictions = model.predict(image_array)[0]
    predicted_index = np.argmax(predictions)
    return {
        "class": class_names[predicted_index],
        "confidence": float(predictions[predicted_index])
    }
