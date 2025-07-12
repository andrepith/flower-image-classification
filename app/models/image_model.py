import tensorflow as tf
import numpy as np
from PIL import Image
import io

MODEL_PATH = "app/models/flower_cnn_model.h5"
IMG_SIZE = (128, 128)
CLASS_NAMES = ["daisy", "dandelion", "rose", "sunflower", "tulip"]

model = tf.keras.models.load_model(MODEL_PATH)

def read_imagefile(file) -> np.ndarray:
    image = Image.open(io.BytesIO(file)).convert("RGB")
    image = image.resize(IMG_SIZE)
    image_array = np.array(image) / 255.0
    return np.expand_dims(image_array, axis=0)

def predict_image_class(file: bytes):
    img_array = read_imagefile(file)
    predictions = model.predict(img_array)
    class_index = np.argmax(predictions[0])
    return {
        "predicted_class": CLASS_NAMES[class_index],
        "confidence": float(predictions[0][class_index])
    }
