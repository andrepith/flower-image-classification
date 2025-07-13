# 🌸 Flower Image Classifier API

A FastAPI-based ML service to classify flower images using a CNN model trained on the [Kaggle Flowers Dataset](https://www.kaggle.com/datasets/alxmamaev/flowers-recognition).

---

## 🚀 Features

- ✅ Upload image for prediction (via file upload)
- ✅ Predict using image URL
- ✅ CNN-based classifier trained on 5 flower classes
- ✅ Automatic dataset download and caching
- ✅ Clean FastAPI structure for scaling or deployment

---

## 🔧 Installation

### 1. Clone the project

```bash
git clone https://github.com/your-username/ml_fastapi_project.git
cd ml_fastapi_project

python -m venv venv
source venv/bin/activate  # or .\venv\Scripts\activate on Windows

pip install -r requirements.txt

🔐 Kaggle Credentials Setup

    Go to https://www.kaggle.com/settings and create an API token.

    Save the downloaded kaggle.json securely.

    Set the environment variables:

Linux / macOS:

export KAGGLE_USERNAME=your_username
export KAGGLE_KEY=your_key

Windows (CMD):

set KAGGLE_USERNAME=your_username
set KAGGLE_KEY=your_key

🧠 Train the Model (Optional)

python scripts/train_image_model.py

This will:

    Automatically download the dataset via Kaggle

    Cache it to your OS-specific directory

    Train a CNN

    Save model to app/models/flower_cnn_model.h5

🔌 Run the API Server

uvicorn app.main:app --reload

Visit the docs at: http://127.0.0.1:8000/docs
📤 Predict from File

POST /api/predict-image-file
Content-Type: multipart/form-data
Swagger UI:

Try uploading a .jpg or .png image directly.
Example curl:

curl -X POST "http://127.0.0.1:8000/api/predict-image-file" \
  -F "file=@sample.jpg"

🌐 Predict from Image URL

POST /api/predict-image-url
Content-Type: application/json
Body:

{
  "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/4/40/Sunflower_sky_backdrop.jpg/800px-Sunflower_sky_backdrop.jpg"
}

✅ Response

{
  "class_": "sunflower",
  "confidence": 0.9746
}

🧪 Classes

The model was trained on the following flower types:

    daisy

    dandelion

    rose

    sunflower

    tulip

📦 Dependencies

    FastAPI

    TensorFlow / Keras

    Pillow

    Requests

    Platformdirs

    tqdm

    Kaggle API

Install all with:

pip install -r requirements.txt

📝 License

MIT License. Free to use and modify for your ML/AI projects.
👨‍💻 Author

Andre Adikara
Built with ❤️ for practical ML API learning


---

Let me know if you’d like to:
- Add deployment instructions (e.g., Vercel or Docker)
- Add test suite info
- Convert this to PDF or downloadable zip package