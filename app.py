import os
import io
import json
import zipfile
import requests
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import load_model
import tensorflow as tf

app = FastAPI()

# Allow all CORS origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Constants
MODEL_URL = "https://huggingface.co/Rathious/NutritionalModel/resolve/main/final_nutrition_model.h5"
MEAN_STD_URL = "https://huggingface.co/Rathious/NutritionalModel/resolve/main/label_mean_std.json"
MODEL_PATH = "final_nutrition_model.h5"
MEAN_STD_PATH = "label_mean_std.json"
IMG_SIZE = (224, 224)

# Load model and stats
model = None
mean_std = None

def download_file(url, local_path):
    print(f"📥 Downloading from: {url}")
    response = requests.get(url)
    response.raise_for_status()
    with open(local_path, "wb") as f:
        f.write(response.content)
    print(f"✅ Downloaded: {local_path}")

def load_model_and_stats():
    global model, mean_std

    # Download model if not found
    if not os.path.exists(MODEL_PATH):
        print(f"📦 {MODEL_PATH} not found, downloading...")
        download_file(MODEL_URL, MODEL_PATH)

    # Download mean/std JSON if not found
    if not os.path.exists(MEAN_STD_PATH):
        print(f"📦 {MEAN_STD_PATH} not found, downloading...")
        download_file(MEAN_STD_URL, MEAN_STD_PATH)

    # Load model
    print(f"🔍 Loading model from {MODEL_PATH} ...")
    model = load_model(MODEL_PATH)
    print("✅ Model loaded.")

    # Load mean/std
    with open(MEAN_STD_PATH, "r") as f:
        mean_std = json.load(f)
    print("✅ Mean/Std loaded.")

@app.on_event("startup")
def startup_event():
    load_model_and_stats()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None or mean_std is None:
        raise HTTPException(status_code=500, detail="Model or mean/std not loaded.")

    try:
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        image = image.resize(IMG_SIZE)
        img_array = np.array(image) / 255.0
        img_tensor = np.expand_dims(img_array, axis=0)

        # Predict
        pred_standardized = model.predict(img_tensor)[0]

        # Un-standardize
        preds = {}
        for i, key in enumerate(["calories", "protein", "carbs", "fats"]):
            mean = mean_std[key]["mean"]
            std = mean_std[key]["std"]
            preds[key] = float((pred_standardized[i] * std) + mean)

        return {"predictions": preds}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
