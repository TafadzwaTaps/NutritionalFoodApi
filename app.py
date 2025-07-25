import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
import requests
import keras
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

print(f"TensorFlow version: {tf.__version__}")
print(f"Keras version: {keras.__version__}")
# ========== FastAPI App ==========
app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========== Config ==========
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
img_height, img_width = 512, 512
model_path = os.path.join(BASE_DIR, "final_nutrition_model.h5")
label_stats_path = os.path.join(BASE_DIR, "label_mean_std.json")
csv_path = os.path.join(BASE_DIR, "nutrition_db.csv")

# Hugging Face URLs
MODEL_URL = "https://huggingface.co/Rathious/NutritionalModel/resolve/main/final_nutrition_model.h5"
STATS_URL = "https://huggingface.co/Rathious/NutritionalModel/resolve/main/label_mean_std.json"

# ========== Safe Downloader ==========
def download_file_if_missing(path, url):
    if not os.path.exists(path):
        print(f"📥 {path} not found, downloading from {url} ...")
        try:
            with requests.get(url, stream=True) as r:
                r.raise_for_status()
                with open(path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
            print(f"✅ Downloaded: {path}")
        except Exception as e:
            print(f"❌ Failed to download {url}: {e}")
            raise RuntimeError(f"Download failed: {url}")

# ========== Load Model and Normalization Stats ==========
try:
    download_file_if_missing(model_path, MODEL_URL)
    download_file_if_missing(label_stats_path, STATS_URL)

    print(f"Trying to load model from: {model_path} (Exists: {os.path.exists(model_path)})")
    model = tf.keras.models.load_model(model_path)

    print(f"Trying to load label stats from: {label_stats_path} (Exists: {os.path.exists(label_stats_path)})")
    with open(label_stats_path, "r") as f:
        stats = json.load(f)
        label_mean = np.array(stats["mean"])
        label_std = np.array(stats["std"])
    print("✅ Model and normalization stats loaded.")
except Exception as e:
    print(f"❌ Failed to load model or stats: {e}")
    raise RuntimeError("Model or label_mean_std.json missing.")

# ========== Load Original CSV ==========
try:
    print(f"Trying to load CSV from: {csv_path} (Exists: {os.path.exists(csv_path)})")
    ground_truth_df = pd.read_csv(csv_path)
    ground_truth_df["filename"] = ground_truth_df["filename"].apply(lambda x: os.path.basename(x))
    print("✅ Ground truth CSV loaded.")
except Exception as e:
    print(f"❌ Failed to load nutrition_db.csv: {e}")
    raise RuntimeError("nutrition_db.csv is missing or malformed.")

# ========== Preprocessing Function ==========
def tf_preprocess_image(image_bytes: bytes):
    try:
        img = tf.io.decode_jpeg(image_bytes, channels=3)
        img = tf.image.resize(img, [img_height, img_width])
        img = tf.image.convert_image_dtype(img, tf.float32)
        img = tf.expand_dims(img, axis=0)
        return img
    except Exception as e:
        raise ValueError(f"Image preprocessing failed: {e}")

# ========== Health Check ==========
@app.get("/")
def root():
    return {"message": "✅ Nutrition model API is running on port 9000."}

# ========== Analyze Endpoint ==========
@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        return JSONResponse(status_code=400, content={"error": "Upload a valid image file."})

    try:
        contents = await file.read()
        if len(contents) < 100:
            return JSONResponse(status_code=400, content={"error": "Image is too small or empty."})

        img_tensor = tf_preprocess_image(contents)

        pred_standardized = model.predict(img_tensor)[0]
        pred_real = (pred_standardized * label_std) + label_mean

        # Extract values and round
        calories_pred, protein_pred, carbs_pred, fats_pred = map(lambda x: round(x, 2), pred_real.tolist())

        # Match filename with original CSV values
        filename_only = os.path.basename(file.filename)
        match = ground_truth_df[ground_truth_df["filename"] == filename_only]

        if not match.empty:
            row = match.iloc[0]
            original_values = {
                "calories_true": float(row["calories"]),
                "protein_true": float(row["protein"]),
                "carbs_true": float(row["carbs"]),
                "fats_true": float(row["fats"]),
            }
        else:
            original_values = {"note": "Original values not found in CSV."}

        result = {
            "filename": filename_only,
            "calories_pred": max(0.0, calories_pred),
            "protein_pred": max(0.0, protein_pred),
            "carbs_pred": max(0.0, carbs_pred),
            "fats_pred": max(0.0, fats_pred),
            "calories": max(0.0, calories_pred),
            "protein": max(0.0, protein_pred),
            "carbs": max(0.0, carbs_pred),
            "fats": max(0.0, fats_pred),
        }
        result.update(original_values)

        return JSONResponse(content=result)

    except ValueError as ve:
        return JSONResponse(status_code=400, content={"error": str(ve)})
    except tf.errors.ResourceExhaustedError:
        return JSONResponse(status_code=507, content={"error": "Server out of memory. Try smaller image."})
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return JSONResponse(status_code=500, content={"error": "Internal server error."})
