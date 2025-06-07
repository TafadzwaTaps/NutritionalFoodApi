from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
import json

app = FastAPI()

# CORS settings (allow all origins)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load your trained model once when the app starts
model = tf.keras.models.load_model("final_nutrition_model_resnet50.h5")

# Load mean and std for label denormalization
with open("label_mean_std.json", "r") as f:
    mean_std_data = json.load(f)
    label_mean = np.array(mean_std_data["mean"])
    label_std = np.array(mean_std_data["std"])

img_height, img_width = 512, 512

def tf_preprocess_image(image_bytes):
    """Decode, resize, normalize and batch the image tensor."""
    img = tf.io.decode_jpeg(image_bytes, channels=3)
    img = tf.image.resize(img, [img_height, img_width])
    img = tf.image.convert_image_dtype(img, tf.float32)  # scale to [0,1]
    img = tf.expand_dims(img, axis=0)  # batch dimension
    return img

@app.get("/")
def home():
    return {"message": "FastAPI nutrition model API running on port 9000."}

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        return JSONResponse(status_code=400, content={"error": "Invalid file type. Please upload an image."})

    try:
        contents = await file.read()
        if len(contents) < 100:  # very small image check
            return JSONResponse(status_code=400, content={"error": "Image file is too small or empty."})

        img_tensor = tf_preprocess_image(contents)
        pred_standardized = model.predict(img_tensor)[0]

        # Denormalize predictions
        pred_real = (pred_standardized * label_std) + label_mean
        calories, protein, carbs, fats = map(lambda x: round(x, 2), pred_real.tolist())


        # Calculate calories using Atwater factors
        calories = round(carbs * 4 + fats * 9 + protein * 4, 2)

        # Basic heuristic for empty or non-food detection
        if calories < 10 and (carbs + fats + protein < 1):
            return JSONResponse(status_code=200, content={
                "carbs": carbs,
                "fats": fats,
                "protein": protein,
                "calories": calories,
                "filename": file.filename,
                "warning": "Low nutritional values detected. Might be empty plate or non-food item."
            })

        return JSONResponse(content={
    "calories": calories,
    "protein": protein,
    "carbs": carbs,
    "fats": fats,
    "filename": file.filename
})


    except tf.errors.InvalidArgumentError as tf_err:
        return JSONResponse(status_code=400, content={"error": f"Image processing failed: {tf_err}. Ensure it's a valid JPEG."})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Unexpected error: {e}"})
