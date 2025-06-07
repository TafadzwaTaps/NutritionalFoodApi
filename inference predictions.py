import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array
# from PIL import Image  # Uncomment this if you prefer using PIL approach instead

# ======== CONFIG ==========
model_path = "food_model_advanced.h5"
label_max_path = "label_max.json"
image_dir = r"C:\Users\manix\source\FoodApi\Data"
img_height, img_width = 512, 512

# ======== Load model and label max ========
model = load_model(model_path)
with open(label_max_path, 'r') as f:
    label_max = np.array(json.load(f), dtype=np.float32)

# ======== Image Preprocessing =========
def preprocess_image(img_path):
    # Using tensorflow.keras.utils
    img = load_img(img_path, target_size=(img_width, img_height))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# # Alternative PIL approach:
# def preprocess_image(img_path):
#     img = Image.open(img_path).convert('RGB')
#     img = img.resize((img_width, img_height))
#     img_array = np.array(img) / 255.0
#     img_array = np.expand_dims(img_array, axis=0)
#     return img_array

# ======== Predict Function =========
def predict_image(img_path):
    img_array = preprocess_image(img_path)
    pred_normalized = model.predict(img_array)[0]
    pred_scaled = pred_normalized * label_max
    return {
        'carbs (g)': float(pred_scaled[0]),
        'fats (g)': float(pred_scaled[1]),
        'protein (g)': float(pred_scaled[2])
    }

# ======== Run Predictions on Folder =========
for fname in os.listdir(image_dir):
    if fname.lower().endswith((".jpg", ".jpeg", ".png")):
        full_path = os.path.join(image_dir, fname)
        result = predict_image(full_path)
        print(f"🖼️ {fname} → {result}")
