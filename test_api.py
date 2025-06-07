import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf

# Config
data_dir = r"C:\Users\manix\source\FoodApi\Data"  # Your image folder path
csv_path = "labels.csv"
model_path = "food_model_advanced.h5"
label_max_path = "label_max.json"
img_height, img_width = 512, 512

# Load model and label max
model = tf.keras.models.load_model(model_path)
with open(label_max_path, "r") as f:
    label_max = np.array(json.load(f))

# Load CSV
df = pd.read_csv(csv_path)
assert all(col in df.columns for col in ['filename', 'carbs', 'fats', 'protein']), "CSV missing required columns"

true_labels = []
pred_labels = []

def tf_preprocess_image_from_path(img_path):
    image_bytes = tf.io.read_file(img_path)
    img = tf.io.decode_jpeg(image_bytes, channels=3)        # Decode jpeg to tensor
    img = tf.image.resize(img, [img_height, img_width])     # Resize exactly like training
    img = tf.image.convert_image_dtype(img, tf.float32)     # Scale to [0,1] float32
    img = tf.expand_dims(img, axis=0)                        # Add batch dim
    return img

for idx, row in df.iterrows():
    img_path = os.path.join(data_dir, row['filename'])
    if not os.path.isfile(img_path):
        print(f"Image not found: {img_path}")
        continue
    
    img_tensor = tf_preprocess_image_from_path(img_path)
    pred = model.predict(img_tensor)[0]
    pred_real = pred * label_max

    true = np.array([row['carbs'], row['fats'], row['protein']])

    true_labels.append(true)
    pred_labels.append(pred_real)

    print(f"{row['filename']}:")
    print(f"  True  - carbs: {true[0]:.2f}, fats: {true[1]:.2f}, protein: {true[2]:.2f}")
    print(f"  Pred  - carbs: {pred_real[0]:.2f}, fats: {pred_real[1]:.2f}, protein: {pred_real[2]:.2f}")
    print("-" * 40)

true_labels = np.array(true_labels)
pred_labels = np.array(pred_labels)

mae = np.mean(np.abs(true_labels - pred_labels), axis=0)
print(f"Mean Absolute Error:")
print(f"  Carbs: {mae[0]:.2f}")
print(f"  Fats: {mae[1]:.2f}")
print(f"  Protein: {mae[2]:.2f}")
