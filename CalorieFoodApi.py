import os
import json
import time
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, applications, callbacks, regularizers


# ========= Config =========
data_dir = r"C:\Users\manix\source\FoodApi\ImageData"
csv_path = 'nutrition_db.csv'
img_height, img_width = 512, 512  # Use smaller size for faster training/debugging
batch_size = 16
epochs = 50
val_split = 0.2
AUTOTUNE = tf.data.AUTOTUNE
seed = 42

# ========= Load CSV =========
df = pd.read_csv(csv_path)
required_cols = ['filename', 'calories', 'protein', 'carbs', 'fats']
assert all(col in df.columns for col in required_cols), "CSV missing required columns"
assert not df[required_cols[1:]].isna().any().any(), "Missing nutritional values!"

# Build image paths
df['filepath'] = df['filename'].apply(lambda x: os.path.join(data_dir, x))
df = df[df['filepath'].apply(os.path.isfile)]

print(f"✅ Found {len(df)} valid image-label pairs.")

# ========= Normalize Labels =========
labels = df[['calories', 'protein', 'carbs', 'fats']].values.astype(np.float32)
label_mean = labels.mean(axis=0)
label_std = labels.std(axis=0) + 1e-7
labels_standardized = (labels - label_mean) / label_std

with open("label_mean_std.json", "w") as f:
    json.dump({"mean": label_mean.tolist(), "std": label_std.tolist()}, f)

# ========= Train/Val Split =========
filepaths = df['filepath'].tolist()
val_size = int(len(filepaths) * val_split)

full_ds = tf.data.Dataset.from_tensor_slices((filepaths, labels_standardized))

def decode_image(path, label):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [img_height, img_width])
    img = tf.image.convert_image_dtype(img, tf.float32)
    return img, label

def augment(img, label):
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_brightness(img, 0.1)
    img = tf.image.random_contrast(img, 0.8, 1.2)
    img = tf.image.random_saturation(img, 0.8, 1.2)  # NEW
    img = tf.image.random_hue(img, 0.05)             # NEW
    img = tf.image.central_crop(img, central_fraction=0.9)  # Optional alternative to crop
    img = tf.image.resize(img, [img_height, img_width])
    img = tf.clip_by_value(img, 0.0, 1.0)
    return img, label


# Shuffle & map
full_ds = full_ds.shuffle(len(filepaths), seed=seed).map(decode_image, num_parallel_calls=AUTOTUNE)
train_ds = full_ds.skip(val_size).map(augment, num_parallel_calls=AUTOTUNE).batch(batch_size).prefetch(AUTOTUNE)
val_ds = full_ds.take(val_size).batch(batch_size).prefetch(AUTOTUNE)

# ========= Model: ResNet50 =========
base_model = applications.ResNet50(include_top=False, weights='imagenet', input_shape=(img_height, img_width, 3), pooling='avg')
base_model.trainable = False

inputs = layers.Input(shape=(img_height, img_width, 3))
x = base_model(inputs, training=False)
x = layers.Dense(1024, activation='relu', kernel_regularizer=regularizers.l2(1e-4))(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(1e-4))(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.4)(x)
outputs = layers.Dense(4, activation='linear')(x)

model = models.Model(inputs, outputs)
model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss='huber', metrics=['mae'])

# ========= Callbacks =========
cb = [
    callbacks.EarlyStopping(patience=8, restore_best_weights=True, monitor='val_loss'),
    callbacks.ReduceLROnPlateau(patience=4, factor=0.5, verbose=1, monitor='val_loss'),
    callbacks.ModelCheckpoint('best_model_resnet50.keras', save_best_only=True, monitor='val_loss', verbose=1)
]

# ========= Train =========
print("🚀 Training ResNet50 (frozen)...")
start = time.time()
history = model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=cb)
print(f"✅ Training done in {(time.time() - start) / 60:.2f} min")

# ========= Fine-Tuning =========
print("🔧 Fine-tuning ResNet50...")
base_model.trainable = True
for i, layer in enumerate(base_model.layers):
    if i < len(base_model.layers) - 50:
        layer.trainable = False
    if isinstance(layer, layers.BatchNormalization):
        layer.trainable = False

model.compile(optimizer=tf.keras.optimizers.Adam(1e-5), loss='huber', metrics=['mae'])
model.fit(train_ds, validation_data=val_ds, epochs=epochs+20, initial_epoch=history.epoch[-1], callbacks=cb)

# ========= Save Model =========
model.export("final_nutrition_model")  # ✅ Keras 3 SavedModel format
print("💾 Saved model: final_nutrition_model.keras")

# ========= Evaluate & Export Predictions =========
print("📊 Generating predictions...")
y_true = np.vstack([y.numpy() for _, y in val_ds.unbatch()])
y_true_real = (y_true * label_std) + label_mean
preds = model.predict(val_ds)
preds_real = (preds * label_std) + label_mean

out_df = pd.DataFrame({
    "calories_true": y_true_real[:, 0],
    "protein_true": y_true_real[:, 1],
    "carbs_true": y_true_real[:, 2],
    "fats_true": y_true_real[:, 3],
    "calories_pred": preds_real[:, 0],
    "protein_pred": preds_real[:, 1],
    "carbs_pred": preds_real[:, 2],
    "fats_pred": preds_real[:, 3],
})
out_df.to_csv("predictions_vs_actual_resnet.csv", index=False)
print("✅ Saved predictions to 'predictions_vs_actual_resnet.csv'")
