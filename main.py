# main.py

import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping

# ======================
# Dataset Paths
# ======================
train_img_dir = "dataset/images/train"
val_img_dir = "dataset/images/val"
train_labels_dir = "dataset/labels/train"
val_labels_dir = "dataset/labels/val"


# ======================
# Load Data Function
# ======================
def load_data(img_dir, labels_dir):
    images, labels = [], []
    for img_name in sorted(os.listdir(img_dir)):
        if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            # Load image
            img_path = os.path.join(img_dir, img_name)
            img = cv2.imread(img_path)
            img = cv2.resize(img, (224, 224))
            images.append(img)

            # Load label (YOLO-style: class_id x_center y_center w h → take first token)
            label_filename = img_name.rsplit('.', 1)[0] + '.txt'
            label_path = os.path.join(labels_dir, label_filename)
            with open(label_path, 'r') as f:
                content = f.read().strip()
            class_id = content.split(' ')[0]
            labels.append(class_id)

    images = np.array(images, dtype="float32") / 255.0
    labels = np.array(labels)
    return images, labels


# ======================
# Load Training & Validation Data
# ======================
train_images, train_labels = load_data(train_img_dir, train_labels_dir)
val_images, val_labels = load_data(val_img_dir, val_labels_dir)

# Convert string class ids to integers
label_list = sorted(list(set(train_labels)))
label_map = {label: idx for idx, label in enumerate(label_list)}
train_labels_num = np.array([label_map[x] for x in train_labels])
val_labels_num = np.array([label_map[x] for x in val_labels])


# ======================
# Data Augmentation
# ======================
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest"
)
datagen.fit(train_images)


# ======================
# CNN Model
# ======================
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation="relu", input_shape=(224, 224, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    layers.Dropout(0.5),
    layers.Dense(len(label_map), activation="softmax")
])

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])


# ======================
# Training with Early Stopping
# ======================
early_stop = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)

history = model.fit(
    datagen.flow(train_images, train_labels_num, batch_size=32),
    validation_data=(val_images, val_labels_num),
    epochs=20,
    callbacks=[early_stop]
)


# ======================
# Save Model
# ======================
model.save("posture_fall_detection_model.keras")
print("✅ Training complete and model saved.")


# ======================
# Test Prediction on 1 Sample
# ======================
sample_idx = 0
pred = model.predict(np.expand_dims(val_images[sample_idx], axis=0))
pred_class = label_list[np.argmax(pred)]
print(f"Sample true label: {val_labels[sample_idx]}, Predicted: {pred_class}")
