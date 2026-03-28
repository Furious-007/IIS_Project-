# ==============================
# 1. IMPORTS
# ==============================
import os
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import kagglehub
import ssl

# Fix SSL certificate issue on macOS for downloading Keras weights
ssl._create_default_https_context = ssl._create_unverified_context

# ==============================
# 2. LOAD CSV
# ==============================
# Download latest version
path = kagglehub.dataset_download("paramaggarwal/fashion-product-images-small")
print("Path to dataset files:", path)

csv_path = os.path.join(path, "styles.csv")
image_dir = os.path.join(path, "images")

if not os.path.exists(csv_path) and os.path.exists(os.path.join(path, "myntradataset", "styles.csv")):
    csv_path = os.path.join(path, "myntradataset", "styles.csv")
    image_dir = os.path.join(path, "myntradataset", "images")

df = pd.read_csv(csv_path, on_bad_lines='skip')

df = df[['id', 'articleType']]
df.dropna(inplace=True)

# Create image filename
df['image'] = df['id'].astype(str) + ".jpg"

print(df.head())

# ==============================
# 3. IMAGE PATH
# ==============================
# image_dir was dynamically set from kagglehub download path above

# Keep only existing images
df = df[df['image'].apply(lambda x: os.path.exists(os.path.join(image_dir, x)))]

print("Total valid images:", len(df))

# ==============================
# 4. REDUCE CLASSES (IMPORTANT)
# ==============================
top_categories = df['articleType'].value_counts().head(10).index
df = df[df['articleType'].isin(top_categories)]

print(df['articleType'].value_counts())

# ==============================
# 5. TRAIN-TEST SPLIT
# ==============================
train_df, val_df = train_test_split(
    df,
    test_size=0.2,
    stratify=df['articleType'],
    random_state=42
)

# ==============================
# 6. DATA GENERATORS
# ==============================
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_dataframe(
    train_df,
    directory=image_dir,
    x_col='image',
    y_col='articleType',
    target_size=(224,224),
    batch_size=32,
    class_mode='categorical'
)

val_gen = val_datagen.flow_from_dataframe(
    val_df,
    directory=image_dir,
    x_col='image',
    y_col='articleType',
    target_size=(224,224),
    batch_size=32,
    class_mode='categorical'
)

# ==============================
# 7. MODEL (EfficientNet)
# ==============================
base_model = tf.keras.applications.EfficientNetB0(
    weights='imagenet',
    include_top=False,
    input_shape=(224,224,3)
)

base_model.trainable = False

# FIXED PART ✅
num_classes = len(train_gen.class_indices)
print("Classes:", train_gen.class_indices)

x = base_model.output
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(128, activation='relu')(x)
x = tf.keras.layers.Dropout(0.3)(x)

output = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

model = tf.keras.Model(inputs=base_model.input, outputs=output)

# ==============================
# 8. COMPILE
# ==============================
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ==============================
# 9. TRAIN
# ==============================
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=5
)

# ==============================
# 10. EVALUATE
# ==============================
loss, acc = model.evaluate(val_gen)
print("Validation Accuracy:", acc)

# ==============================
# 11. PLOT
# ==============================
plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='val')
plt.legend()
plt.title("Accuracy")
plt.show()
base_model.trainable = True

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(train_gen, validation_data=val_gen, epochs=3)