import streamlit as st
import os
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import kagglehub
import ssl

st.set_page_config(page_title="Fashion Image Classification", page_icon="👕")

st.markdown("#  Fashion Image Classification")
st.write("This model classifies fashion product images into top categories using a fine-tuned EfficientNetB0 model.")

# Fix SSL certificate issue on macOS for downloading Keras weights
ssl._create_default_https_context = ssl._create_unverified_context

@st.cache_data
def download_data():
    path = kagglehub.dataset_download("paramaggarwal/fashion-product-images-small")
    return path

with st.spinner("Downloading/Locating Dataset..."):
    path = download_data()

csv_path = os.path.join(path, "styles.csv")
image_dir = os.path.join(path, "images")

if not os.path.exists(csv_path) and os.path.exists(os.path.join(path, "myntradataset", "styles.csv")):
    csv_path = os.path.join(path, "myntradataset", "styles.csv")
    image_dir = os.path.join(path, "myntradataset", "images")

@st.cache_data
def load_and_preprocess_dataframe(csv_path, image_dir):
    df = pd.read_csv(csv_path, on_bad_lines='skip')
    df = df[['id', 'articleType']]
    df.dropna(inplace=True)
    df['image'] = df['id'].astype(str) + ".jpg"
    
    # Filter only existing images
    df = df[df['image'].apply(lambda x: os.path.exists(os.path.join(image_dir, x)))]
    
    # Reduce classes
    top_categories = df['articleType'].value_counts().head(10).index
    df = df[df['articleType'].isin(top_categories)]
    return df

with st.spinner("Preprocessing Data..."):
    df = load_and_preprocess_dataframe(csv_path, image_dir)

col1, col2 = st.columns(2)
with col1:
    st.subheader("Data Overview")
    st.write(f"**Total valid images:** `{len(df)}`")
    st.dataframe(df.head())

with col2:
    st.subheader("Top Categories Distribution")
    fig, ax = plt.subplots(figsize=(6,4))
    df['articleType'].value_counts().plot(kind='bar', ax=ax, color='coral')
    st.pyplot(fig)

st.write("---")
st.write("### Model Training")
st.warning("Training the EfficientNetB0 model dynamically may take a few minutes as it downloads imagery and trains for 5 epochs.")

if st.button("Start Training Model"):
    train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['articleType'], random_state=42)

    train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=20, zoom_range=0.2, horizontal_flip=True)
    val_datagen = ImageDataGenerator(rescale=1./255)

    train_gen = train_datagen.flow_from_dataframe(
        train_df, directory=image_dir, x_col='image', y_col='articleType', 
        target_size=(224,224), batch_size=32, class_mode='categorical'
    )
    val_gen = val_datagen.flow_from_dataframe(
        val_df, directory=image_dir, x_col='image', y_col='articleType', 
        target_size=(224,224), batch_size=32, class_mode='categorical'
    )

    base_model = tf.keras.applications.EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224,224,3))
    base_model.trainable = False

    num_classes = len(train_gen.class_indices)
    x = base_model.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    output = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

    model = tf.keras.Model(inputs=base_model.input, outputs=output)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    with st.spinner("Training EfficientNet for 5 epochs... Check your terminal for epoch-by-epoch progress."):
        history = model.fit(train_gen, validation_data=val_gen, epochs=5)
    
    loss, acc = model.evaluate(val_gen)
    st.success(f"Training Complete! Validation Accuracy: **{acc:.4f}**")
    
    st.write("### Accuracy Plot")
    fig2, ax2 = plt.subplots()
    ax2.plot(history.history['accuracy'], label='train', color='blue')
    ax2.plot(history.history['val_accuracy'], label='val', color='orange')
    ax2.legend()
    ax2.set_title("Accuracy over Epochs")
    st.pyplot(fig2)
