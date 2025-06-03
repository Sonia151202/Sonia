# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 15:44:57 2025

@author: hp
"""

import streamlit as st
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import tensorflow as tf

# Load the saved model
model = tf.keras.models.load_model('densenet121_fine_tuned_model.h5')

# Define your class labels (Ensure this matches the training data order)
class_labels = ['8cell-A', '8cell-B', '8cell-C', 'Morula-A', 'Morula-B', 'Morula-C', 'Blastocyst-A', 'Blastocyst-B', 'Blastocyst-C', 'Does not belong to any class']

# Function to preprocess image
def preprocess_image(image):
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# Streamlit UI
st.title("Embryo Image Classification App")
st.write("Upload an image from your PC or paste an image URL from Google to classify it into one of the 10 classes.")

# Upload Image from PC
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Paste Image URL
image_url = st.text_input("Or paste an image URL:")

image = None

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)  # UPDATED

elif image_url:
    try:
        response = requests.get(image_url)
        image = Image.open(BytesIO(response.content))
        st.image(image, caption="Image from URL", use_container_width=True)  # UPDATED
    except Exception as e:
        st.error(f"Failed to load image from URL: {e}")

if image:
    # Preprocess the image
    processed_image = preprocess_image(image)

    # Predict the class
    predictions = model.predict(processed_image)
    predicted_class_index = np.argmax(predictions)
    predicted_class_label = class_labels[predicted_class_index]
    confidence_score = np.max(predictions) * 100

    # Display Prediction
    st.write(f"**Prediction: {predicted_class_label}**")
    st.write(f"**Confidence: {confidence_score:.2f}%**")
