# -*- coding: utf-8 -*-
"""
Created on Fri May  2 17:55:22 2025

@author: hp
"""
import streamlit as st
from PIL import Image
from ultralytics import YOLO
import tempfile

# Load the trained YOLOv8 model
model = YOLO("best.pt")  # Replace with your model path

st.title("Pallet Detection using YOLOv8")

# Upload multiple images
uploaded_files = st.file_uploader("Upload image(s)", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files:
    for uploaded_file in uploaded_files:
        img = Image.open(uploaded_file)

        # Save uploaded image temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            img.save(temp_file.name)

            # Run inference
            results = model(temp_file.name)
            res_plotted = results[0].plot()

            # Count pallets
            num_pallets = len(results[0].boxes)

            # Layout side-by-side using Streamlit columns
            col1, col2 = st.columns(2)
            with col1:
                st.image(img, caption="Original Image", use_container_width=True)
            with col2:
                st.image(res_plotted, caption=f"Detected ({num_pallets} pallets)", use_container_width=True)
