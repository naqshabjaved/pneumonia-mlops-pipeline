import streamlit as st
import numpy as np
import tensorflow as tf
from src.predict import predict_pneumonia
from PIL import Image

st.set_page_config(
    page_title="X-ray Pneumonia Detector (MLOps Demo)",
    layout="centered"
)

st.title("🩺 X-ray Pneumonia Detector")
st.subheader("MLOps Deployment Demo (DVC + MLFlow)")

st.markdown("""
This application uses a Convolutional Neural Network (MobileNetV2) trained 
to classify chest X-ray images as **NORMAL** or showing signs of **PNEUMONIA**.

*Model Metrics (from latest pipeline run):*
* **Accuracy:** 89.42%
* **Recall:** 93.33% (Critical for avoiding false negatives)
""")

uploaded_file = st.file_uploader(
    "Upload a Chest X-ray Image (JPG/JPEG/PNG)", 
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded X-ray Image', use_column_width=True)
    st.markdown("---")
    
    if st.button("Analyze X-ray"):
        st.info("Analyzing image...")
        
       
        predicted_class, confidence_percent, raw_prob = predict_pneumonia(uploaded_file)
        
        st.markdown("### 🔬 Prediction Result")

        if predicted_class == "PNEUMONIA":
            st.error(f"**Classification: {predicted_class}**")
            st.markdown(f"**Confidence:** {confidence_percent:.2f}%")
            st.warning("Immediate medical review is recommended.")
            
        elif predicted_class == "NORMAL":
            st.success(f"**Classification: {predicted_class}**")
            st.markdown(f"**Confidence:** {confidence_percent:.2f}%")
            st.info("This image is classified as normal.")
            
        else:
            st.exception("An internal error occurred during prediction.")

        st.markdown(f"*(Raw model output probability of PNEUMONIA: {raw_prob:.4f})*")

else:
    st.info("Please upload an X-ray image to start the analysis.")