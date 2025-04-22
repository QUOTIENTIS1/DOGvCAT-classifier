# -*- coding: utf-8 -*-
"""
Created on Tue Apr 22 16:49:17 2025

@author: praji
"""

# dac.py - Streamlit App for Dog vs Cat Classification (H5 Model Version)
import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import warnings

# Suppress TensorFlow warnings (optional)
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="🐶 vs 😺 Classifier",
    page_icon="🐾",
    layout="wide"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .stApp {
        background-color: #f9f9f9;
    }
    .uploadedImage {
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .prediction-result {
        font-size: 1.2rem;
        padding: 10px;
        border-radius: 5px;
    }
    .dog-result {
        background-color: #e3f2fd;
        color: #0d47a1;
    }
    .cat-result {
        background-color: #fce4ec;
        color: #880e4f;
    }
</style>
""", unsafe_allow_html=True)

# Load model with caching
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model('dog_cat_classifier.h5')
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

model = load_model()

# Class labels (must match your training)
class_names = ['dog', 'cat']

# Prediction function
def predict_image(img):
    try:
        # Resize and preprocess
        img = img.resize((224, 224))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)  # Add batch dimension
        img_array = img_array / 255.0  # Normalize (same as training)
        
        # Predict
        predictions = model.predict(img_array)
        predicted_class = class_names[np.argmax(predictions[0])]
        confidence = round(np.max(predictions[0]) * 100, 2)
        
        return predicted_class, confidence
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None, None

# Main app UI
st.title("🐶 Dog vs Cat Classifier")
st.markdown("""
Upload an image of a dog or cat, and the AI will classify it!
""")

uploaded_file = st.file_uploader(
    "Choose an image...", 
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed"
)

if uploaded_file is not None:
    try:
        # Display uploaded image
        image = Image.open(uploaded_file)
        st.image(
            image,
            caption="Uploaded Image",
            width=300,
            use_column_width=False,
            output_format="auto",
            clamp=True,
            channels="RGB"
        )
        
        # Make prediction
        with st.spinner('🔍 Analyzing image...'):
            predicted_class, confidence = predict_image(image)
        
        # Show results
        if predicted_class and confidence:
            if predicted_class == 'dog':
                result_class = "dog-result"
                emoji = "🐶"
                gif = "https://media.giphy.com/media/4Zo41lhzKt6iZ8xff9/giphy.gif"
            else:
                result_class = "cat-result"
                emoji = "😺"
                gif = "https://media.giphy.com/media/JIX9t2j0ZTN9S/giphy.gif"
            
            st.markdown(
                f"""
                <div class="prediction-result {result_class}">
                    {emoji} <strong>Prediction:</strong> {predicted_class} 
                    <br>
                    <strong>Confidence:</strong> {confidence}%
                </div>
                """,
                unsafe_allow_html=True
            )
            
            st.image(gif, width=200)
            
            # Optional: Show raw prediction values
            with st.expander("Show raw prediction values"):
                st.write(f"Prediction array: {model.predict(tf.expand_dims(tf.keras.preprocessing.image.img_to_array(image.resize((224, 224)))/255.0, 0))}")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

# Add footer
st.markdown("---")
st.markdown("""
Built with:
- TensorFlow for the deep learning model
- Streamlit for the web interface
- Kaggle Dogs vs Cats dataset
""")
