import os
import json
from PIL import Image
import numpy as np
import tensorflow as tf
import streamlit as st

# Streamlit App with Enhanced UI
st.set_page_config(page_title="Plant Disease Classification", page_icon="🌿", layout="wide")

# Add a custom background color (grey)
st.markdown(
    """
    <style>
    .stApp {
        background-color: #f0f0f0;
    }
    </style>
    """,
    unsafe_allow_html=True
)


st.markdown(
    """
    <style>
    .title {
        font-size: 40px;
        font-weight: bold;
        color: #2d6a4f;
    }
    .upload-button {
        background-color: #40916c;
        color: white;
        font-size: 16px;
        font-weight: bold;
    }
    .classify-button {
        background-color: #add8e6;
        color: white;
        font-size: 18px;
        font-weight: bold;
        margin-top: 20px;
    }
    .prediction-result {
        font-size: 24px;
        font-weight: bold;
        color: #1b4332;
    }
    </style>
    """,
    unsafe_allow_html=True
)
model_path = "model_training_notebook/plant_disease_prediction_model.h5"
# Load the pre-trained model
model = tf.keras.models.load_model(model_path)

# Loading the class names
class_indices = json.load(open("app/class_indices.json"))


# Function to Load and Preprocess the Image using Pillow
def load_and_preprocess_image(image_path, target_size=(224, 224)):
    # Load the image
    img = Image.open(image_path)
    # Resize the image
    img = img.resize(target_size)
    # Convert the image to a numpy array
    img_array = np.array(img)
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    # Scale the image values to [0, 1]
    img_array = img_array.astype('float32') / 255.
    return img_array


# Function to Predict the Class of an Image
def predict_image_class(model, image_path, class_indices):
    preprocessed_img = load_and_preprocess_image(image_path)
    predictions = model.predict(preprocessed_img)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class_name = class_indices[str(predicted_class_index)]
    return predicted_class_name


# Streamlit App
st.title('🌿 Plant Disease Classifier')

uploaded_image = st.file_uploader("Upload an image...", type=["jpg", "jpeg"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    col1, col2 = st.columns(2)

    with col1:
        resized_img = load_and_preprocess_image(uploaded_image)
        st.image(resized_img)

    with col2:
        if st.button('Classify'):
            # Preprocess the uploaded image and predict the class
            prediction = predict_image_class(model, uploaded_image, class_indices)
            st.success(f'Prediction: {str(prediction)}')
