import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps

# Load the trained model from the local directory
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('MobileNetV2_best_model.h5')  # Ensure this file is in the same directory
    return model

# Load the model
model = load_model()

# Title and file upload
st.write("""
# Pedestrian Detection System
Upload a road image and let the AI detect if there's a pedestrian or not.
""")

file = st.file_uploader("Choose a road photo from your computer", type=["jpg", "png"])

# Image preprocessing and prediction
def import_and_predict(image_data, model):
    size = (64, 64)  # Input shape expected by your model
    image = ImageOps.fit(image_data, size, Image.Resampling.LANCZOS)
    img = np.asarray(image).astype(np.float32) / 255.0  # Normalize the image
    img_reshape = np.expand_dims(img, axis=0)  # Add batch dimension
    prediction = model.predict(img_reshape)
    return prediction

# Prediction output
if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    prediction = import_and_predict(image, model)
    class_names = ['pedestrian', 'no pedestrian']
    result = "The output is: **" + class_names[np.argmax(prediction)] + "**"
    st.success(result)
