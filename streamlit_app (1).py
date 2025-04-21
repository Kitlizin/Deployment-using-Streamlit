import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

@st.cache_resource
def load_model():
    # Replace with your actual model filename
    model = tf.keras.models.load_model('best_final_model_{best_model_name}.h5')
    return model

model = load_model()

st.write("# PEDESTRIAN OR NO PEDESTRIAN Detection System")
file = st.file_uploader("Choose a photo from your computer", type=["jpg", "png"])

def import_and_predict(image_data, model):
    size = (64, 64)  # Update this to match your model's expected input size
    image = ImageOps.fit(image_data, size, Image.Resampling.LANCZOS)
    img = np.asarray(image).astype(np.float32) / 255.0
    img_reshape = np.expand_dims(img, axis=0)
    prediction = model.predict(img_reshape)
    return prediction

if file is None:
    st.text("Please upload an image file")
else:
    # Convert image to RGB to handle alpha channels
    image = Image.open(file).convert('RGB')
    st.image(image, use_column_width=True)
    prediction = import_and_predict(image, model)
    class_names = ['pedestrian', 'no pedestrian']
    string = f"The output is: {class_names[np.argmax(prediction)]}"
    st.success(string)
