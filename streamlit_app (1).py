import streamlit as st
import tensorflow as tf

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('/content/drive/MyDrive/your_folder/best_final_model_{best_model_name}.h5')  
    return model

model = load_model()

st.write("""
# Pedestrian Detection System
""")

file = st.file_uploader("Choose a road photo from your computer", type=["jpg", "png"])

import cv2
from PIL import Image, ImageOps
import numpy as np

def import_and_predict(image_data, model):
    size = (64, 64)  
    image = ImageOps.fit(image_data, size, Image.Resampling.LANCZOS)
    img = np.asarray(image).astype(np.float32) / 255.0  
    img_reshape = np.expand_dims(img, axis=0)
    prediction = model.predict(img_reshape)
    return prediction

if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_container_width=True)
    prediction = import_and_predict(image, model)
    class_names = ['pedestrian', 'no Pedestrian']
    result = "The output is: " + class_names[np.argmax(prediction)]
    st.success(result)
