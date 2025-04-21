import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('best_final_model_{best_model_name}.h5')  # Update with your actual filename
    return model

model = load_model()

st.title("Pedestrian or No Pedestrian Detection System")

file = st.file_uploader("Upload a road photo", type=["jpg", "png"])

def import_and_predict(image_data, model):
    size = (64, 64)  
    image = image_data.convert("RGB")  
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    img_array = np.asarray(image).astype(np.float32) / 255.0
    img_reshape = np.expand_dims(img_array, axis=0)  
    prediction = model.predict(img_reshape)
    return prediction

if file is None:
    st.info("Please upload an image file.")
else:
    image = Image.open(file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    prediction = import_and_predict(image, model)
    class_names = ['pedestrian', 'no pedestrian']
    result = f"**OUTPUT:** {class_names[np.argmax(prediction)]}"
    st.success(result)
