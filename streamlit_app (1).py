import streamlit as st
import tensorflow as tf
import cv2
from PIL import Image, ImageOps
import numpy as np

@st.cache(allow_output_mutation=True)
def load_model():
    try:
        model = tf.keras.models.load_model('best_final_model_{best_model_name}.h5')
        st.success("Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

model = load_model()

if model is None:
    st.stop()  # Stop execution if model fails to load

st.write("""
# PEDESTRIAN OR NO PEDESTRIAN Detection System"""
)
file = st.file_uploader("Choose Road photo from computer", type=["jpg", "png"])

def import_and_predict(image_data, model):
    try:
        size = (64, 64)  # Adjust this if your model expects different dimensions
        image = ImageOps.fit(image_data, size, Image.Resampling.LANCZOS)
        img = np.asarray(image)
        
        # Check if image is RGB (3 channels)
        if len(img.shape) == 2:  # Grayscale
            img = np.stack((img,)*3, axis=-1)
        elif img.shape[2] == 4:  # RGBA
            img = img[:, :, :3]
            
        # Normalize if your model expects values between 0-1
        img = img.astype('float32') / 255.0
        
        # Add batch dimension
        img_reshape = img[np.newaxis, ...]
        
        # Debug print (remove in production)
        st.write(f"Input shape: {img_reshape.shape}")
        
        prediction = model.predict(img_reshape)
        return prediction
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None

if file is None:
    st.text("Please upload an image file")
else:
    try:
        image = Image.open(file)
        st.image(image, use_column_width=True)
        prediction = import_and_predict(image, model)
        
        if prediction is not None:
            class_names = ['pedestrian', 'no pedestrian']
            string = "OUTPUT : " + class_names[np.argmax(prediction)]
            st.success(string)
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
