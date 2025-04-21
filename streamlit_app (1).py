import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import cv2

def main():
    @st.cache_resource
    def load_model():
        model = tf.keras.models.load_model('best_final_model_{best_model_name}.h5')
        return model

    model = load_model()
    
    st.write("# PEDESTRIAN OR NO PEDESTRIAN Detection System")
    file = st.file_uploader("Choose a photo from your computer", type=["jpg", "png"])

    def import_and_predict(image_data, model):
        target_size = model.input_shape[1:3]
        image = ImageOps.fit(image_data, target_size, Image.Resampling.LANCZOS)
        img = np.asarray(image).astype(np.float32) / 255.0

        if model.input_shape[-1] == 1:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            img = np.expand_dims(img, axis=-1)
        
        img_reshape = np.expand_dims(img, axis=0)
        return model.predict(img_reshape)

    if file is None:
        st.text("Please upload an image file")
    else:
        image = Image.open(file).convert('RGB')
        st.image(image, use_container_width=True)
        prediction = import_and_predict(image, model)
        class_names = ['pedestrian', 'no pedestrian']
        st.success(f"The output is: {class_names[np.argmax(prediction)]}")

if __name__ == "__main__":
    main()
