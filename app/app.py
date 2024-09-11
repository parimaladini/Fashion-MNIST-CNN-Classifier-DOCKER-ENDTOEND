import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np


model = tf.keras.models.load_model("trained_model/fashionMNIST.h5")

st.title("Fashion MNIST CNN Classifier WEB APP")

class_names = ["T-Shit/Top", "Trouser", "PullOver", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle Boot"]

def data_preprocessing(img):
    img = Image.open(img)
    img = img.resize((28, 28)) 
    img = img.convert('L')
    img = np.array(img)/255.0
    img = img.reshape((1, 28, 28, 1))
    return img 

uploaded_img = st.file_uploader("Upload an image", type=["jpeg", "jpg", "png", "webp"])

if uploaded_img is not None:
    image = Image.open(uploaded_img)
    col1, col2 = st.columns(2)

    with col1:
        resized_img = image.resize((100, 100))
        st.image(resized_img)

    with col2:
        if st.button("Classify"):
            img = data_preprocessing(uploaded_img)
            predictions = model.predict(img)
            predictions = np.argmax(predictions)

            st.success(f"It is a: {class_names[predictions]}")


    

