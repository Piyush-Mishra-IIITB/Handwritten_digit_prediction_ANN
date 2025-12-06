import streamlit as st
import numpy as np
import tensorflow as tf
import cv2

# Load model
model = tf.keras.models.load_model("mnist_ann_model.h5")

st.title("Handwritten Digit Recognition From (0-9)")

uploaded_file = st.file_uploader("Upload a digit image (png/jpg)", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Read image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)

    # Show original image
    st.image(img, caption="Uploaded Digit", width=200)

    # Preprocess: resize to 28x28
    img = cv2.resize(img, (28, 28))
    img = img.reshape(1, 784)
    img = img / 255.0  # same scaling used in training

    # Predict
    pred = model.predict(img)
    digit = pred.argmax()

    st.subheader(f"Predicted Digit: {digit}")
