import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model # type: ignore
from keras.preprocessing import image
from PIL import Image
import numpy as np

# Load your trained model
model = load_model('English_sigh_language.h5')

# List of class names from A to Z
class_names = [chr(i) for i in range(ord('A'), ord('Z') + 1)]

# Adding additional classes for "space", "delete", and "nothing"
class_names.extend(['space', 'delete', 'nothing'])
st.title("ASL Alphabet Recognition")
st.write("Upload an image of an ASL si gn to identify the letter.")
uploaded_file = st.file_uploader("Choose an image...", type="jpg")
if uploaded_file is not None:
    # Display the uploaded image
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)
    st.write("")

    # Preprocess the image
    img = img.resize((200, 200))  # Resize to the model's input size
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize to [0, 1] range

    # Predict the class
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction, axis=1)
    predicted_label = class_names[predicted_class[0]]

    # Display the prediction
    st.write(f"Prediction: **{predicted_label}**")
