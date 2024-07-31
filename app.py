import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import json
 
# Load the pre-trained model
model = tf.keras.models.load_model('staff_mobilenet_v2_model.h5')
 
# Load class names
with open('class_names.json', 'r') as f:
    class_names = json.load(f)
 
# Define image preprocessing function
def preprocess_image(image):
    image = image.resize((640, 640))  # Resize to match the model input
    image = np.array(image) / 255.0   # Normalize pixel values
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image
 
# Streamlit app layout
st.title("Staff Image Recognition")
 
st.write("Upload an image of a staff member to get predictions.")
 
# Upload image
uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
 
if uploaded_image is not None:
    # Display uploaded image
    image = Image.open(uploaded_image)
    st.image(image, caption='Uploaded Image', use_column_width=True)
 
    # Preprocess and predict
    processed_image = preprocess_image(image)
    predictions = model.predict(processed_image)
    predicted_class_index = np.argmax(predictions)
    predicted_class = class_names[predicted_class_index]
 
    # Display prediction
    st.write(f"Prediction: {predicted_class} with probability {np.max(predictions):.2f}")