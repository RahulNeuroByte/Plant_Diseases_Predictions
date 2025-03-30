
import numpy as np
import streamlit as st
import cv2
from keras.models import load_model, model_from_json
import os

# Streamlit App Title
st.title(" Plant Disease Detection App")
st.markdown("Upload an image of a **plant leaf** to detect the disease.")

#  Try loading the `.keras` model first, else use JSON + Weights
MODEL_KERAS_PATH = r"D:\myprojects\Project2\plant_disease.keras"
MODEL_JSON_PATH = r"D:\myprojects\Project2\plant_model.json"
MODEL_WEIGHTS_PATH = r"D:\myprojects\Project2\plant_model.weights.h5"

def load_keras_model():
    """Loads a .keras model if available."""
    if os.path.exists(MODEL_KERAS_PATH):
        return load_model(MODEL_KERAS_PATH)
    return None

def load_json_model():
    """Loads a model from JSON + weights if .keras is unavailable."""
    if os.path.exists(MODEL_JSON_PATH) and os.path.exists(MODEL_WEIGHTS_PATH):
        with open(MODEL_JSON_PATH, "r") as json_file:
            model_json = json_file.read()
        model = model_from_json(model_json)
        model.load_weights(MODEL_WEIGHTS_PATH)
        return model
    return None

# Try loading models
model = load_keras_model() or load_json_model()

if model:
    st.success(" Model Loaded Successfully!")
else:
    st.error(" No model file found! Ensure `plant_disease.keras` or `plant_model.json` & `plant_model.weights.h5` exist.")
    st.stop()

#  Get Model Input Shape
model_input_shape = model.input_shape
st.info(f"ℹ Model expects input shape: {model_input_shape}")

#  Class Labels
CLASS_NAMES = ['Corn-Common_rust', 'Potato-Early_blight', 'Tomato-Bacterial_spot']

#  Image Upload Section
plant_image = st.file_uploader("📂 Upload an image...", type=["jpg", "jpeg", "png"])

#  Prediction Logic
if st.button("🔍 Predict"):
    if plant_image is None:
        st.warning("⚠ Please upload an image first!")
        st.stop()

    try:
        # Convert uploaded file to OpenCV format
        file_bytes = np.frombuffer(plant_image.getvalue(), np.uint8)
        opencv_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        # Validate Image
        if opencv_image is None:
            st.error(" Error loading image. Try another image.")
            st.stop()

        # Convert BGR to RGB
        opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)

        # Show Uploaded Image
        st.image(opencv_image, channels="RGB", caption="📷 Uploaded Image", use_column_width=True)

        #  Preprocess Image
        input_size = model.input_shape[1:3]
        opencv_image = cv2.resize(opencv_image, (input_size[0], input_size[1]))
        opencv_image = opencv_image.astype('float32') / 255.0
        opencv_image = np.expand_dims(opencv_image, axis=0)

        #  Model Prediction
        Y_pred = model.predict(opencv_image)

        # Validate Output Shape
        if Y_pred.shape[1] != len(CLASS_NAMES):
            st.error(" Model output does not match CLASS_NAMES. Check your model!")
            st.stop()

        # Get Predicted Class
        predicted_class = np.argmax(Y_pred)
        result = CLASS_NAMES[predicted_class]
        confidence = np.max(Y_pred) * 100

        # Show Prediction Results
        st.success(f" Detected: **{result.split('-')[0]} Leaf** with **{result.split('-')[1]}**.")
        st.info(f" Confidence: **{confidence:.2f}%**")

    except Exception as e:
        st.error(f" Error Processing Image: {str(e)}")
        st.warning(" Please check the image format and try again.")




