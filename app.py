import streamlit as st
import numpy as np
import gdown
import os

# Error handling for joblib import
try:
    import joblib
except ImportError:
    st.error("joblib not found. Please install with: pip install joblib")
    st.stop()

# Download model from Google Drive using gdown
model_file_id = "1blkPGoHq53vcJ8Pg8iZU_8rd3FKueBdH"
model_gdown_url = f"https://drive.google.com/uc?id={model_file_id}"
model_path = "heart_disease_model.pkl"

# Download model only if not already present
if not os.path.exists(model_path):
    with st.spinner("Downloading model..."):
        try:
            gdown.download(model_gdown_url, model_path, quiet=False)
        except Exception as e:
            st.error(f"Error downloading model: {e}")
            st.stop()

# Download features file from Google Drive
features_file_id = "1r_sWAxqUfblI6MVGBQ8P0zij_GUEoBFQ"
features_gdown_url = f"https://drive.google.com/uc?id={features_file_id}"
features_path = "model_features.pkl"

# Download features only if not already present
if not os.path.exists(features_path):
    with st.spinner("Downloading features..."):
        try:
            gdown.download(features_gdown_url, features_path, quiet=False)
        except Exception as e:
            st.error(f"Error downloading features: {e}")
            st.stop()

# Load the model and features with error handling
try:
    model = joblib.load(model_path)
    features = joblib.load(features_path)
except Exception as e:
    st.error(f"Error loading model or features: {e}")
    st.stop()

st.title("Heart Disease Risk Prediction")

# User input
user_input = []
for feature in features:
    value = st.number_input(f"Enter value for {feature}", step=1.0)
    user_input.append(value)

# Predict
if st.button("Predict"):
    try:
        prediction = model.predict([user_input])[0]
        proba = model.predict_proba([user_input])[0][1]

        result = "At Risk" if prediction == 1 else "Not at Risk"
        st.success(f"Prediction: **{result}**")
        st.info(f"Confidence Score: **{proba:.2f}**")
    except Exception as e:
        st.error(f"Error making prediction: {e}")