import streamlit as st
import joblib
import numpy as np
import gdown  # ✅ use gdown instead of requests
import os

# Download model from Google Drive using gdown
model_file_id = "1blkPGoHq53vcJ8Pg8iZU_8rd3FKueBdH"
model_gdown_url = f"https://drive.google.com/uc?id={model_file_id}"
model_path = "heart_disease_model.pkl"

# Download model only if not already present
if not os.path.exists(model_path):
    gdown.download(model_gdown_url, model_path, quiet=False)

# Download features file from Google Drive
features_file_id = "1r_sWAxqUfblI6MVGBQ8P0zij_GUEoBFQ"
features_gdown_url = f"https://drive.google.com/uc?id={features_file_id}"
features_path = "model_features.pkl"

# Download features only if not already present
if not os.path.exists(features_path):
    gdown.download(features_gdown_url, features_path, quiet=False)

# Load the model
model = joblib.load(model_path)

# Load feature list from downloaded file
features = joblib.load(features_path)

st.title("Heart Disease Risk Prediction")

# User input
user_input = []
for feature in features:
    value = st.number_input(f"Enter value for {feature}", step=1.0)
    user_input.append(value)

# Predict
if st.button("Predict"):
    prediction = model.predict([user_input])[0]
    proba = model.predict_proba([user_input])[0][1]

    result = "At Risk" if prediction == 1 else "Not at Risk"
    st.success(f"Prediction: **{result}**")
    st.info(f"Confidence Score: **{proba:.2f}**")