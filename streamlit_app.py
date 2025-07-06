import os
import gdown
import joblib
import numpy as np
import pandas as pd
import streamlit as st

MODELS_FILE_ID = "1fnusPs63tnDnBUfDg9sPtxvcxFzU7uX1"
KMEANS_FILE_ID = "1jDaLgu7wpIjgC4cYAF63ypSqoc35JiNg"

def download_from_gdrive(file_id, dest_path):
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, dest_path, quiet=False)

os.makedirs("models", exist_ok=True)

if not os.path.exists("models/models.pkl"):
    download_from_gdrive(MODELS_FILE_ID, "models/models.pkl")

if not os.path.exists("models/kmeans.pkl"):
    download_from_gdrive(KMEANS_FILE_ID, "models/kmeans.pkl")

results = joblib.load("models/models.pkl")
kmeans = joblib.load("models/kmeans.pkl")

st.set_page_config(page_title="California House Price Prediction", layout="centered")
st.title("🏠 California House Price Prediction")
st.markdown("""
    Built with Scikit-learn & Gradio — by Kuldii Project

    This app predicts the estimated house price in California based on property features such as:
    - Median Income
    - House Age
    - Average Rooms & Bedrooms
    - Population & Occupants
    - Latitude & Longitude

    🏷️ **Model Selection**: Choose from different regression models (Random Forest, Linear Regression, etc.)  
    📍 **Region Clustering**: The app automatically assigns a region based on the location (Latitude & Longitude), clustered using KMeans into a total of 10 regions.  
    📊 **Dataset Source**: This app uses the California Housing dataset provided by Scikit-learn.

    Just fill in the values and click **Predict** to see the estimated price!
""")

# Main UI (not sidebar)
with st.form("prediction_form"):
    st.subheader("Input Property Features")
    col1, col2 = st.columns(2)
    with col1:
        MedInc = st.slider('Median Income', 0.0, 100.0, 3.0)
        HouseAge = st.slider('House Age', 0, 100, 20)
        AveRooms = st.slider('Average Rooms', 0.0, 200.0, 5.0)
        AveBedrms = st.slider('Average Bedrooms', 0.0, 50.0, 1.0)
    with col2:
        Population = st.slider('Population', 0, 15000, 1000)
        AveOccup = st.slider('Average Occupants', 0.0, 20.0, 3.0)
        Latitude = st.slider('Latitude', 32.0, 42.0, 34.0)
        Longitude = st.slider('Longitude', -124.0, -114.0, -118.0)
    model_name = st.selectbox(
        "Select Regression Model",
        list(results.keys()),
        index=0
    )
    submitted = st.form_submit_button("📈 Predict House Price")

if submitted:
    region = kmeans.predict([[Latitude, Longitude]])[0]
    input_dict = {
        'MedInc': [MedInc],
        'HouseAge': [HouseAge],
        'AveRooms': [AveRooms],
        'AveBedrms': [AveBedrms],
        'Population': [Population],
        'AveOccup': [AveOccup],
        'Region': [region]
    }
    input_df = pd.DataFrame(input_dict)
    model = results[model_name]['model']
    pred = model.predict(input_df)[0]
    st.success(f"✅ Prediction Complete!\n\n🏠 Region: {region}\n💲 Estimated House Price: ${pred * 100000:,.2f}")
