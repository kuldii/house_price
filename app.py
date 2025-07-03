import joblib

import numpy as np
import pandas as pd
import gradio as gr

# ---------------------------------------------
# Load trained models and kmeans
# ---------------------------------------------
results = joblib.load("models/models.pkl")
kmeans = joblib.load("models/kmeans.pkl")

# ---------------------------------------------
# Prediction function
# ---------------------------------------------
def predict_house_price_with_model(
    model_name,
    MedInc,
    HouseAge,
    AveRooms,
    AveBedrms,
    Population,
    AveOccup,
    Latitude,
    Longitude
):
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

    return (
        f"✅ Prediction Complete!\n\n"
        f"🏠 Region: {region}\n"
        f"💲 Estimated House Price: ${pred * 100000:,.2f}\n"
    )


# ---------------------------------------------
# Gradio UI
# ---------------------------------------------
with gr.Blocks() as demo:
    gr.Markdown("""
        ## 🏠 California House Price Prediction
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

    model_choice = gr.Dropdown(
        choices=list(results.keys()),
        label="Select Regression Model",
        value=list(results.keys())[0]
    )

    with gr.Row():
        MedInc = gr.Slider(minimum=0, maximum=100, value=0, label='Median Income')
        HouseAge = gr.Slider(minimum=0, maximum=100, value=0, label='House Age')

    with gr.Row():
        AveRooms = gr.Slider(minimum=0, maximum=200, value=0, label='Average Rooms')
        AveBedrms = gr.Slider(minimum=0, maximum=50, value=0, label='Average Bedrooms')

    with gr.Row():
        Population = gr.Slider(minimum=0, maximum=15000, value=0, label='Population')
        AveOccup = gr.Slider(minimum=0, maximum=20, value=0, label='Average Occupants')

    with gr.Row():
        Latitude = gr.Slider(minimum=32, maximum=42, value=32, label='Latitude')
        Longitude = gr.Slider(minimum=-124, maximum=-114, value=-124, label='Longitude')

    predict_btn = gr.Button("📈 Predict House Price")
    output = gr.Textbox(label="Prediction Result")

    predict_btn.click(
        fn=predict_house_price_with_model,
        inputs=[model_choice, MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude],
        outputs=output
    )

demo.launch(server_name="0.0.0.0", root_path="/house_price", server_port=9001)
