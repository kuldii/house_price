# California House Price Prediction App

A professional, production-ready machine learning app for predicting California house prices using the scikit-learn California housing dataset. Built with robust preprocessing, region clustering, multiple regression models, and a modern Gradio UI. Fully containerized for easy deployment.

---

## 🚀 Features

- **Robust Preprocessing**: Outlier removal, feature engineering, and standardization
- **Region Clustering**: KMeans clustering on Latitude/Longitude to create a `Region` feature
- **Multiple Regression Models**: Linear Regression, Random Forest, Gradient Boosting, SVR (with hyperparameter tuning)
- **Rich Visualizations & EDA**: Boxplots, histograms, scatterplots, pairplots, correlation heatmap, region clustering map, feature importance
- **Interactive Gradio UI**: User-friendly sliders, model selection, and region assignment from coordinates
- **Production-Ready**: Dockerized, reproducible environment, and easy deployment

---

## 🏗️ Project Structure

```
├── app.py                # Gradio app for prediction (production-ready)
├── california_house_price.ipynb  # Full EDA, modeling, and training notebook
├── models/
│   ├── models.pkl        # Trained regression models (joblib)
│   └── kmeans.pkl        # Trained KMeans clustering model
├── requirements.txt      # Python dependencies
├── Dockerfile            # Containerization for deployment
└── README.md             # Project documentation
```

---

## 📊 Data & Preprocessing

- **Dataset**: [California Housing Dataset](https://scikit-learn.org/stable/datasets/real_world.html#california-housing-dataset)
- **Preprocessing**:
  - Outlier removal for `AveRooms`, `AveBedrms`, `Population`, `AveOccup`
  - KMeans clustering (10 regions) on Latitude/Longitude
  - Standardization of numeric features
  - One-hot encoding for region

---

## 🧠 Models

- **Linear Regression**
- **Random Forest Regressor** (with GridSearchCV)
- **Gradient Boosting Regressor** (with GridSearchCV)
- **Support Vector Regressor (SVR)** (with GridSearchCV)

All models are trained and saved for instant prediction in the app.

---

## 🖥️ Gradio App

- **Sliders** for all features (custom min/max for each)
- **Model selection** dropdown
- **Region** is auto-assigned from Latitude/Longitude
- **Prediction output**: Estimated price and region
- **Production config**: Runs on `0.0.0.0:9001` with `/house_price` root path (for Docker)

---

## 📦 Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/kuldii/house_price.git
cd house_price
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. (Optional) Train Models
- All models and clustering objects are pre-trained and saved in `models/`.
- To retrain, use the notebook `california_house_price.ipynb` and re-export the models.

### 4. Run the App
```bash
python app.py
```
- The app will be available at `http://localhost:9001/house_price` by default.

---

## 🐳 Docker Deployment

### 1. Build the Docker Image
```bash
docker build -t house-price .
```

### 2. Run the Container
```bash
docker run -p 9001:9001 house-price
```
- Access the app at `http://localhost:9001/house_price`

---

## 🖥️ Usage

1. Open the app in your browser.
2. Input property features (Median Income, House Age, Rooms, Bedrooms, Population, Occupants, Latitude, Longitude).
3. Select a regression model.
4. Click **Predict House Price** to get the estimated value and region.

---

## 📊 Visualizations & EDA
- See `california_house_price.ipynb` for:
  - Outlier analysis
  - KMeans region clustering
  - Feature importance (tree models)
  - Correlation heatmap
  - Histograms, boxplots, pairplots, and more

---

## 📝 Model Details
- **Preprocessing**: StandardScaler, outlier removal, KMeans clustering for region.
- **Models**: RandomForestRegressor, LinearRegression, GradientBoostingRegressor, SVR (with GridSearchCV for tuning).
- **Region Feature**: Latitude/Longitude mapped to a region cluster (10 clusters) using KMeans.

---

## 📁 File Descriptions
- `app.py`: Gradio app, loads models, handles prediction and UI.
- `models/models.pkl`: Dictionary of trained regression models.
- `models/kmeans.pkl`: KMeans model for region assignment.
- `requirements.txt`: Python dependencies.
- `Dockerfile`: Containerization instructions.
- `california_house_price.ipynb`: Full EDA, preprocessing, model training, and export.

---

## 🌐 Demo & Credits
- **Author**: Sandikha Rahardi (Kuldii Project)
- **Website**: https://hello.kuldiiproject.com/house_price
- **Dataset**: [Scikit-learn California Housing](https://scikit-learn.org/stable/datasets/real_world.html#california-housing-dataset)
- **UI**: [Gradio](https://gradio.app/)
- **ML**: [Scikit-learn](https://scikit-learn.org/)

---

For questions or contributions, please open an issue or pull request.
