import tracemalloc
tracemalloc.start()

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import streamlit as st

@st.cache_data
def load_data(uploaded_file):
    data = pd.read_csv(uploaded_file, dtype={'block': str, 'street_name': str, 'storey_range': str, 'flat_model': str}, low_memory=False)
    return data

def preprocess_data(data):
    data = data.dropna().copy()
    data['lease_commence_date'] = pd.to_datetime(data['lease_commence_date'], format='%Y')
    data['lease_age'] = data['lease_commence_date'].dt.year.apply(lambda x: 2024 - x)
    categorical_features = ['town', 'flat_type', 'flat_model']
    data = pd.get_dummies(data, columns=categorical_features, drop_first=True)
    return data

def feature_engineering(data):
    features = ['floor_area_sqm', 'lease_age']
    features += [col for col in data.columns if col.startswith(('town_', 'flat_type_', 'flat_model_'))]
    X = data[features]
    y = data['resale_price']
    return X, y

# Model Training
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)*100
    return model, mae, mse, rmse, r2

st.title("Singapore Resale Price Predictor")

uploaded_file = st.file_uploader("Upload Resale Flat Prices Dataset (CSV)", type="csv")
if uploaded_file is not None:
    data = load_data(uploaded_file)
    data = preprocess_data(data)
    X, y = feature_engineering(data)
    model, mae, mse, rmse, r2 = train_model(X, y)
    
    st.subheader("Model Performance Metrics")
    st.write(f"**Mean Absolute Error (MAE):** {mae:.2f}")
    st.write(f"**Mean Squared Error (MSE):** {mse:.2f}")
    st.write(f"**Root Mean Squared Error (RMSE):** {rmse:.2f}")
    st.write(f"**R² Score:** {r2:.2f}")

    st.subheader("Make a Prediction")
    floor_area_sqm = st.number_input("Floor Area (sqm)", min_value=30.0, max_value=200.0, step=1.0)
    lease_commence_date = st.number_input("Lease Commence Year", min_value=1990, max_value=2024, step=1)
    lease_age = 2024 - lease_commence_date

    input_data = pd.DataFrame([[floor_area_sqm, lease_age]], columns=['floor_area_sqm', 'lease_age'])

    # Ensure all columns in the model are present in the input data
    for col in X.columns:
        if col not in input_data.columns:
            input_data[col] = 0

    if st.button("Predict Resale Price"):
        try:
            prediction = model.predict(input_data)[0]
            st.success(f"Predicted Resale Price: ${prediction:,.2f}")
        except Exception as e:
            st.error(f"An error occurred during prediction: {str(e)}")
else:
    st.warning("Please upload a dataset to proceed.")
