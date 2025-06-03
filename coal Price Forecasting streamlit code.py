# -*- coding: utf-8 -*-
"""
Created on Sat Apr  5 12:48:30 2025

@author: hp
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ---------------------- Load Data ----------------------

st.title("🛢️ Coal Price Forecasting App")

df = pd.read_excel(r'C:\Users\hp\Downloads\merged_file (3).xlsx')
forecasted_df = pd.read_excel(r'C:\Users\hp\final external factors predictions.xlsx')

# Rename columns to match historical
forecasted_df = forecasted_df.rename(columns={
    'Natural Gas Price Forecast': 'Natural Gas Price',
    'US Crude Oil WTI Prices Forecast': 'US Crude Oil WTI Prices',
    'Dubai Crude Oil Forecast': 'Dubai Crude Oil',
    'Dutch Natural Gas Forecast': 'Dutch Natural Gas'
})

# Convert dates
df['Date'] = pd.to_datetime(df['Date'])
forecasted_df['Date'] = pd.to_datetime(forecasted_df['Date'])

# Drop future dates from historical to avoid leakage
df = df[~df['Date'].isin(forecasted_df['Date'])]

# Create 'Days' column
df['Days'] = (df['Date'] - df['Date'].min()).dt.days
forecasted_df['Days'] = (forecasted_df['Date'] - df['Date'].min()).dt.days

# ---------------------- User Inputs ----------------------

st.sidebar.header("📅 Select Date Range")
start_date = st.sidebar.date_input("Start Date", forecasted_df['Date'].min().date())
end_date = st.sidebar.date_input("End Date", forecasted_df['Date'].max().date())

# Filter forecasted_df for selected date range
filtered_forecast = forecasted_df[
    (forecasted_df['Date'] >= pd.to_datetime(start_date)) & 
    (forecasted_df['Date'] <= pd.to_datetime(end_date))
]

# Features and targets
features = ['Natural Gas Price', 'US Crude Oil WTI Prices', 'Dubai Crude Oil', 'Dutch Natural Gas', 'Days']

target_columns = [
    'Coal Richards Bay 4800kcal NAR fob, London close, USD/t',
    'Coal Richards Bay 5500kcal NAR fob, London close, USD/t',
    'Coal Richards Bay 5700kcal NAR fob, London close, USD/t',
    'Coal Richards Bay 6000kcal NAR fob current week avg, No time stamp, USD/t',
    'Coal India 5500kcal NAR cfr, London close, USD/t'
]

# ---------------------- Model Training & Prediction ----------------------

for target in target_columns:
    st.markdown(f"### 🔹 {target}")

    # Forward fill missing values
    df[target] = df[target].ffill()

    # Prepare training data
    train_data = df[features + [target]].copy()
    X_train = train_data[features]
    y_train = train_data[target]

    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Predict on training data for evaluation
    y_pred = model.predict(X_train)

    # Predict on future data
    X_forecast = filtered_forecast[features]
    future_predictions = model.predict(X_forecast)

    # Calculate metrics
    r2 = r2_score(y_train, y_pred)
    mae = mean_absolute_error(y_train, y_pred)
    mse = mean_squared_error(y_train, y_pred)
    mape = np.mean(np.abs((y_train - y_pred) / y_train)) * 100

    # Show metrics
    st.markdown("**📊 Model Evaluation (on historical data):**")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("R² Score", f"{r2:.4f}")
    col2.metric("MAE", f"{mae:.2f}")
    col3.metric("MSE", f"{mse:.2f}")
    col4.metric("MAPE", f"{mape:.2f}%")

    # Plot predicted prices
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(filtered_forecast['Date'], future_predictions, marker='o', linestyle='-')
    ax.set_title(f"{target} Forecast")
    ax.set_xlabel("Date")
    ax.set_ylabel("USD/t")
    ax.grid(True)
    st.pyplot(fig)


