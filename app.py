import streamlit as st
import joblib
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

model = joblib.load("xgboost_model.pkl")

test_results = pd.read_csv("test_predictions.csv")
test_results["timestamp"] = pd.to_datetime(test_results["timestamp"])

st.sidebar.title("🔍 Navigation & Settings")
view_option = st.sidebar.radio("📊 Choose View:", ["Single Date", "Full Tested Period"])

st.sidebar.subheader("📈 Model Performance")
st.sidebar.write("Mean Absolute Error (MAE): **1.75 EUR/MWh**")
st.sidebar.write("Root Mean Squared Error (RMSE): **4.75 EUR/MWh**")

st.title("⚡ Electricity Price Prediction Showcase")
st.write("""
📌 **Important:** The model is trained on data from **February 2018 to October 2018**  
🕒 This restriction exists due to **open-source historical data limitations**.  
📊 Below, you can explore model performance and test real-time predictions.
""")

if view_option == "Single Date":
    min_date = test_results["timestamp"].min().date()
    max_date = test_results["timestamp"].max().date()
    initial_date = min_date + timedelta(days=1)
    selected_date = st.date_input("📅 Select a Date", min_value=min_date, max_value=max_date, value=initial_date)

    filtered_data = test_results[test_results["timestamp"].dt.date == selected_date]

    if not filtered_data.empty:
        st.subheader(f"📊 Actual vs. Predicted Prices on {selected_date}")
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(filtered_data["timestamp"], filtered_data["actual_price"], label="Actual Price", color="red")
        ax.plot(filtered_data["timestamp"], filtered_data["predicted_price"], label="Predicted Price", linestyle="dashed", color="blue")
        ax.set_xlabel("Time")
        ax.set_ylabel("Price (EUR/MWh)")
        ax.set_title(f"Electricity Prices on {selected_date}")
        ax.legend()
        st.pyplot(fig)

        st.write("📊 **Test Data for Selected Date:**")
        st.dataframe(filtered_data[["timestamp", "actual_price", "predicted_price"]])
    else:
        st.warning("🚨 No data available for the selected date. Please choose another date.")

elif view_option == "Full Tested Period":
    st.subheader("📊 Full Test Period: Actual vs. Predicted Prices")
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(test_results["timestamp"], test_results["actual_price"], label="Actual Price", color="red")
    ax.plot(test_results["timestamp"], test_results["predicted_price"], label="Predicted Price", linestyle="dashed", color="blue")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (EUR/MWh)")
    ax.set_title("Electricity Prices Over the Full Tested Period")
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax.legend()
    st.pyplot(fig)

    st.write("📊 **Full Test Period Data:**")
    st.dataframe(test_results[["timestamp", "actual_price", "predicted_price"]])

st.subheader("🔮 Try a Custom Forecast")
st.write("Modify conditions below to generate a new prediction.")

temperature = st.slider("🌡 Temperature (°C)", min_value=-10.0, max_value=40.0, value=10.0, help="Enter expected temperature.")
wind_speed = st.slider("💨 Wind Speed (m/s)", min_value=0.0, max_value=20.0, value=5.0, help="Higher wind speeds can lower electricity prices.")
hour = st.slider("⏳ Hour of the Day", min_value=0, max_value=23, value=12, help="Electricity prices vary by time of day.")

price_lag_1 = test_results["actual_price"].shift(1).fillna(method="ffill").iloc[-1]
price_lag_24 = test_results["actual_price"].shift(24).fillna(method="ffill").iloc[-1]

if st.button("⚡ Predict New Price"):
    input_data = np.array([[temperature, 50, wind_speed, 0, hour, 2, 6, 
                            price_lag_1, price_lag_24, 50, 5, 10]])
    
    predicted_price = model.predict(input_data)[0]
    st.success(f"✅ Predicted Electricity Price: {predicted_price:.2f} EUR/MWh")

st.subheader("📖 How This Model Works")
st.write("""
This model uses **XGBoost**, a machine learning algorithm trained on historical electricity prices from 2018.  
It considers factors like:
- **🌡 Temperature & 💨 Wind Speed** → Impacts energy demand & supply.
- **📈 Past Prices** → Market trends & volatility are key indicators.
- **⏳ Time Factors** → Hourly, daily, and seasonal patterns influence prices.  
""")
