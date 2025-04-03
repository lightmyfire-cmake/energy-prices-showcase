import streamlit as st
import numpy as np
from datetime import timedelta
from model.model_manager import ModelManager
from ui.visualizer import Visualizer

class PricePredictorApp:
    def __init__(self):
        self.model_manager = ModelManager("xgboost_model.pkl", "test_predictions.csv")
        self.test_data = self.model_manager.test_data

    def run(self):
        self.render_sidebar()
        self.render_header()
        self.render_view_options()
        self.render_forecast_section()
        self.render_model_explanation()
        self.render_future_improvements()

    def render_sidebar(self):
        st.sidebar.title("🔍 Navigation & Settings")
        self.view_option = st.sidebar.radio("📊 Choose View:", ["Single Date", "Full Tested Period"])
        st.sidebar.subheader("📈 Model Performance")
        st.sidebar.write("Mean Absolute Error (MAE): **1.75 EUR/MWh**")
        st.sidebar.write("Root Mean Squared Error (RMSE): **4.75 EUR/MWh**")

    def render_header(self):
        st.title("⚡ Electricity Price Prediction")
        with st.expander("📜 Click here to learn about the model, data, and engineering process"):
            st.markdown("""
            - **Data Sources:** Open Power System Data & Meteostat  
            - **Features:** Time-based, weather, lagged prices  
            - **Model:** XGBoost with grid search optimization  
            """)

        st.markdown("### 📈 Real-world forecasting with historical Austrian energy data")

    def render_view_options(self):
        if self.view_option == "Single Date":
            self.render_single_date_view()
        else:
            self.render_full_period_view()

    def render_single_date_view(self):
        min_date = self.test_data["timestamp"].min().date()
        max_date = self.test_data["timestamp"].max().date()
        selected_date = st.date_input("📅 Select a Date", min_value=min_date, max_value=max_date, value=min_date + timedelta(days=1))

        filtered_data = self.test_data[self.test_data["timestamp"].dt.date == selected_date]
        if not filtered_data.empty:
            st.subheader(f"📊 Prices on {selected_date}")
            Visualizer.plot_prices(filtered_data, f"Electricity Prices on {selected_date}")
            st.dataframe(filtered_data[["timestamp", "actual_price", "predicted_price"]])
        else:
            st.warning("🚨 No data for the selected date.")

    def render_full_period_view(self):
        st.subheader("📊 Full Test Period: Actual vs. Predicted")
        Visualizer.plot_prices(self.test_data, "Electricity Prices Over the Full Tested Period")
        st.dataframe(self.test_data[["timestamp", "actual_price", "predicted_price"]])

    def render_forecast_section(self):
        st.subheader("🔮 Try a Custom Forecast")
        temperature = st.slider("🌡 Temperature (°C)", -10.0, 40.0, 10.0)
        wind_speed = st.slider("💨 Wind Speed (m/s)", 0.0, 20.0, 5.0)
        hour = st.slider("⏳ Hour of the Day", 0, 23, 12)

        price_lag_1 = self.test_data["actual_price"].shift(1).ffill().iloc[-1]
        price_lag_24 = self.test_data["actual_price"].shift(24).ffill().iloc[-1]


        if st.button("⚡ Predict New Price"):
            input_data = np.array([[temperature, 50, wind_speed, 0, hour, 2, 6,
                                    price_lag_1, price_lag_24, 50, 5, 10]])
            prediction = self.model_manager.predict(input_data)
            st.success(f"✅ Predicted Electricity Price: {prediction:.2f} EUR/MWh")

    def render_model_explanation(self):
        st.subheader("📖 How This Model Works")
        st.write("""
        - Weather, past prices, and time influence prices  
        - XGBoost predicts trends with past values  
        """)

    def render_future_improvements(self):
        st.subheader("🚀 Future Improvements")
        st.markdown("""
        - Outlier detection  
        - More features (consumption trends, sentiment)  
        - Model tuning & automation  
        """)
