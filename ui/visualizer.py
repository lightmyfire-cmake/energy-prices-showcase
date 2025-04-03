import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import streamlit as st

class Visualizer:
    @staticmethod
    def plot_prices(data, title):
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(data["timestamp"], data["actual_price"], label="Actual Price", color="red")
        ax.plot(data["timestamp"], data["predicted_price"], label="Predicted Price", linestyle="dashed", color="blue")
        ax.set_xlabel("Time")
        ax.set_ylabel("Price (EUR/MWh)")
        ax.set_title(title)
        ax.legend()
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b %Y'))
        st.pyplot(fig)
