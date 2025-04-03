# ⚡ Electricity Price Prediction App

An interactive Streamlit web app for visualizing and forecasting Austrian electricity prices using historical data and machine learning (XGBoost).

---

## 📦 Features

- 📈 Visualize actual vs. predicted electricity prices (single date or full range)
- 🔮 Predict custom prices with adjustable weather and time inputs
- 🤖 Powered by XGBoost trained on real data from Open Power System Data & Meteostat
- 📊 View model performance (MAE, RMSE)
- 🚀 Roadmap for future improvements included

---

## 🛠️ Tech Stack

- [Python](https://www.python.org/)
- [Streamlit](https://streamlit.io/)
- [XGBoost](https://xgboost.readthedocs.io/)
- [Pandas, NumPy, Matplotlib](https://pandas.pydata.org/)

---

## 🧪 Running the App Locally

### 1. Clone the repo

```bash
git clone https://github.com/your-username/energy-prices-showcase.git
cd energy-prices-showcase
```

### 2. Create & activate a virtual environment

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Launch the app

```bash
streamlit run app.py
```

---

## 📁 Project Structure

```
energy-prices-showcase/
├── app.py                  # Entry point
├── core/                   # Streamlit app logic
│   └── price_predictor.py
├── model/                  # Model loading & prediction logic
│   └── model_manager.py
├── ui/                     # Plotting / visualization utilities
│   └── visualizer.py
├── tests/                  # Unit tests
│   ├── test_model_manager.py
│   └── test_visualizer.py
├── test_predictions.csv    # Model output for test set
├── xgboost_model.pkl       # Pre-trained model
├── requirements.txt
└── README.md               # This file
```

---

## ✅ Unit Tests

To run all tests:

```bash
python -m unittest discover tests
```

---

## 📈 Model Performance

| Metric | Value        |
|--------|--------------|
| MAE    | 1.75 EUR/MWh |
| RMSE   | 4.75 EUR/MWh |

---

## 📚 Data Sources

- [Open Power System Data](https://data.open-power-system-data.org/)
- [Meteostat Weather API](https://meteostat.net/)

---

## 💡 Future Improvements

- 🔹 Outlier detection & anomaly handling
- 🔹 Advanced feature engineering (economic trends, holidays)
- 🔹 Sentiment analysis from news & market reports
- 🔹 Real-time weather & market integration (API-based)
- 🔹 Model ensembling and automated retraining pipelines