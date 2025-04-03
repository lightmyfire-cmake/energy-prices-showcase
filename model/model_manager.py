import joblib
import pandas as pd

class ModelManager:
    def __init__(self, model_path, data_path):
        self.model = self.load_model(model_path)
        self.test_data = self.load_data(data_path)

    def load_model(self, path):
        return joblib.load(path)

    def load_data(self, path):
        df = pd.read_csv(path)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df

    def predict(self, features):
        return self.model.predict(features)[0]
