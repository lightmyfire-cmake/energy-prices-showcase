import unittest
import numpy as np
from model.model_manager import ModelManager

class TestModelManager(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manager = ModelManager("xgboost_model.pkl", "test_predictions.csv")

    def test_model_loaded(self):
        self.assertIsNotNone(self.manager.model)

    def test_data_loaded(self):
        self.assertFalse(self.manager.test_data.empty)
        self.assertIn("timestamp", self.manager.test_data.columns)

    def test_prediction_shape(self):
        input_data = np.array([[10, 50, 5, 0, 12, 2, 6, 50, 50, 50, 5, 10]])
        prediction = self.manager.predict(input_data)
        self.assertIsInstance(prediction, (float, np.floating))

if __name__ == "__main__":
    unittest.main()
