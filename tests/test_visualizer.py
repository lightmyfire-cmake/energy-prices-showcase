import unittest
import pandas as pd
from datetime import datetime
from ui.visualizer import Visualizer

class TestVisualizer(unittest.TestCase):
    def test_plot_prices_runs(self):
        # Minimal fake data
        df = pd.DataFrame({
            "timestamp": [datetime(2024, 1, 1, h) for h in range(3)],
            "actual_price": [40, 42, 41],
            "predicted_price": [39, 43, 40]
        })
        try:
            Visualizer.plot_prices(df, "Test Plot")
        except Exception as e:
            self.fail(f"plot_prices() raised an exception: {e}")

if __name__ == "__main__":
    unittest.main()
