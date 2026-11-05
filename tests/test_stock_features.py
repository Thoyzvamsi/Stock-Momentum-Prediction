import unittest

import numpy as np
import pandas as pd

from training.engine.indicators import Indicators
from training.features.engineering import StockFeatureEngineer, StockTargetEngineer


def sample_prices(rows=150):
    index = pd.date_range("2025-01-01", periods=rows, freq="15min")
    close = 100 + np.arange(rows) * 0.1 + np.sin(np.arange(rows) / 4)
    return pd.DataFrame(
        {
            "Open": close - 0.1,
            "High": close + 0.5,
            "Low": close - 0.5,
            "Close": close,
            "Volume": 1000 + np.arange(rows) * 2,
            "Ticker": "TEST",
        },
        index=index,
    )


class StockFeatureTests(unittest.TestCase):
    def test_features_have_no_all_nan_columns(self):
        features = StockFeatureEngineer(sample_prices()).create_all_features()

        self.assertFalse(features.isna().all().any())

    def test_features_do_not_change_when_future_price_changes(self):
        prices = sample_prices()
        baseline = StockFeatureEngineer(prices).create_all_features()

        changed_prices = prices.copy()
        changed_prices.iloc[-1, changed_prices.columns.get_loc("Close")] *= 2
        changed_prices.iloc[-1, changed_prices.columns.get_loc("High")] *= 2
        changed = StockFeatureEngineer(changed_prices).create_all_features()

        pd.testing.assert_frame_equal(
            baseline.iloc[:-1], changed.iloc[:-1], check_exact=False, atol=1e-12
        )

    def test_forward_target_does_not_label_missing_future_rows_neutral(self):
        target = StockTargetEngineer.create_forward_return_target(
            sample_prices(10), forward_bars=3
        )

        self.assertTrue(target.iloc[-3:].isna().all())
        self.assertTrue(target.iloc[:-3].notna().all())

    def test_atr_matches_hand_calculated_values(self):
        prices = pd.DataFrame(
            {
                "High": [10.0, 12.0, 13.0],
                "Low": [8.0, 10.0, 11.0],
                "Close": [9.0, 11.0, 12.0],
            }
        )

        atr = Indicators.atr(prices, period=2)["atr"]

        np.testing.assert_allclose(atr, [2.0, 2.5, 2.5])


if __name__ == "__main__":
    unittest.main()
