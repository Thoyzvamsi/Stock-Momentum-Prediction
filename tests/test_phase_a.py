import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.preprocessing import StandardScaler

from training.engine.indicators import Indicators
from training.features.engineering import (
    GeneralizedFeatureEngineer,
    GeneralizedTargetEngineer,
)
from training.model_training import DataSplitter, GeneralizedMLTrainer


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


def two_ticker_prices(rows=60):
    first = sample_prices(rows).assign(Ticker="FIRST")
    second = sample_prices(rows).copy()
    for column in ("Open", "High", "Low", "Close"):
        second[column] *= 10
    second["Ticker"] = "SECOND"
    combined = pd.concat([first, second])
    combined["Datetime"] = combined.index
    return combined.sort_values(["Datetime", "Ticker"]).reset_index(drop=True)


class PhaseATests(unittest.TestCase):
    def test_features_have_no_all_nan_columns(self):
        features = GeneralizedFeatureEngineer(sample_prices()).create_all_features()
        self.assertFalse(features.isna().all().any())

    def test_features_do_not_change_when_future_price_changes(self):
        prices = sample_prices()
        baseline = GeneralizedFeatureEngineer(prices).create_all_features()

        changed_prices = prices.copy()
        changed_prices.iloc[-1, changed_prices.columns.get_loc("Close")] *= 2
        changed_prices.iloc[-1, changed_prices.columns.get_loc("High")] *= 2
        changed = GeneralizedFeatureEngineer(changed_prices).create_all_features()

        pd.testing.assert_frame_equal(
            baseline.iloc[:-1], changed.iloc[:-1], check_exact=False, atol=1e-12
        )

    def test_forward_target_does_not_label_missing_future_rows_neutral(self):
        prices = sample_prices(10)
        target = GeneralizedTargetEngineer.create_forward_return_target(
            prices, forward_bars=3
        )
        self.assertTrue(target.iloc[-3:].isna().all())
        self.assertTrue(target.iloc[:-3].notna().all())

    def test_multi_ticker_features_match_independent_calculations(self):
        prices = two_ticker_prices()
        combined_features = GeneralizedFeatureEngineer(prices).create_all_features()

        for _, group in prices.groupby("Ticker", sort=False):
            independent = GeneralizedFeatureEngineer(group).create_all_features()
            pd.testing.assert_frame_equal(
                combined_features.loc[group.index],
                independent,
                check_exact=False,
                atol=1e-12,
            )

    def test_multi_ticker_target_never_crosses_ticker_boundary(self):
        prices = two_ticker_prices(rows=12)
        target = GeneralizedTargetEngineer.create_forward_return_target(
            prices, forward_bars=3
        )
        for _, group in prices.groupby("Ticker", sort=False):
            self.assertTrue(target.loc[group.index[-3:]].isna().all())

    def test_multi_ticker_indicators_match_independent_calculations(self):
        prices = two_ticker_prices()
        combined = Indicators.add_all(prices)

        for _, group in prices.groupby("Ticker", sort=False):
            independent = Indicators.add_all(group)
            pd.testing.assert_frame_equal(
                combined.loc[group.index, ["atr", "adx", "di_plus", "di_minus"]],
                independent[["atr", "adx", "di_plus", "di_minus"]],
                check_exact=False,
                atol=1e-12,
            )

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

    def test_model_save_load_predict_round_trip(self):
        X = pd.DataFrame({"feature": [-4.0, -3.0, -2.0, -1.0, 1.0, 2.0, 3.0, 4.0]})
        y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        scaler = StandardScaler().fit(X)
        X_scaled = pd.DataFrame(
            scaler.transform(X), columns=X.columns, index=X.index
        )
        model = LGBMClassifier(
            n_estimators=5, max_depth=2, min_child_samples=1, verbose=-1, random_state=42
        ).fit(X_scaled, y)
        expected = model.predict(X_scaled)

        with tempfile.TemporaryDirectory() as directory:
            trainer = GeneralizedMLTrainer(model_dir=Path(directory))
            trainer.model = model
            trainer.scaler = scaler
            trainer.training_config = {"feature_names": ["feature"]}
            trainer.save_model("round_trip")

            reloaded = GeneralizedMLTrainer(model_dir=Path(directory))
            loaded_model, loaded_scaler, _ = reloaded.load_model("round_trip")
            scaled = pd.DataFrame(
                loaded_scaler.transform(X), columns=X.columns, index=X.index
            )
            actual = loaded_model.predict(scaled)

        np.testing.assert_array_equal(actual, expected)

    def test_training_scaler_fits_only_training_partition(self):
        X_train = pd.DataFrame({"feature": [1.0, 3.0, 5.0, 7.0]})
        X_test = pd.DataFrame({"feature": [100.0, 200.0]})
        y_train = pd.Series([0, 0, 1, 1])

        with tempfile.TemporaryDirectory() as directory:
            trainer = GeneralizedMLTrainer(model_dir=Path(directory))
            trainer.train_final_model(X_train, y_train)
            np.testing.assert_allclose(trainer.scaler.mean_, [4.0])
            trainer.evaluate(X_test, pd.Series([0, 1]))

    def test_chronological_split_keeps_tickers_together_and_embargoes_gap(self):
        timestamps = pd.date_range("2025-01-01", periods=10, freq="15min")
        rows = pd.DataFrame(
            {
                "feature": np.arange(20),
                "timestamp": [time for time in timestamps for _ in range(2)],
            }
        )
        X = rows[["feature"]]
        y = pd.Series(np.arange(len(rows)) % 2)

        X_train, X_test, _, _ = DataSplitter.train_test_split_timeseries(
            X, y, test_size=0.2, timestamps=rows["timestamp"], gap=2
        )

        train_times = rows.loc[X_train.index, "timestamp"]
        test_times = rows.loc[X_test.index, "timestamp"]
        self.assertEqual(set(train_times), set(timestamps[:6]))
        self.assertEqual(set(test_times), set(timestamps[8:]))
        self.assertFalse(set(train_times) & set(test_times))
        self.assertEqual(len(X_test), 4)


if __name__ == "__main__":
    unittest.main()
