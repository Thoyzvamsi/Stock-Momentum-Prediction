import tempfile
import time

import numpy as np
import pandas as pd

from training.config import (
    FORWARD_BARS,
    SMOKE_MODEL_PARAMS,
    SMOKE_WALK_FORWARD_PARAMS,
    TARGET_THRESHOLD,
)
from training.features.engineering import (
    GeneralizedFeatureEngineer,
    GeneralizedTargetEngineer,
)
from training.model_training import DataSplitter, GeneralizedMLTrainer


def make_synthetic_data():
    rng = np.random.default_rng(42)
    sessions = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=125)
    intraday = pd.to_timedelta(np.arange(26) * 15 + 9 * 60 + 15, unit="m")
    timestamps = pd.DatetimeIndex(
        [session + offset for session in sessions for offset in intraday]
    )
    frames = []

    for ticker_index, ticker in enumerate(("SMOKE_A", "SMOKE_B", "SMOKE_C")):
        returns = rng.normal(0.00005, 0.003, len(timestamps))
        close = (100 + 50 * ticker_index) * np.cumprod(1 + returns)
        open_price = np.r_[close[0], close[:-1]]
        spread = rng.uniform(0.0005, 0.004, len(close))
        frame = pd.DataFrame(
            {
                "Datetime": timestamps,
                "Open": open_price,
                "High": np.maximum(open_price, close) * (1 + spread),
                "Low": np.minimum(open_price, close) * (1 - spread),
                "Close": close,
                "Volume": rng.integers(1_000, 50_000, len(close)),
                "Ticker": ticker,
            }
        )
        frames.append(frame)

    return pd.concat(frames, ignore_index=True).sort_values(
        ["Datetime", "Ticker"], kind="stable"
    ).reset_index(drop=True)


def main():
    started_at = time.monotonic()
    frame = make_synthetic_data()
    print("Smoke training config:")
    print(f"  Tickers: 3 | rows: {len(frame)} | bars: 15m | sessions: 125")
    print(f"  Forward bars: {FORWARD_BARS} | target threshold: {TARGET_THRESHOLD}")
    print(
        f"  Final rounds: {SMOKE_MODEL_PARAMS['n_estimators']} | "
        f"validation rounds: {SMOKE_WALK_FORWARD_PARAMS['n_estimators']}"
    )

    features = GeneralizedFeatureEngineer(frame).create_all_features()
    all_nan_columns = features.columns[features.isna().all()].tolist()
    if all_nan_columns:
        raise AssertionError(f"All-NaN feature columns: {all_nan_columns}")

    changed_frame = frame.copy()
    future_position = len(frame) // 2
    future_ticker = frame.loc[future_position, "Ticker"]
    future_timestamp = frame.loc[future_position, "Datetime"]
    changed_frame.loc[future_position, "Close"] *= 1.5
    changed_frame.loc[future_position, "High"] *= 1.5
    changed_features = GeneralizedFeatureEngineer(
        changed_frame
    ).create_all_features()
    past_rows = (
        frame["Ticker"].eq(future_ticker)
        & frame["Datetime"].lt(future_timestamp)
    )
    pd.testing.assert_frame_equal(
        features.loc[past_rows],
        changed_features.loc[past_rows],
        check_exact=False,
        atol=1e-12,
    )

    target = GeneralizedTargetEngineer.create_forward_return_target(
        frame, forward_bars=FORWARD_BARS, threshold=TARGET_THRESHOLD
    )
    for _, group in frame.groupby("Ticker", sort=False):
        if not target.loc[group.index[-FORWARD_BARS:]].isna().all():
            raise AssertionError("Forward targets crossed a ticker boundary.")

    features = features.dropna()
    frame = frame.loc[features.index]
    target = target.loc[features.index]
    trainer = GeneralizedMLTrainer()
    X, y = trainer.prepare_data(features, target)
    if y.nunique() != 2:
        raise AssertionError(f"Smoke data must produce both classes; got {y.unique()}.")

    timestamps = frame.loc[X.index, "Datetime"]
    trainer.walk_forward_validation(
        X,
        y,
        n_splits=3,
        timestamps=timestamps,
        gap=FORWARD_BARS,
        model_params=SMOKE_WALK_FORWARD_PARAMS,
    )
    X_train, X_test, y_train, y_test = DataSplitter.train_test_split_timeseries(
        X, y, timestamps=timestamps, gap=FORWARD_BARS
    )
    train_times = timestamps.loc[X_train.index]
    test_times = timestamps.loc[X_test.index]
    all_times = pd.Index(timestamps.unique()).sort_values()
    embargo_times = all_times[
        (all_times > train_times.max()) & (all_times < test_times.min())
    ]
    if len(embargo_times) < FORWARD_BARS:
        raise AssertionError("The train/test split does not have a full embargo.")
    print("  Leakage checks: no all-NaN columns, future feature changes, or split overlap")
    trainer.train_final_model(X_train, y_train, model_params=SMOKE_MODEL_PARAMS)
    predictions = trainer.model.predict(
        pd.DataFrame(
            trainer.scaler.transform(X_test),
            columns=X_test.columns,
            index=X_test.index,
        )
    )

    with tempfile.TemporaryDirectory(prefix="stock-momentum-smoke-") as model_dir:
        trainer.model_dir = model_dir
        trainer.save_model("smoke")
        reloaded = GeneralizedMLTrainer(model_dir=model_dir)
        model, scaler, _ = reloaded.load_model("smoke")
        if model is None or scaler is None:
            raise AssertionError("Saved smoke model could not be reloaded.")
        reloaded_predictions = model.predict(
            pd.DataFrame(
                scaler.transform(X_test),
                columns=X_test.columns,
                index=X_test.index,
            )
        )

    if not np.array_equal(predictions, reloaded_predictions):
        raise AssertionError("Model predictions changed after save/reload.")
    if not set(y_test.unique()).issubset({-1, 1}):
        raise AssertionError("Smoke test produced invalid target labels.")

    print(
        f"Smoke test passed: {len(X_train)} train rows, {len(X_test)} "
        f"validation rows, save/load predictions match."
    )
    print(f"Elapsed: {time.monotonic() - started_at:.1f}s")


if __name__ == "__main__":
    main()
