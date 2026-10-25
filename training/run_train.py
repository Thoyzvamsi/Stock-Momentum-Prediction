#!/usr/bin/env python3
"""
Multi-Stock Generalized Model Training
Combines historical data from multiple stocks and trains a single generalized model
"""

import argparse
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime
import json
import pandas as pd
import sys
import time
import traceback
from training.config import (
    COMMISSION_RATE,
    EXPERIMENT_RUNS_DIR,
    LOGS_DIR,
    FORWARD_BARS,
    MIN_CONFIDENCE,
    MODEL_NAME,
    RAW_DATA_FILE,
    SLIPPAGE_RATE,
    TARGET_THRESHOLD,
    TEST_SIZE,
)
from training.data.pipeline import MultiStockDataPipeline
from training.features.engineering import GeneralizedFeatureEngineer, GeneralizedTargetEngineer
from training.model_training import GeneralizedMLTrainer, DataSplitter
from training.engine.backtest import CostAwareBacktester


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for stream in self.streams:
            try:
                stream.write(text)
            except UnicodeEncodeError:
                encoding = stream.encoding or "utf-8"
                stream.write(text.encode(encoding, errors="replace").decode(encoding))
            stream.flush()
        return len(text)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def write_experiment_record(run_id, started_at, status, settings, result=None, error=None):
    EXPERIMENT_RUNS_DIR.mkdir(parents=True, exist_ok=True)
    report = {
        "run_id": run_id,
        "started_at": started_at.isoformat(timespec="seconds"),
        "status": status,
        "settings": settings,
        "result": result,
        "error": error,
    }
    report_path = EXPERIMENT_RUNS_DIR / f"{run_id}_generalized_momentum.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    log_path = EXPERIMENT_RUNS_DIR.parent / "LOG.md"
    previous_log = log_path.read_text(encoding="utf-8") if log_path.exists() else ""
    summary = (
        f"## {started_at.strftime('%Y-%m-%d %H:%M:%S %Z')} — "
        f"Generalized momentum training ({status})\n\n"
        f"- Run report: `runs/{report_path.name}`\n"
        f"- Training log: `../training/logs/{run_id}.log`\n"
    )
    if result:
        metrics = result["metrics"]
        summary += (
            f"- Data: {result['data']['rows']:,} rows, "
            f"{result['data']['tickers']} tickers; "
            f"{result['data']['samples']:,} labeled samples "
            f"({result['data']['start']} to {result['data']['end']})\n"
            f"- Walk-forward accuracy: {result['walk_forward']['mean']:.3f} "
            f"(± {result['walk_forward']['std']:.3f})\n"
            f"- Holdout accuracy: {metrics['accuracy']:.3f}; "
            f"F1: {metrics['f1']:.3f}\n"
        )
        if result.get("backtest"):
            summary += (
                f"- Backtest after costs: return "
                f"{result['backtest']['total_return']:.1%}, "
                f"Sharpe {result['backtest']['sharpe_ratio']:.2f}, "
                f"max drawdown {result['backtest']['max_drawdown']:.1%}, "
                f"{result['backtest']['total_trades']} trades\n"
            )
    if error:
        summary += f"- Error: `{error}`\n"
    log_path.write_text(
        summary + "\n" + previous_log, encoding="utf-8"
    )
    return report_path


def main(forward_bars=FORWARD_BARS, threshold=TARGET_THRESHOLD,
         test_size=TEST_SIZE, commission=COMMISSION_RATE,
         slippage=SLIPPAGE_RATE, position_size=10,
         min_confidence=MIN_CONFIDENCE, hold_bars=15):
    """
    Complete training pipeline for generalized multi-stock model
    """
    started = time.monotonic()
    
    print("\n" + "="*70)
    print("  GENERALIZED MULTI-STOCK MOMENTUM PREDICTION")
    print("  Training a model that works across multiple stocks")
    print("="*70 + "\n")
    
    # ========================================================================
    # 1. DATA LOADING
    # ========================================================================
    print("STEP 1: LOADING MULTI-STOCK TRAINING DATA")
    print("-" * 70)
    
    pipeline = MultiStockDataPipeline()
    df_train = pd.read_csv(RAW_DATA_FILE)
    if 'Datetime' not in df_train.columns:
        raise ValueError("Training data must include a Datetime column.")
    df_train['Datetime'] = pd.to_datetime(df_train['Datetime'])
    df_train = df_train.sort_values(
        ['Datetime', 'Ticker'], kind='stable'
    ).reset_index(drop=True)
        
    if len(df_train) < 1:
        raise ValueError(f"No training data found at {RAW_DATA_FILE}.")
    
    
    tickers = pipeline.get_training_tickers(df_train)
    
    if len(tickers) == 0:
        raise ValueError("The training dataset contains no valid tickers.")
    
    print(f"Found {len(tickers)} stocks: {', '.join(tickers)}\n")
    
    if not pipeline.validate_training_data(df_train):
        raise ValueError("Training data validation failed.")
    
    # ========================================================================
    # 2. FEATURE ENGINEERING
    # ========================================================================
    print("\nSTEP 2: ENGINEERING GENERALIZED FEATURES")
    print("-" * 70)
    
    fe = GeneralizedFeatureEngineer(df_train)
    features = fe.create_all_features()
    
    features_clean = features.dropna()
    df_train_clean = df_train.loc[features_clean.index]
    
    print(f"Feature matrix shape: {features_clean.shape}")
    
    # ========================================================================
    # 3. TARGET CREATION
    # ========================================================================
    print("\nSTEP 3: CREATING TARGET VARIABLE")
    print("-" * 70)
    
    target = GeneralizedTargetEngineer.create_forward_return_target(
        df_train_clean,
        forward_bars=forward_bars,
        threshold=threshold
    )
    
    target_dist = target.value_counts().to_dict()
    print(f"Target distribution: {target_dist}")
    
    # ========================================================================
    # 4. DATA PREPARATION
    # ========================================================================
    print("\nSTEP 4: PREPARING DATA FOR ML")
    print("-" * 70)
    
    trainer = GeneralizedMLTrainer()
    X, y = trainer.prepare_data(features_clean, target)
    
    # ========================================================================
    # 5. WALK-FORWARD VALIDATION
    # ========================================================================
    print("\nSTEP 5: WALK-FORWARD VALIDATION")
    print("-" * 70)
    print("(Proves no lookahead bias and realistic performance)\n")
    
    timestamps = df_train_clean.loc[X.index, 'Datetime']
    trainer.walk_forward_validation(
        X, y, n_splits=5, timestamps=timestamps, gap=forward_bars
    )
    
    # ========================================================================
    # 6. TRAIN & TEST SPLIT
    # ========================================================================
    print("\nSTEP 6: SPLITTING DATA")
    print("-" * 70)
    
    X_train, X_test, y_train, y_test = DataSplitter.train_test_split_timeseries(
        X,
        y,
        test_size=test_size,
        timestamps=timestamps,
        gap=forward_bars,
    )
    
    # ========================================================================
    # 7. TRAIN FINAL MODEL
    # ========================================================================
    print("\nSTEP 7: TRAINING GENERALIZED MODEL")
    print("-" * 70)
    
    trainer.train_final_model(X_train, y_train)
    
    # ========================================================================
    # 8. EVALUATE
    # ========================================================================
    print("\nSTEP 8: EVALUATING MODEL")
    print("-" * 70)
    
    metrics = trainer.evaluate(X_test, y_test)
    
    # ========================================================================
    # 9. FEATURE IMPORTANCE
    # ========================================================================
    print("\nSTEP 9: FEATURE IMPORTANCE")
    print("-" * 70)
    
    importance_df = trainer.feature_importance(top_n=15)
    
    # ========================================================================
    # 10. BACKTESTING (with transaction costs)
    # ========================================================================
    print("\nSTEP 10: BACKTESTING WITH TRANSACTION COSTS")
    print("-" * 70)
    print(f"  Commission: {commission*100:.2f}%  |  Slippage: {slippage*100:.2f}%")
    print(f"  Position size: {position_size} shares  |  Min confidence: {min_confidence:.0%}")
    print(f"  Hold period: {hold_bars} bars\n")

    # Use the test portion of the original (pre-prepare) data so we have OHLCV prices
    # X_test index tells us which rows of df_train_clean belong to the test window
    df_backtest = df_train_clean.loc[X_test.index]

    backtester = CostAwareBacktester(
        df=df_backtest,
        features_df=X_test,
        model=trainer.model,
        scaler=trainer.scaler,
        commission=commission,
        slippage=slippage,
        min_confidence=min_confidence,
    )

    bt_metrics = backtester.run()
    backtester.print_results(bt_metrics)

    # ========================================================================
    # 11. SAVE MODEL
    # ========================================================================
    print("\nSTEP 11: SAVING MODEL ARTIFACTS")
    print("-" * 70)
    
    trainer.save_model(MODEL_NAME)
    
    # ========================================================================
    # RESULTS & RECOMMENDATIONS
    # ========================================================================
    print("\n" + "="*70)
    print("  TRAINING COMPLETE - RESULTS & ANALYSIS")
    print("="*70)
    
    print("\nMODEL PERFORMANCE:")
    print(f"  Accuracy:  {metrics['accuracy']:.3f}")
    print(f"  Precision: {metrics['precision']:.3f}")
    print(f"  Recall:    {metrics['recall']:.3f}")
    print(f"  F1-Score:  {metrics['f1']:.3f}")
    
    print("\nWALK-FORWARD VALIDATION:")
    print(f"  Mean: {trainer.training_config['wfv_mean']:.3f}")
    print(f"  Std:  {trainer.training_config['wfv_std']:.3f}")
    
    print("\nTRAINING DATA:")
    print(f"  Samples: {trainer.training_config['n_samples']:,}")
    print(f"  Features: {trainer.training_config['n_features']}")
    print(f"  Stocks: {len(tickers)}")
    
    print("\nTOP FEATURES:")
    for i, (feat, imp) in enumerate(zip(
        importance_df.head(5)['feature'],
        importance_df.head(5)['importance']
    ), 1):
        print(f"  {i}. {feat}: {imp:.3f}")
    
    if bt_metrics is not None:
        print("\nBACKTEST (test set, with costs):")
        print(f"  Win Rate:      {bt_metrics['win_rate']:.1%}")
        print(f"  Total Return:  {bt_metrics['total_return']:.1%}")
        print(f"  Sharpe Ratio:  {bt_metrics['sharpe_ratio']:.2f}")
        print(f"  Max Drawdown:  {bt_metrics['max_drawdown']:.1%}")
        print(f"  Profit Factor: {bt_metrics['profit_factor']:.2f}x")
        print(f"  Total Trades:  {bt_metrics['total_trades']}")
    
    # Assessment
    print("\n" + "-"*70)
    print("ASSESSMENT:")
    
    if metrics['accuracy'] > 0.60:
        print("Model achieves >60% accuracy - GOOD")
    elif metrics['accuracy'] > 0.55:
        print("Model achieves 55-60% accuracy - ACCEPTABLE")
    else:
        print("Model achieves <55% accuracy - NEEDS IMPROVEMENT")
    
    if trainer.training_config['wfv_mean'] > 0.58:
        print("Walk-forward validation is strong - NO OVERFITTING")
    else:
        print("Walk-forward validation shows some variance")
    
    if bt_metrics is not None and bt_metrics['expected_value'] > 0:
        print("Backtest shows POSITIVE expected value after costs")
    elif bt_metrics is not None:
        print("Backtest expected value is negative - review cost assumptions")
    
    print("\n" + "="*70)
    print("NEXT STEPS:")
    print("="*70)
    print("1. Launch dashboard: streamlit run dashboard.py")
    print("2. Make predictions on any stock symbol")
    print("3. Test on different stocks to verify generalization")
    elapsed_seconds = time.monotonic() - started
    print(f"Total training elapsed: {elapsed_seconds:.1f} seconds")
    print("="*70 + "\n")

    return {
        "data": {
            "rows": int(len(df_train)),
            "tickers": int(len(tickers)),
            "start": str(df_train["Datetime"].min()),
            "end": str(df_train["Datetime"].max()),
            "samples": int(trainer.training_config["n_samples"]),
            "features": int(trainer.training_config["n_features"]),
        },
        "walk_forward": {
            "mean": float(trainer.training_config["wfv_mean"]),
            "std": float(trainer.training_config["wfv_std"]),
        },
        "metrics": {
            key: float(metrics[key])
            for key in ("accuracy", "precision", "recall", "f1")
        },
        "backtest": (
            {
                "total_return": float(bt_metrics["total_return"]),
                "sharpe_ratio": float(bt_metrics["sharpe_ratio"]),
                "max_drawdown": float(bt_metrics["max_drawdown"]),
                "win_rate": float(bt_metrics["win_rate"]),
                "total_trades": int(bt_metrics["total_trades"]),
            }
            if bt_metrics is not None
            else None
        ),
        "model": MODEL_NAME,
        "elapsed_seconds": round(elapsed_seconds, 1),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Train Generalized Multi-Stock Momentum Model'
    )
    
    parser.add_argument('--forward-bars', type=int, default=FORWARD_BARS,
                        help='Forward look period (default: 8)')
    parser.add_argument('--threshold', type=float, default=TARGET_THRESHOLD,
                        help='Return threshold for classification (default: 0.005)')
    parser.add_argument('--test-size', type=float, default=TEST_SIZE,
                        help='Test set size (default: 0.2)')
    parser.add_argument('--commission', type=float, default=COMMISSION_RATE,
                        help='Commission per trade (default: 0.001)')
    parser.add_argument('--slippage', type=float, default=SLIPPAGE_RATE,
                        help='Slippage per trade (default: 0.002)')
    parser.add_argument('--position-size', type=int, default=10,
                        help='Position size in shares (default: 10)')
    parser.add_argument('--min-confidence', type=float, default=MIN_CONFIDENCE,
                        help='Min confidence threshold (default: 0.65)')
    parser.add_argument('--hold-bars', type=int, default=15,
                        help='Holding period in bars (default: 15)')
    
    args = parser.parse_args()
    
    settings = {
        "forward_bars": args.forward_bars,
        "threshold": args.threshold,
        "test_size": args.test_size,
        "commission": args.commission,
        "slippage": args.slippage,
        "position_size": args.position_size,
        "min_confidence": args.min_confidence,
        "hold_bars": args.hold_bars,
    }
    started_at = datetime.now().astimezone()
    run_id = started_at.strftime("%Y%m%d_%H%M%S")
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOGS_DIR / f"{run_id}.log"

    with log_path.open("w", encoding="utf-8") as log_file:
        tee = Tee(sys.stdout, log_file)
        try:
            with redirect_stdout(tee), redirect_stderr(tee):
                print(f"Training log: {log_path}")
                print(f"Run settings: {json.dumps(settings, sort_keys=True)}")
                result = main(**settings)
                report_path = write_experiment_record(
                    run_id, started_at, "completed", settings, result=result
                )
                print(f"Experiment report: {report_path}")
        except Exception as error:
            with redirect_stdout(tee), redirect_stderr(tee):
                write_experiment_record(
                    run_id,
                    started_at,
                    "failed",
                    settings,
                    error=f"{type(error).__name__}: {error}",
                )
                traceback.print_exc()
            raise
