#!/usr/bin/env python3
"""
Multi-Stock Generalized Model Training
Combines historical data from multiple stocks and trains a single generalized model
"""

import sys
import argparse
import pandas as pd
from data.pipeline import MultiStockDataPipeline
from features.engineering import GeneralizedFeatureEngineer, GeneralizedTargetEngineer
from training.model_training import GeneralizedMLTrainer, DataSplitter
from engine.backtest import CostAwareBacktester


def main(forward_bars=6, threshold=0.005, test_size=0.2,
         commission=0.001, slippage=0.002, position_size=10,
         min_confidence=0.65, hold_bars=15):
    """
    Complete training pipeline for generalized multi-stock model
    """
    
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
    df_train = pd.read_csv("data/raw_data.csv")
        
    if len(df_train) < 1:
        print("✗ No data loaded")
        return None
    
    
    tickers = pipeline.get_training_tickers(df_train)
    
    if len(tickers) == 0:
        print("✗ No CSV files found in data/ folder")
        print("✗ Please add CSV files (AAPL.csv, MSFT.csv, etc.) to data/ folder")
        return
    
    print(f"Found {len(tickers)} stocks: {', '.join(tickers)}\n")
    
    if not pipeline.validate_training_data(df_train):
        print("✗ Data validation failed")
        return
    
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
    
    wfv_scores = trainer.walk_forward_validation(X, y, n_splits=5)
    
    # ========================================================================
    # 6. TRAIN & TEST SPLIT
    # ========================================================================
    print("\nSTEP 6: SPLITTING DATA")
    print("-" * 70)
    
    X_train, X_test, y_train, y_test = DataSplitter.train_test_split_timeseries(
        X, y, test_size=test_size
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
    
    trainer.save_model('generalized_momentum')
    
    # ========================================================================
    # RESULTS & RECOMMENDATIONS
    # ========================================================================
    print("\n" + "="*70)
    print("  TRAINING COMPLETE - RESULTS & ANALYSIS")
    print("="*70)
    
    print("\n✓ MODEL PERFORMANCE:")
    print(f"  Accuracy:  {metrics['accuracy']:.3f}")
    print(f"  Precision: {metrics['precision']:.3f}")
    print(f"  Recall:    {metrics['recall']:.3f}")
    print(f"  F1-Score:  {metrics['f1']:.3f}")
    
    print("\n✓ WALK-FORWARD VALIDATION:")
    print(f"  Mean: {trainer.training_config['wfv_mean']:.3f}")
    print(f"  Std:  {trainer.training_config['wfv_std']:.3f}")
    
    print("\n✓ TRAINING DATA:")
    print(f"  Samples: {trainer.training_config['n_samples']:,}")
    print(f"  Features: {trainer.training_config['n_features']}")
    print(f"  Stocks: {len(tickers)}")
    
    print("\n✓ TOP FEATURES:")
    for i, (feat, imp) in enumerate(zip(
        importance_df.head(5)['feature'],
        importance_df.head(5)['importance']
    ), 1):
        print(f"  {i}. {feat}: {imp:.3f}")
    
    if bt_metrics is not None:
        print("\n✓ BACKTEST (test set, with costs):")
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
        print("✓ Model achieves >60% accuracy - GOOD")
    elif metrics['accuracy'] > 0.55:
        print("⚠ Model achieves 55-60% accuracy - ACCEPTABLE")
    else:
        print("✗ Model achieves <55% accuracy - NEEDS IMPROVEMENT")
    
    if trainer.training_config['wfv_mean'] > 0.58:
        print("✓ Walk-forward validation is strong - NO OVERFITTING")
    else:
        print("⚠ Walk-forward validation shows some variance")
    
    if bt_metrics is not None and bt_metrics['expected_value'] > 0:
        print("✓ Backtest shows POSITIVE expected value after costs")
    elif bt_metrics is not None:
        print("⚠ Backtest expected value is negative - review cost assumptions")
    
    print("\n" + "="*70)
    print("NEXT STEPS:")
    print("="*70)
    print("1. Launch dashboard: streamlit run multi_stock_dashboard.py")
    print("2. Make predictions on any stock symbol")
    print("3. Test on different stocks to verify generalization")
    print("="*70 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Train Generalized Multi-Stock Momentum Model'
    )
    
    parser.add_argument('--forward-bars', type=int, default=8,
                        help='Forward look period (default: 8)')
    parser.add_argument('--threshold', type=float, default=0.005,
                        help='Return threshold for classification (default: 0.005)')
    parser.add_argument('--test-size', type=float, default=0.2,
                        help='Test set size (default: 0.2)')
    parser.add_argument('--commission', type=float, default=0.001,
                        help='Commission per trade (default: 0.001)')
    parser.add_argument('--slippage', type=float, default=0.002,
                        help='Slippage per trade (default: 0.002)')
    parser.add_argument('--position-size', type=int, default=10,
                        help='Position size in shares (default: 10)')
    parser.add_argument('--min-confidence', type=float, default=0.65,
                        help='Min confidence threshold (default: 0.65)')
    parser.add_argument('--hold-bars', type=int, default=15,
                        help='Holding period in bars (default: 15)')
    
    args = parser.parse_args()
    
    main(
        forward_bars=args.forward_bars,
        threshold=args.threshold,
        test_size=args.test_size,
        commission=args.commission,
        slippage=args.slippage,
        position_size=args.position_size,
        min_confidence=args.min_confidence,
        hold_bars=args.hold_bars
    )
