import pandas as pd
import numpy as np
from datetime import datetime


class CostAwareBacktester:
    """
    Backtest trading strategy with REALISTIC transaction costs.
    Works with the generalized multi-stock model.
    """

    def __init__(self, df, features_df, model, scaler,
                 commission=0.001, slippage=0.002,
                 position_size=10, min_confidence=0.65,
                 hold_bars=15):
        """
        Args:
            df           : OHLCV DataFrame (must contain 'Close'; optionally 'Ticker')
            features_df  : Already-scaled feature DataFrame aligned to df
            model        : Trained LightGBM classifier
            scaler       : Fitted StandardScaler (used only when features are NOT yet scaled)
            commission   : One-way commission rate  (default 0.001 = 0.1 %)
            slippage     : One-way slippage estimate (default 0.002 = 0.2 %)
            position_size: Shares per trade
            min_confidence: Minimum predicted-class probability required to trade
            hold_bars    : Number of bars to hold each position
        """
        self.df = df.reset_index(drop=False)   # keep original index as a column if needed
        self.features = features_df.reset_index(drop=True)
        self.model = model
        self.scaler = scaler

        self.commission = commission
        self.slippage = slippage
        self.total_cost_per_side = commission + slippage
        self.round_trip_cost = self.total_cost_per_side * 2

        self.position_size = position_size
        self.min_confidence = min_confidence
        self.hold_bars = hold_bars

        self.trades = []
        self.equity_curve = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self):
        """Run backtest and return metrics dict (or None if no trades fired)."""
        print("Running backtest with transaction costs...")

        # features_df arriving from multi_stock_main is already scaled by the
        # trainer's scaler, so we use it directly.
        predictions = self.model.predict(self.features)
        probabilities = self.model.predict_proba(self.features)
        pred_confidence = probabilities.max(axis=1)

        equity = 100_000.0   # Starting capital $100 k

        has_ticker_col = 'Ticker' in self.df.columns

        for i in range(len(self.df) - self.hold_bars):
            pred = predictions[i]
            prob = pred_confidence[i]

            # Filter by confidence and direction
            if prob < self.min_confidence:
                continue
            if pred not in (-1, 1):
                continue

            entry_price = self.df['Close'].iloc[i]
            exit_bar   = min(i + self.hold_bars, len(self.df) - 1)
            exit_price  = self.df['Close'].iloc[exit_bar]

            # ---- P&L with round-trip costs ----
            if pred == 1:   # Long
                entry_with_cost = entry_price * (1 + self.total_cost_per_side)
                exit_with_cost  = exit_price  * (1 - self.total_cost_per_side)
                pnl = (exit_with_cost - entry_with_cost) * self.position_size
            else:            # Short
                entry_with_cost = entry_price * (1 - self.total_cost_per_side)
                exit_with_cost  = exit_price  * (1 + self.total_cost_per_side)
                pnl = (entry_with_cost - exit_with_cost) * self.position_size

            pnl_pct_actual = pnl / (entry_price * self.position_size)

            trade = {
                'entry_bar'    : i,
                'entry_date'   : self.df.index[i],
                'entry_price'  : entry_price,
                'exit_bar'     : exit_bar,
                'exit_date'    : self.df.index[exit_bar],
                'exit_price'   : exit_price,
                'direction'    : 'LONG' if pred == 1 else 'SHORT',
                'confidence'   : prob,
                'holding_bars' : exit_bar - i,
                'pnl'          : pnl,
                'pnl_pct'      : pnl_pct_actual,
                'profit'       : 1 if pnl > 0 else 0,
            }
            if has_ticker_col:
                trade['ticker'] = self.df['Ticker'].iloc[i]

            self.trades.append(trade)
            equity += pnl
            self.equity_curve.append(equity)

        if not self.trades:
            print("✗ No trades fired — check min_confidence or hold_bars settings")
            return None

        print(f"✓ Backtest complete: {len(self.trades)} trades executed")
        return self._calculate_metrics()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _calculate_metrics(self):
        """Compute comprehensive backtest statistics."""
        trades_df = pd.DataFrame(self.trades)

        total_trades   = len(trades_df)
        winning_trades = trades_df[trades_df['pnl'] > 0]
        losing_trades  = trades_df[trades_df['pnl'] < 0]

        win_rate   = len(winning_trades) / total_trades if total_trades else 0
        total_pnl  = trades_df['pnl'].sum()
        avg_trade  = trades_df['pnl'].mean()

        # Profit factor
        gross_profit = winning_trades['pnl'].sum() if len(winning_trades) else 0
        gross_loss   = abs(losing_trades['pnl'].sum()) if len(losing_trades) else 0
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float('inf')

        # Max drawdown
        equity_array = np.array([100_000.0] + self.equity_curve)
        running_max  = np.maximum.accumulate(equity_array)
        drawdown     = (equity_array - running_max) / running_max
        max_drawdown = float(np.min(drawdown))

        # Return / annualised
        initial_capital = 100_000.0
        final_equity    = float(equity_array[-1])
        total_return    = (final_equity - initial_capital) / initial_capital

        trading_days = (trades_df['exit_date'].max() - trades_df['entry_date'].min())
        years        = max(trading_days / 252, 1e-6)
        annual_return = total_return / years

        # Sharpe (trade-level approximation)
        pnl_array = trades_df['pnl'].values
        if len(pnl_array) > 1 and pnl_array.std() > 0:
            unit_returns = pnl_array / (initial_capital / len(pnl_array))
            sharpe = (unit_returns.mean() / unit_returns.std()) * np.sqrt(252)
        else:
            sharpe = 0.0

        avg_win  = float(winning_trades['pnl'].mean()) if len(winning_trades) else 0.0
        avg_loss = float(losing_trades['pnl'].mean())  if len(losing_trades)  else 0.0
        expected_value = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)

        # Per-ticker breakdown (when available)
        ticker_stats = {}
        if 'ticker' in trades_df.columns:
            for ticker, grp in trades_df.groupby('ticker'):
                wins = grp[grp['pnl'] > 0]
                ticker_stats[ticker] = {
                    'trades'  : len(grp),
                    'win_rate': len(wins) / len(grp),
                    'total_pnl': grp['pnl'].sum(),
                }

        return {
            'total_trades'  : total_trades,
            'winning_trades': len(winning_trades),
            'losing_trades' : len(losing_trades),
            'win_rate'      : win_rate,
            'total_pnl'     : total_pnl,
            'avg_trade'     : avg_trade,
            'profit_factor' : profit_factor,
            'max_drawdown'  : max_drawdown,
            'total_return'  : total_return,
            'annual_return' : annual_return,
            'sharpe_ratio'  : sharpe,
            'avg_win'       : avg_win,
            'avg_loss'      : avg_loss,
            'expected_value': expected_value,
            'final_equity'  : final_equity,
            'ticker_stats'  : ticker_stats,
            'trades_df'     : trades_df,
        }

    def print_results(self, metrics):
        """Pretty-print backtest results."""
        if metrics is None:
            print("✗ Backtest failed — no valid trades")
            return

        print("\n" + "="*55)
        print("  BACKTEST RESULTS (WITH TRANSACTION COSTS)")
        print("="*55)

        print(f"\nTrade Summary:")
        print(f"  Total Trades   : {metrics['total_trades']}")
        print(f"  Winning Trades : {metrics['winning_trades']}")
        print(f"  Losing Trades  : {metrics['losing_trades']}")
        print(f"  Win Rate       : {metrics['win_rate']:.1%}")

        print(f"\nProfitability:")
        print(f"  Total P&L      : ${metrics['total_pnl']:>12,.2f}")
        print(f"  Avg Trade      : ${metrics['avg_trade']:>12,.2f}")
        print(f"  Avg Win        : ${metrics['avg_win']:>12,.2f}")
        print(f"  Avg Loss       : ${metrics['avg_loss']:>12,.2f}")
        pf = metrics['profit_factor']
        print(f"  Profit Factor  : {pf:.2f}x" if pf != float('inf') else "  Profit Factor  : ∞ (no losing trades)")

        print(f"\nReturn Metrics:")
        print(f"  Total Return   : {metrics['total_return']:.1%}")
        print(f"  Annual Return  : {metrics['annual_return']:.1%}")
        print(f"  Sharpe Ratio   : {metrics['sharpe_ratio']:.2f}")
        print(f"  Max Drawdown   : {metrics['max_drawdown']:.1%}")
        print(f"  Final Equity   : ${metrics['final_equity']:>12,.2f}")

        print(f"\nExpectation:")
        print(f"  Expected Value : ${metrics['expected_value']:>12,.2f}")

        if metrics['ticker_stats']:
            print(f"\nPer-Ticker Breakdown:")
            for ticker, stats in sorted(metrics['ticker_stats'].items()):
                print(f"  {ticker:6s}  trades={stats['trades']:4d}  "
                      f"win={stats['win_rate']:.0%}  "
                      f"P&L=${stats['total_pnl']:>10,.2f}")

        print()
        if metrics['expected_value'] > 0:
            print("✓ Strategy is PROFITABLE after transaction costs")
        else:
            print("✗ Strategy is NOT PROFITABLE after transaction costs")

        print("="*55)


# ---------------------------------------------------------------------------
# Standalone entry point — uses the multi-stock pipeline
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from data.pipeline import MultiStockDataPipeline
    from features.engineering import GeneralizedFeatureEngineer, GeneralizedTargetEngineer
    from training.model_training import GeneralizedMLTrainer, DataSplitter

    # 1. Load data
    pipeline = MultiStockDataPipeline()
    df = pipeline.load_training_data()

    if df is None:
        print("✗ Could not load training data")
        raise SystemExit(1)

    if not pipeline.validate_training_data(df):
        print("✗ Data validation failed")
        raise SystemExit(1)

    # 2. Features + target
    fe       = GeneralizedFeatureEngineer(df)
    features = fe.create_all_features()
    features_clean = features.dropna()
    df_clean = df.loc[features_clean.index]

    target = GeneralizedTargetEngineer.create_forward_return_target(df_clean, forward_bars=8)

    # 3. Prepare (scales features internally)
    trainer = GeneralizedMLTrainer()
    X, y = trainer.prepare_data(features_clean, target)

    # 4. Split preserving time order
    X_train, X_test, y_train, y_test = DataSplitter.train_test_split_timeseries(X, y)

    # 5. Train
    trainer.train_final_model(X_train, y_train)
    trainer.evaluate(X_test, y_test)

    # 6. Backtest on test window
    df_backtest = df_clean.loc[X_test.index]

    backtester = CostAwareBacktester(
        df=df_backtest,
        features_df=X_test,        # already scaled by trainer.prepare_data
        model=trainer.model,
        scaler=trainer.scaler,
        commission=0.001,
        slippage=0.002,
        position_size=10,
        min_confidence=0.65,
        hold_bars=15,
    )

    bt_metrics = backtester.run()
    backtester.print_results(bt_metrics)