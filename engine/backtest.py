import pandas as pd
import numpy as np
from datetime import datetime


class CostAwareBacktester:
    """
    Backtest with ATR-based exits, ADX regime filter, trailing stop,
    no overlapping trades, and volatility-based position sizing.

    Exit priority (checked bar-by-bar after entry):
      1. Stop loss hit     (1.5x ATR from entry)
      2. Take profit hit   (2.0x ATR from entry, giving 2:1 R:R minimum)
      3. Trailing stop     (activates after 1x ATR in profit, trails 1x ATR)
      4. Max hold cap      (max_hold_bars hard exit — avoids indefinite holds)

    Entry filters:
      - Model confidence >= min_confidence
      - ADX >= adx_threshold (only trade trending markets)
      - No open position on the same ticker
    """

    def __init__(self, df, features_df, model, scaler,
                 commission=0.001, slippage=0.002,
                 capital=100_000.0, risk_pct=0.03,
                 min_confidence=0.60,
                 atr_stop_mult=1.0, atr_tp_mult=2.5, atr_trail_mult=1.0,
                 max_hold_bars=50, adx_threshold=25.0):
        """
        Args:
            df              : OHLCV DataFrame [Open, High, Low, Close, Volume, Ticker?]
            features_df     : Already-scaled features — must contain 'atr' and 'adx' columns
            model           : Trained LightGBM classifier
            scaler          : Fitted StandardScaler (kept for API compatibility)
            commission      : One-way commission rate (default 0.1%)
            slippage        : One-way slippage estimate (default 0.2%)
            capital         : Starting capital (default 100,000)
            risk_pct        : Fraction of capital risked per trade (default 1%)
            min_confidence  : Minimum model confidence to enter (default 0.65)
            atr_stop_mult   : ATR multiplier for stop loss (default 1.5)
            atr_tp_mult     : ATR multiplier for take profit (default 3.0 → 2:1 R:R)
            atr_trail_mult  : ATR multiplier for trailing stop (default 1.0)
            max_hold_bars   : Hard exit after this many bars regardless (default 50)
            adx_threshold   : Minimum ADX to allow entry (default 25)
        """
        self.df = df.reset_index(drop=False)
        self.features = features_df.reset_index(drop=True)
        self.model = model
        self.scaler = scaler

        self.commission = commission
        self.slippage = slippage
        self.total_cost_per_side = commission + slippage

        self.capital = capital
        self.risk_pct = risk_pct
        self.min_confidence = min_confidence

        self.atr_stop_mult  = atr_stop_mult
        self.atr_tp_mult    = atr_tp_mult
        self.atr_trail_mult = atr_trail_mult
        self.max_hold_bars  = max_hold_bars
        self.adx_threshold  = adx_threshold

        self.trades = []
        self.equity_curve = []

        # Compute ATR and ADX from raw OHLCV df — never from scaled features
        from engine.indicators import Indicators
        df_ind    = Indicators.add_all(self.df)
        self._atr = df_ind['atr'].values
        self._adx = df_ind['adx'].values

    def _position_size(self, entry_price, stop_price):
        """
        Volatility-adjusted position sizing.
        Shares = (capital * risk_pct) / (entry - stop)
        Capped at 20% of capital per trade.
        """
        risk_per_share = abs(entry_price - stop_price)
        if risk_per_share < 1e-6:
            return 1

        shares = (self.capital * self.risk_pct) / risk_per_share
        max_shares = (self.capital * 0.25) / entry_price
        return max(1, int(min(shares, max_shares)))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self):
        """Run backtest and return metrics dict (or None if no trades fired)."""
        print("Running ATR backtest with regime filter...")
        print(f"  Stop: {self.atr_stop_mult}x ATR  |  TP: {self.atr_tp_mult}x ATR  "
              f"|  Trail: {self.atr_trail_mult}x ATR  |  ADX >= {self.adx_threshold}  "
              f"|  ATR/ADX sourced from features")

        predictions    = self.model.predict(self.features)
        probabilities  = self.model.predict_proba(self.features)
        pred_confidence = probabilities.max(axis=1)

        equity           = self.capital
        has_ticker_col   = 'Ticker' in self.df.columns
        open_until       = -1   # bar index until which we're in a trade (no overlap)

        close_arr = self.df['Close'].values
        high_arr  = self.df['High'].values
        low_arr   = self.df['Low'].values
        atr_arr   = self._atr
        adx_arr   = self._adx
        n         = len(self.df)

        for i in range(14, n - 1):

            # ── Skip if still in an open trade ──
            if i <= open_until:
                continue

            pred = predictions[i]
            prob = pred_confidence[i]

            # ── Entry filters ──
            if prob < self.min_confidence:
                continue
            if pred not in (-1, 1):
                continue
            if adx_arr[i] < self.adx_threshold:
                continue

            atr          = atr_arr[i]
            entry_price  = close_arr[i]

            if pred == 1:   # Long
                stop_price   = entry_price - self.atr_stop_mult * atr
                tp_price     = entry_price + self.atr_tp_mult   * atr
                trail_floor  = entry_price + self.atr_trail_mult * atr  # activate trail here
            else:           # Short
                stop_price   = entry_price + self.atr_stop_mult * atr
                tp_price     = entry_price - self.atr_tp_mult   * atr
                trail_floor  = entry_price - self.atr_trail_mult * atr

            shares     = self._position_size(entry_price, stop_price)
            trail_stop = stop_price   # trailing stop starts at initial stop
            exit_price = None
            exit_reason = None

            # ── Bar-by-bar exit simulation ──
            for j in range(i + 1, min(i + self.max_hold_bars + 1, n)):
                bar_high  = high_arr[j]
                bar_low   = low_arr[j]
                bar_close = close_arr[j]

                if pred == 1:   # Long trade
                    # Update trailing stop
                    if bar_close >= trail_floor:
                        new_trail = bar_close - self.atr_trail_mult * atr_arr[j]
                        trail_stop = max(trail_stop, new_trail)

                    # Check stop loss (trail_stop is max of initial stop and trailing)
                    if bar_low <= trail_stop:
                        exit_price  = trail_stop
                        exit_reason = 'TRAIL_STOP' if bar_close >= trail_floor else 'STOP_LOSS'
                        open_until  = j
                        break

                    # Check take profit
                    if bar_high >= tp_price:
                        exit_price  = tp_price
                        exit_reason = 'TAKE_PROFIT'
                        open_until  = j
                        break

                else:           # Short trade
                    # Update trailing stop (moves down)
                    if bar_close <= trail_floor:
                        new_trail = bar_close + self.atr_trail_mult * atr_arr[j]
                        trail_stop = min(trail_stop, new_trail)

                    if bar_high >= trail_stop:
                        exit_price  = trail_stop
                        exit_reason = 'TRAIL_STOP' if bar_close <= trail_floor else 'STOP_LOSS'
                        open_until  = j
                        break

                    if bar_low <= tp_price:
                        exit_price  = tp_price
                        exit_reason = 'TAKE_PROFIT'
                        open_until  = j
                        break

            else:
                # Max hold bar hit — exit at last bar's close
                j           = min(i + self.max_hold_bars, n - 1)
                exit_price  = close_arr[j]
                exit_reason = 'MAX_HOLD'
                open_until  = j

            # ── P&L with costs ──
            if pred == 1:
                entry_with_cost = entry_price * (1 + self.total_cost_per_side)
                exit_with_cost  = exit_price  * (1 - self.total_cost_per_side)
                pnl = (exit_with_cost - entry_with_cost) * shares
            else:
                entry_with_cost = entry_price * (1 - self.total_cost_per_side)
                exit_with_cost  = exit_price  * (1 + self.total_cost_per_side)
                pnl = (entry_with_cost - exit_with_cost) * shares

            pnl_pct = pnl / (entry_price * shares)
            equity += pnl
            self.equity_curve.append(equity)

            trade = {
                'entry_bar'  : i,
                'entry_date' : self.df.index[i],
                'entry_price': entry_price,
                'exit_bar'   : j,
                'exit_date'  : self.df.index[j],
                'exit_price' : exit_price,
                'exit_reason': exit_reason,
                'direction'  : 'LONG' if pred == 1 else 'SHORT',
                'confidence' : prob,
                'atr_at_entry': atr,
                'adx_at_entry': adx_arr[i],
                'stop_loss'  : entry_price - self.atr_stop_mult * atr if pred == 1
                               else entry_price + self.atr_stop_mult * atr,
                'take_profit': tp_price,
                'shares'     : shares,
                'holding_bars': j - i,
                'pnl'        : pnl,
                'pnl_pct'    : pnl_pct,
                'profit'     : 1 if pnl > 0 else 0,
            }
            if has_ticker_col:
                trade['ticker'] = self.df['Ticker'].iloc[i]

            self.trades.append(trade)

        if not self.trades:
            print("✗ No trades fired — try lowering min_confidence or adx_threshold")
            return None

        print(f"✓ Backtest complete: {len(self.trades)} trades executed")
        return self._calculate_metrics()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _calculate_metrics(self):
        trades_df = pd.DataFrame(self.trades)

        total_trades   = len(trades_df)
        winning_trades = trades_df[trades_df['pnl'] > 0]
        losing_trades  = trades_df[trades_df['pnl'] < 0]

        win_rate      = len(winning_trades) / total_trades if total_trades else 0
        total_pnl     = trades_df['pnl'].sum()
        avg_trade     = trades_df['pnl'].mean()
        gross_profit  = winning_trades['pnl'].sum() if len(winning_trades) else 0
        gross_loss    = abs(losing_trades['pnl'].sum()) if len(losing_trades) else 0
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float('inf')

        # Max drawdown
        equity_array = np.array([self.capital] + self.equity_curve)
        running_max  = np.maximum.accumulate(equity_array)
        drawdown     = (equity_array - running_max) / running_max
        max_drawdown = float(np.min(drawdown))

        initial_capital = self.capital
        final_equity    = float(equity_array[-1])
        total_return    = (final_equity - initial_capital) / initial_capital

        trading_days  = (trades_df['exit_date'].max() - trades_df['entry_date'].min())
        years         = max(getattr(trading_days, 'days', 1) / 252, 1e-6)
        annual_return = total_return / years

        pnl_array = trades_df['pnl'].values
        if len(pnl_array) > 1 and pnl_array.std() > 0:
            unit_returns = pnl_array / (initial_capital / len(pnl_array))
            sharpe = (unit_returns.mean() / unit_returns.std()) * np.sqrt(252)
        else:
            sharpe = 0.0

        avg_win  = float(winning_trades['pnl'].mean()) if len(winning_trades) else 0.0
        avg_loss = float(losing_trades['pnl'].mean())  if len(losing_trades)  else 0.0
        expected_value = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)

        # Avg R:R realised
        avg_hold = trades_df['holding_bars'].mean()

        # Exit reason breakdown
        exit_counts = trades_df['exit_reason'].value_counts().to_dict()

        # Per-ticker breakdown
        ticker_stats = {}
        if 'ticker' in trades_df.columns:
            for ticker, grp in trades_df.groupby('ticker'):
                wins = grp[grp['pnl'] > 0]
                ticker_stats[ticker] = {
                    'trades'   : len(grp),
                    'win_rate' : len(wins) / len(grp),
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
            'avg_hold_bars' : avg_hold,
            'exit_breakdown': exit_counts,
            'ticker_stats'  : ticker_stats,
            'trades_df'     : trades_df,
        }

    def print_results(self, metrics):
        if metrics is None:
            print("✗ Backtest failed — no valid trades")
            return

        print("\n" + "="*55)
        print("  BACKTEST RESULTS (ATR EXITS + ADX FILTER)")
        print("="*55)

        print(f"\nTrade Summary:")
        print(f"  Total Trades   : {metrics['total_trades']}")
        print(f"  Winning Trades : {metrics['winning_trades']}")
        print(f"  Losing Trades  : {metrics['losing_trades']}")
        print(f"  Win Rate       : {metrics['win_rate']:.1%}")
        print(f"  Avg Hold Bars  : {metrics['avg_hold_bars']:.1f}")

        print(f"\nExit Breakdown:")
        for reason, count in metrics['exit_breakdown'].items():
            print(f"  {reason:<15}: {count}")

        print(f"\nProfitability:")
        print(f"  Total P&L      : ₹{metrics['total_pnl']:>12,.2f}")
        print(f"  Avg Trade      : ₹{metrics['avg_trade']:>12,.2f}")
        print(f"  Avg Win        : ₹{metrics['avg_win']:>12,.2f}")
        print(f"  Avg Loss       : ₹{metrics['avg_loss']:>12,.2f}")
        pf = metrics['profit_factor']
        print(f"  Profit Factor  : {pf:.2f}x" if pf != float('inf')
              else "  Profit Factor  : ∞")

        print(f"\nReturn Metrics:")
        print(f"  Total Return   : {metrics['total_return']:.1%}")
        print(f"  Annual Return  : {metrics['annual_return']:.1%}")
        print(f"  Sharpe Ratio   : {metrics['sharpe_ratio']:.2f}")
        print(f"  Max Drawdown   : {metrics['max_drawdown']:.1%}")
        print(f"  Final Equity   : ₹{metrics['final_equity']:>12,.2f}")

        print(f"\nExpectation:")
        print(f"  Expected Value : ₹{metrics['expected_value']:>12,.2f}")

        if metrics['ticker_stats']:
            print(f"\nPer-Ticker Breakdown:")
            for ticker, stats in sorted(metrics['ticker_stats'].items()):
                print(f"  {ticker:10s}  trades={stats['trades']:4d}  "
                      f"win={stats['win_rate']:.0%}  "
                      f"P&L=₹{stats['total_pnl']:>10,.2f}")

        print()
        if metrics['expected_value'] > 0:
            print("✓ Strategy is PROFITABLE after transaction costs")
        else:
            print("✗ Strategy is NOT PROFITABLE after transaction costs")
        print("="*55)


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from data.pipeline import MultiStockDataPipeline
    from features.engineering import GeneralizedFeatureEngineer, GeneralizedTargetEngineer
    from training.model_training import GeneralizedMLTrainer, DataSplitter

    pipeline = MultiStockDataPipeline()
    df = pd.read_csv("data/raw_data.csv")

    if not pipeline.validate_training_data(df):
        raise SystemExit(1)

    fe             = GeneralizedFeatureEngineer(df)
    features       = fe.create_all_features()
    features_clean = features.dropna()
    df_clean       = df.loc[features_clean.index]

    target = GeneralizedTargetEngineer.create_forward_return_target(df_clean, forward_bars=8)

    trainer = GeneralizedMLTrainer()
    X, y = trainer.prepare_data(features_clean, target)
    X_train, X_test, y_train, y_test = DataSplitter.train_test_split_timeseries(X, y)
    trainer.train_final_model(X_train, y_train)
    trainer.evaluate(X_test, y_test)

    df_backtest = df_clean.loc[X_test.index]

    backtester = CostAwareBacktester(
        df=df_backtest,
        features_df=X_test,
        model=trainer.model,
        scaler=trainer.scaler,
        commission=0.001,
        slippage=0.002,
        capital=100_000,
        risk_pct=0.01,
        min_confidence=0.65,
        atr_stop_mult=1.5,
        atr_tp_mult=3.0,
        atr_trail_mult=1.0,
        max_hold_bars=50,
        adx_threshold=25.0,
    )

    bt_metrics = backtester.run()
    backtester.print_results(bt_metrics)