
import pandas as pd
import numpy as np
 
 
class Indicators:
    """
    Standalone technical indicator calculator.
    All methods are static — pass in a standard OHLCV DataFrame,
    get back the same DataFrame with indicator columns appended.
 
    Expected columns (case-insensitive): Open, High, Low, Close, Volume
    """
 
    @staticmethod
    def atr(df: pd.DataFrame, period: int = 14, col: str = 'atr') -> pd.DataFrame:
        """
        Average True Range.
        Measures volatility — used for stop loss and position sizing.
 
        Args:
            df     : OHLCV DataFrame
            period : Lookback period (default 14)
            col    : Output column name (default 'atr')
 
        Returns:
            df with '{col}' column appended
        """
        df = df.copy()
        high  = df['High']
        low   = df['Low']
        close = df['Close']
 
        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low  - close.shift(1)).abs()
        ], axis=1).max(axis=1)
 
        df[col] = tr.rolling(period, min_periods=1).mean()
        return df
 
    @staticmethod
    def adx(df: pd.DataFrame, period: int = 14,
            col_adx: str = 'adx',
            col_di_plus: str = 'di_plus',
            col_di_minus: str = 'di_minus') -> pd.DataFrame:
        """
        Average Directional Index + DI+ / DI-.
        ADX measures trend strength (not direction):
          < 20  : No trend / ranging
          20-25 : Weak trend
          25-40 : Strong trend  ← good for momentum entries
          > 40  : Very strong trend
 
        DI+ > DI- : Bullish trend
        DI- > DI+ : Bearish trend
 
        Args:
            df          : OHLCV DataFrame
            period      : Lookback period (default 14)
            col_adx     : Output column name for ADX (default 'adx')
            col_di_plus : Output column name for DI+ (default 'di_plus')
            col_di_minus: Output column name for DI- (default 'di_minus')
 
        Returns:
            df with 'adx', 'di_plus', 'di_minus' columns appended
        """
        df = df.copy()
        high  = df['High']
        low   = df['Low']
        close = df['Close']
 
        # True Range
        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low  - close.shift(1)).abs()
        ], axis=1).max(axis=1)
 
        # Directional Movement
        dm_plus  = high - high.shift(1)
        dm_minus = low.shift(1) - low
 
        # Keep only positive and dominant DM
        dm_plus  = dm_plus.where((dm_plus > dm_minus) & (dm_plus > 0), 0.0)
        dm_minus = dm_minus.where((dm_minus > dm_plus) & (dm_minus > 0), 0.0)
 
        # Smoothed values
        atr_s     = tr.rolling(period, min_periods=1).mean()
        di_plus   = 100 * dm_plus.rolling(period, min_periods=1).mean()  / (atr_s + 1e-10)
        di_minus  = 100 * dm_minus.rolling(period, min_periods=1).mean() / (atr_s + 1e-10)
 
        dx = 100 * (di_plus - di_minus).abs() / (di_plus + di_minus + 1e-10)
 
        df[col_adx]      = dx.rolling(period, min_periods=1).mean()
        df[col_di_plus]  = di_plus
        df[col_di_minus] = di_minus
 
        return df
 
    @staticmethod
    def add_all(df: pd.DataFrame,
                atr_period: int = 14,
                adx_period: int = 14) -> pd.DataFrame:
        """
        Convenience method — appends ATR, ADX, DI+, DI- in one call.
 
        Returns:
            df with columns: atr, adx, di_plus, di_minus
        """
        df = Indicators.atr(df, period=atr_period)
        df = Indicators.adx(df, period=adx_period)
        return df
 
    @staticmethod
    def adx_signal(adx_value: float,
                   di_plus: float,
                   di_minus: float,
                   threshold: float = 25.0) -> dict:
        """
        Interpret ADX value into a human-readable signal.
        Used by the dashboard to display ADX status.
 
        Returns:
            dict with keys: trending, strength, bias, color
        """
        trending = adx_value >= threshold
 
        if adx_value >= 40:
            strength = "Very Strong"
            color    = "#00e676"
        elif adx_value >= 25:
            strength = "Strong"
            color    = "#69f0ae"
        elif adx_value >= 20:
            strength = "Weak"
            color    = "#ffb74d"
        else:
            strength = "No Trend"
            color    = "#ff5252"
 
        bias = "Bullish" if di_plus > di_minus else "Bearish"
 
        return {
            'trending' : trending,
            'strength' : strength,
            'bias'     : bias,
            'color'    : color,
            'adx'      : round(adx_value, 1),
            'di_plus'  : round(di_plus, 1),
            'di_minus' : round(di_minus, 1),
        }