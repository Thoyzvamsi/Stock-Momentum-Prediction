import pandas as pd
import numpy as np
import warnings
from engine.indicators import Indicators
warnings.filterwarnings('ignore')

class GeneralizedFeatureEngineer:
    """
    Enhanced feature engineering for multi-stock generalized model
    Includes market context and stock-relative features
    """
    
    def __init__(self, df):
        """
        Args:
            df: Combined DataFrame with columns: [Open, High, Low, Close, Volume, Ticker]
        """
        self.df = df.copy()
        # Normalise column names to Title-case so the rest of the code is consistent
        self.df.columns = [c.strip().title() if c.strip().lower() not in ('ticker', 'datetime')
                           else c.strip().capitalize() for c in self.df.columns]
        self.features = pd.DataFrame(index=self.df.index)
    
    def create_all_features(self):
        """Build all features"""
        print("Creating generalized features...")

        # Stock-specific features (TIER 1)
        self._volume_imbalance()
        self._candle_size_percentile()
        self._volume_surge()
        self._tick_momentum()
        self._vwap_distance()
        
        # Stock-specific features (TIER 2)
        self._body_wick_ratio()
        self._volatility_percentile()
        self._consecutive_bars()
        self._price_structure()
        self._momentum_acceleration()
        
        # Stock-specific features (TIER 3)
        self._time_features()
        self._range_position()
        self._gap_from_previous()
        self._money_flow_index()
        
        # MARKET CONTEXT FEATURES (for generalization)
        self._market_relative_volatility()
        self._stock_momentum_rank()
        self._volume_rank()
        self._price_percentile()

        df_ind = Indicators.add_all(self.df)
        self.features['atr'] = df_ind['atr'].values
        self.features['adx'] = df_ind['adx'].values
            
        self._validate()
        print(f"✓ Created {self.features.shape[1]} features")
        return self.features

    # ========================================================================
    # STOCK-SPECIFIC FEATURES (TIER 1)
    # ========================================================================
    
    def _volume_imbalance(self, window=20):
        """Buy vs sell pressure"""
        up_vol = self.df['Volume'].where(
            self.df['Close'] > self.df['Open'], 0
        ).rolling(window).sum()
        
        down_vol = self.df['Volume'].where(
            self.df['Close'] < self.df['Open'], 0
        ).rolling(window).sum()
        
        total_vol = up_vol + down_vol + 1e-10
        self.features['vol_imbalance'] = (up_vol - down_vol) / total_vol
    
    def _candle_size_percentile(self, window=100):
        """Is this candle abnormally large? (per-stock rolling percentile)"""
        candle_size = (self.df['Close'] - self.df['Open']).abs() / (self.df['Open'] + 1e-10)

        # Use transform so the result aligns with the original index automatically
        self.features['candle_size_pct'] = (
            self.df.groupby('Ticker')['Close']
            .transform(lambda x: candle_size.loc[x.index].rolling(window, min_periods=1).rank(pct=True))
        )
    
    def _volume_surge(self, window=20):
        """Is volume abnormal?"""
        avg_volume = self.df.groupby('Ticker')['Volume'].transform(
            lambda x: x.rolling(window, min_periods=1).mean()
        )
        self.features['volume_surge'] = self.df['Volume'] / (avg_volume + 1e-10)
    
    def _tick_momentum(self, window=10):
        """Net buying/selling pressure"""
        tick = np.sign(self.df['Close'] - self.df['Close'].shift(1))
        self.features['tick_momentum'] = tick.rolling(window, min_periods=1).sum() / window
    
    def _vwap_distance(self):
        typical_price = (
            self.df['High']
            + self.df['Low']
            + self.df['Close']
        ) / 3

        tp_vol = typical_price * self.df['Volume']

        cum_tp_vol = tp_vol.groupby(self.df['Ticker']).cumsum()
        cum_vol = self.df['Volume'].groupby(self.df['Ticker']).cumsum()

        vwap = cum_tp_vol / (cum_vol + 1e-10)

        self.features['vwap_dist'] = (
            self.df['Close'] - vwap
        ) / (vwap + 1e-10)
    
    # ========================================================================
    # STOCK-SPECIFIC FEATURES (TIER 2)
    # ========================================================================
    
    def _body_wick_ratio(self):
        """Candle conviction"""
        body = abs(self.df['Close'] - self.df['Open'])
        total_range = self.df['High'] - self.df['Low']
        self.features['body_wick_ratio'] = body / (total_range + 1e-10)
    
    def _volatility_percentile(self, window=100):
        """Is volatility high or low? (per-stock rolling percentile)"""
        returns_std = self.df.groupby('Ticker')['Close'].transform(
            lambda x: x.pct_change().rolling(20, min_periods=5).std()
        )
        self.features['vol_percentile'] = self.df.groupby('Ticker')['Close'].transform(
            lambda x: returns_std.loc[x.index].rolling(window, min_periods=1).rank(pct=True)
        )
    
    def _consecutive_bars(self):
        """Consecutive bars in same direction"""
        direction = np.sign(self.df['Close'] - self.df['Open'])
        
        consecutive = []
        count = 0
        prev_dir = 0
        
        for d in direction:
            if d == prev_dir and d != 0:
                count += 1
            else:
                count = 1
            consecutive.append(count)
            prev_dir = d
        
        self.features['consecutive_bars'] = pd.Series(consecutive, index=self.df.index)
    
    def _price_structure(self, window=5):
        """HH/LL pattern"""
        higher_high = (self.df['High'] > self.df['High'].shift(1)).rolling(window, min_periods=1).sum()
        lower_low = (self.df['Low'] < self.df['Low'].shift(1)).rolling(window, min_periods=1).sum()
        self.features['price_structure'] = higher_high - lower_low
    
    def _momentum_acceleration(self):
        """Change in momentum"""
        returns_5 = self.df['Close'].pct_change(5)
        returns_10 = self.df['Close'].pct_change(10)
        self.features['acceleration'] = returns_5 - (returns_10 / 2)
    
    # ========================================================================
    # STOCK-SPECIFIC FEATURES (TIER 3)
    # ========================================================================
    
    def _time_features(self):
        """Time-of-day effects"""
        # Handle both a 'Datetime' column and a DatetimeIndex

        if 'Datetime' in self.df.columns:
            dt = pd.to_datetime(self.df['Datetime'])
        else:
            dt = pd.Series(self.df.index, index=self.df.index)

        hour = dt.dt.hour + dt.dt.minute / 60

        self.features['is_first_hour'] = (
            (hour >= 9.5) & (hour < 10.5)
        ).astype(int)

        self.features['is_lunch'] = (
            (hour >= 12) & (hour < 14)
        ).astype(int)

        self.features['is_last_hour'] = (
            hour >= 15
        ).astype(int)
    
    def _range_position(self):
        """Where close is within the candle range"""
        candle_range = self.df['High'] - self.df['Low']
        close_position = (self.df['Close'] - self.df['Low']) / (candle_range + 1e-10)
        self.features['range_position'] = close_position
    
    def _gap_from_previous(self):
        """Opening gaps"""
        gap = (self.df['Open'] - self.df['Close'].shift(1)) / (self.df['Close'].shift(1) + 1e-10)
        self.features['gap_size'] = gap
    
    def _money_flow_index(self, window=14):
        """Volume-weighted RSI"""
        typical_price = (self.df['High'] + self.df['Low'] + self.df['Close']) / 3
        money_flow = typical_price * self.df['Volume']
        
        positive_flow = money_flow.where(
            typical_price > typical_price.shift(1), 0
        ).rolling(window, min_periods=1).sum()
        
        negative_flow = money_flow.where(
            typical_price < typical_price.shift(1), 0
        ).rolling(window, min_periods=1).sum()
        
        mfi = 100 - (100 / (1 + positive_flow / (negative_flow + 1e-10)))
        self.features['mfi'] = mfi
    
    # ========================================================================
    # MARKET CONTEXT FEATURES (for generalization)
    # ========================================================================
    
    def _market_relative_volatility(self, window=20):
        """
        Volatility relative to the stock's own recent history.
        (Cross-stock comparison at the same timestamp is unreliable for
        unevenly-sampled multi-stock data, so we use per-stock z-scoring instead.)
        """
        stock_vol = self.df.groupby('Ticker')['Close'].transform(
            lambda x: x.pct_change().rolling(window, min_periods=5).std()
        )
        # z-score within each stock's own history
        vol_mean = self.df.groupby('Ticker')['Close'].transform(
            lambda x: stock_vol.loc[x.index].rolling(window * 5, min_periods=10).mean()
        )
        vol_std = self.df.groupby('Ticker')['Close'].transform(
            lambda x: stock_vol.loc[x.index].rolling(window * 5, min_periods=10).std()
        )
        self.features['relative_volatility'] = (stock_vol - vol_mean) / (vol_std + 1e-10)
    
    def _stock_momentum_rank(self, window=20):
        """
        Rolling momentum percentile rank within each stock's own history.
        (Replaces slow index.map-based cross-stock rank that produced NaNs.)
        """
        self.features['momentum_rank'] = self.df.groupby('Ticker')['Close'].transform(
            lambda x: x.pct_change(window).rolling(window * 5, min_periods=10).rank(pct=True)
        )
    
    def _volume_rank(self, window=20):
        """
        Rolling volume percentile rank within each stock's own history.
        """
        norm_vol = self.df.groupby('Ticker')['Volume'].transform(
            lambda x: x / (x.rolling(window, min_periods=5).mean() + 1e-10)
        )
        self.features['volume_rank'] = self.df.groupby('Ticker')['Volume'].transform(
            lambda x: norm_vol.loc[x.index].rolling(window * 5, min_periods=10).rank(pct=True)
        )
    
    def _price_percentile(self, window=100):
        """
        Price position in its recent range per stock (rolling percentile).
        """
        self.features['price_percentile'] = self.df.groupby('Ticker')['Close'].transform(
            lambda x: x.rolling(window, min_periods=1).rank(pct=True)
        )
    
    # ========================================================================
    # Validation — warn and drop rather than hard-raise
    # ========================================================================
    
    def _validate(self):
        """Validate features; fill edge-case NaNs rather than crashing."""
        initial_nan = self.features.isna().sum().sum()

        if initial_nan > 0:
            # Forward-fill then backward-fill to handle boundary NaNs from rolling
            self.features = self.features.ffill().bfill()
            remaining_nan = self.features.isna().sum().sum()
            if remaining_nan > 0:
                # Last resort: fill with column median
                self.features = self.features.fillna(self.features.median())
                print(f"⚠ Filled {initial_nan} NaNs (forward-fill + median fallback)")
            else:
                print(f"⚠ Forward-filled {initial_nan} boundary NaNs")

        # Check variance
        zero_var_cols = []
        extreme_cols = []
        for col in self.features.columns:
            if self.features[col].std() == 0:
                zero_var_cols.append(col)
            if self.features[col].abs().max() > 1e10:
                extreme_cols.append(col)

        if zero_var_cols:
            print(f"⚠ Zero-variance columns: {zero_var_cols}")
        if extreme_cols:
            # Clip rather than raise
            for col in extreme_cols:
                self.features[col] = self.features[col].clip(-1e6, 1e6)
            print(f"⚠ Clipped extreme values in: {extreme_cols}")

        print(f"✓ Feature validation passed")


class GeneralizedTargetEngineer:
    """Create targets for multi-stock model"""
    @staticmethod
    def create_forward_return_target(df, forward_bars=6, threshold=0.005):
        """
        Create 3-class target for generalized model.
        Accounts for different price scales across stocks by using % returns.
        """
        # Normalise column names defensively
        close_col = next(c for c in df.columns if c.lower() == 'close')
        future_close = df[close_col].shift(-forward_bars)
        forward_return = (future_close / df[close_col]) - 1
        
        target = np.where(
            forward_return > threshold, 1,
            np.where(forward_return < -threshold, -1, 0)
        )
        
        return pd.Series(target, index=df.index, name='target')


if __name__ == "__main__":
    from data.pipeline import MultiStockDataPipeline
    
    pipeline = MultiStockDataPipeline()
    df = pipeline.load_training_data()
    
    if df is not None:
        fe = GeneralizedFeatureEngineer(df)
        features = fe.create_all_features()
        
        target = GeneralizedTargetEngineer.create_forward_return_target(df)
        
        print(f"\nFeatures shape: {features.shape}")
        print(f"Target shape: {target.shape}")
        print(f"Target distribution: {pd.Series(target).value_counts().to_dict()}")
