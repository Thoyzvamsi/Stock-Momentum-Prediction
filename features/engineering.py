import pandas as pd
import numpy as np

class Features_Engineering:
    def __init__(self,data):
        self.data = data

    def features(self):
        data_ = []
        data_ = pd.DataFrame(data_)

        data_["date"] = self.data["date"]
        data_["time"] = self.data["time"]
        data_["close"] = self.data["close"]
        data_["returns"] = self.data["close"].pct_change()

        data_["Target"] = self.create_target(self.data)

        data_["Volume_imbalance"] = self.volume_imbalance(self.data)
        data_["volume_surge"] = self.volume_surge(self.data)
        data_["candle_size"] = self.candle_size_percentile(self.data)
        data_["tick_momentum"] = self.tick_momentum(self.data)
        data_["vwap_distance"] = self.vwap_distance(self.data)
        data_["body_wick_ratio"] = self.body_wick_ratio(self.data)
        data_["volatility_percentile"] = self.volatility_percentile(self.data)
        data_["consecutive_same_direction"] = self.consecutive_same_direction(self.data)
        data_["price_structure"] = self.price_structure(self.data)

        data_ = data_.dropna()

        return data_

    def create_target(self,df, forward_bars=8):  # 8 bars = 2 hours on 15min
    
        forward_return = df['close'].shift(-forward_bars) / df['close'] - 1
        
        # 3-class: Strong Buy, Hold, Strong Sell
        target = pd.cut(forward_return, 
                        bins=[-np.inf, -0.005, 0.005, np.inf],
                        labels=[-1, 0, 1])
        return target
    def volume_imbalance(self,df, window=20):
        """Are buyers or sellers in control?"""
        up_volume = df['volume'].where(df['close'] > df['open'], 0).rolling(window).sum()
        down_volume = df['volume'].where(df['close'] < df['open'], 0).rolling(window).sum()
        return (up_volume - down_volume) / (up_volume + down_volume + 1e-10)

    def candle_size_percentile(self,df, window=100):
        """Is this candle abnormally large?"""
        candle_size = abs(df['close'] - df['open']) / df['open']
        return candle_size.rolling(window).rank(pct=True)

    def volume_surge(self,df, window=20):
        """Is volume abnormal?"""
        avg_volume = df['volume'].rolling(window).mean()
        return df['volume'] / avg_volume

    def tick_momentum(self,df, window=10):
        """Net buying/selling pressure"""
        tick = np.sign(df['close'] - df['close'].shift(1))
        return tick.rolling(window).sum()

    def vwap_distance(self,df):
        """How far from institutional anchor?"""
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        vwap = (df['volume'] * typical_price).cumsum() / df['volume'].cumsum()
        return (df['close'] - vwap) / vwap

    def body_wick_ratio(self,df):
        """Is the candle showing conviction or indecision?"""
        body = abs(df['close'] - df['open'])
        total_range = df['high'] - df['low']
        return body / (total_range + 1e-10)

    def volatility_percentile(self,df, window=100):
        """Is volatility high or low?"""
        returns_std = df['close'].pct_change().rolling(20).std()
        return returns_std.rolling(window).rank(pct=True)

    # 9. CONSECUTIVE BARS - Trend exhaustion
    def consecutive_same_direction(self,df):
        """How many bars in same direction?"""
        direction = np.sign(df['close'] - df['open'])
        
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
        
        return pd.Series(consecutive, index=df.index)

    def price_structure(self,df, window=5):
        """Is price making HH/LL?"""
        higher_high = (df['high'] > df['high'].shift(1)).rolling(window).sum()
        lower_low = (df['low'] < df['low'].shift(1)).rolling(window).sum()
        return higher_high - lower_low


