import pandas as pd
import numpy as np
import yfinance as yf
import os
import pickle
from datetime import datetime, timedelta

class MultiStockDataPipeline:
    """
    Multi-stock data pipeline:
    - Prediction: Fetch live data for any stock
    """
    
    def __init__(self, data_dir='data', models_dir='models'):
        self.data_dir = data_dir
        self.models_dir = models_dir
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(models_dir, exist_ok=True)
    
    
    def validate_training_data(self, df):
        """Validate combined training data"""
        if df is None or len(df) == 0:
            return False
        
        # Check columns
        required = ['open', 'high', 'low', 'close', 'volume', 'ticker']
        cols_lower = [col.lower() for col in df.columns]
        
        if not all(req in cols_lower for req in required):
            print(f"✗ Missing columns. Found: {df.columns.tolist()}")
            return False
        
        # Check for NaNs
        if df.isna().sum().sum() > len(df) * 0.01:
            print("✗ Too many NaN values")
            return False
        
        # Check price sanity
        if (df['High'] < df['Low']).any():
            print("✗ Invalid prices (High < Low)")
            return False
        
        print(f"✓ Data validation passed: {len(df)} bars, {df['Ticker'].nunique()} stocks")
        return True
    
    # ========================================================================
    # PREDICTION: Fetch live data for any stock
    # ========================================================================
    
    def fetch_live_data(self, ticker, period='60d', interval='15m'):
        """
        Fetch live data for prediction
        
        Args:
            ticker: Stock symbol
            period: Data period for features context
            interval: Timeframe
        
        Returns:
            DataFrame with OHLCV data
        """
        try:
            print(f"Fetching live {ticker} data ({period})...")
            df = yf.download(
                ticker,
                period=period,
                interval=interval,
                progress=False,
                auto_adjust=True,
                threads=False
            )
            print("Shape:", None if df is None else df.shape)

            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            if df is None or len(df) == 0:
                print(f"✗ No data returned for {ticker}")
                return None
                    
            # Standardize columns
            df.columns = [col.strip().capitalize() for col in df.columns]
            df = df.dropna()
            df = df[~df.index.duplicated(keep='first')]
            
            # Add ticker column
            df['Ticker'] = ticker

            df.index = pd.to_datetime(df.index)

            print("Columns:", df.columns.tolist())
            print("Index type:", type(df.index))
            print("Shape:", df.shape)

            print(f"✓ Fetched {len(df)} live bars for {ticker}")
            return df
            
        except Exception as e:
            print(f"✗ Error fetching {ticker}: {str(e)}")
            return None
    
    # ========================================================================
    # Utilities
    # ========================================================================
    
    def save_combined_data(self, df, filename='combined_training_data.csv'):
        """Save combined training data"""
        filepath = os.path.join(self.data_dir, filename)
        df.to_csv(filepath)
        print(f"✓ Saved combined data to {filepath}")
    
    def load_combined_data(self, filename='combined_training_data.csv'):
        """Load previously saved combined data"""
        filepath = os.path.join(self.data_dir, filename)
        if os.path.exists(filepath):
            df = pd.read_csv(filepath, index_col=0, parse_dates=True)
            print(f"✓ Loaded combined data: {len(df)} bars")
            return df
        else:
            print(f"✗ Combined data file not found: {filepath}")
            return None
    
    def get_training_tickers(self,df):
        """Get list of tickers in data/ folder"""
        tickers = df["Ticker"].unique()
        return tickers


if __name__ == "__main__":
    pipeline = MultiStockDataPipeline()
    
    # Load training data
    df_train = pipeline.load_training_data()
    
    if df_train is not None and pipeline.validate_training_data(df_train):
        pipeline.save_combined_data(df_train)
        print("\nTraining data ready!")
    
    # Fetch live data for prediction
    df_live = pipeline.fetch_live_data('AAPL', period='3mo')
    if df_live is not None:
        print("Live data ready for prediction!")
