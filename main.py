import pandas as pd
import matplotlib.pyplot as plt
from backtest.engine import Strategy

data = pd.read_csv("data/raw_data.csv")

market_type = 0
capital = 15000
data = Strategy.strategy_execution(data,market_type,capital)

data.to_csv("Test_data.csv",index=False)