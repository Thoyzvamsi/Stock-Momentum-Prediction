# 🌍 Generalized Multi-Stock Prediction System

## Overview

A complete machine learning system that trains a **single generalized model** on multiple stocks combined, then uses it to make predictions on **any stock symbol**.

```
Training Phase:
┌─────────────────┐
│ AAPL.csv        │
│ MSFT.csv        │  ──→ Combine ──→ Create Features ──→ Train Model
│ NVDA.csv        │
└─────────────────┘

Prediction Phase:
┌─────────────────┐
│ Any Stock       │ ──→ Fetch ──→ Feature Engineering ──→ Predict
│ (TSLA, AMD)     │
└─────────────────┘
```

---

## Architecture

### Files Overview

| File | Purpose |
|------|---------|
| **multi_stock_data_pipeline.py** | Load multiple CSVs + fetch live data |
| **generalized_feature_engineer.py** | 15 features + market context |
| **generalized_ml_trainer.py** | Train on combined data |
| **stock_predictor.py** | Predict single stock |
| **multi_stock_dashboard.py** | Streamlit interactive dashboard |
| **multi_stock_main.py** | Training pipeline script |

---

## Quick Start

### Step 1: Prepare Training Data

Create CSV files in `data/` folder:

```
data/
├── AAPL.csv      (15-min OHLCV)
├── MSFT.csv
├── NVDA.csv
└── ...
```

**Format:**
```csv
Date,Open,High,Low,Close,Volume
2024-01-01 09:30:00,150.50,151.00,150.00,150.80,1000000
```

### Step 2: Train

```bash
# Install dependencies
pip install -r requirements.txt

# Train generalized model
python multi_stock_main.py
```

**Output:**
- Model: `models/generalized_momentum_model.pkl`
- Scaler: `models/generalized_momentum_scaler.pkl`
- Config: `models/generalized_momentum_config.json`

### Step 3: Predict

```bash
# Launch interactive dashboard
streamlit run multi_stock_dashboard.py
```

Select **🔮 Prediction** mode, enter any stock symbol, get instant predictions.

---

## System Design

### Training Phase

1. **Load Multiple Stocks** (e.g., AAPL, MSFT, NVDA)
2. **Combine Into One Dataset** (~100K+ bars from different stocks)
3. **Engineer Features** (15 features + market context)
4. **Create Targets** (8-bar forward return classification)
5. **Train One Model** (learns patterns across all stocks)
6. **Walk-Forward Validate** (5-fold time-series validation)
7. **Save Artifacts** (model, scaler, config)

### Prediction Phase

1. **User Enters Stock Symbol** (e.g., TSLA)
2. **Fetch Live Data** (3-month recent data via yfinance)
3. **Apply Same Features** (exact same engineering as training)
4. **Use Same Scaler** (normalization)
5. **Load Trained Model**
6. **Generate Predictions** (signal + probability + confidence)

---

## Features Engineering

### Stock-Specific (15 features)

```python
1. Volume Imbalance      - Buy vs sell pressure
2. Candle Size %ile      - Relative candle size
3. Volume Surge          - Abnormal volume
4. Tick Momentum         - Direction momentum
5. VWAP Distance         - Institutional anchor
6. Body/Wick Ratio       - Candle conviction
7. Volatility %ile       - Vol regime
8. Consecutive Bars      - Trend exhaustion
9. Price Structure       - HH/LL pattern
10. Acceleration         - Momentum change
11. Time Features        - Hour of day effects
12. Range Position       - Close in range
13. Gap Size             - Opening gaps
14. Money Flow Index     - Volume-weighted RSI
15. *Reserved for expansion*
```

### Market Context (4 features)

```python
1. Relative Volatility   - Stock vol vs market
2. Momentum Rank         - Stock rank among all
3. Volume Rank          - Volume rank among all
4. Price Percentile     - Price position in range
```

**Why Market Context?**
- Helps model generalize across different volatility regimes
- Normalizes for different price scales
- Adapts to market conditions

---

## Model Architecture

### LightGBM Classifier
```python
lgb.LGBMClassifier(
    n_estimators=500,
    max_depth=5,
    learning_rate=0.01,
    class_weight='balanced'  # Important for imbalanced classes
)
```

### Outputs
- **Prediction**: -1 (SELL), 0 (HOLD), 1 (BUY)
- **Probability**: [P(-1), P(0), P(1)]
- **Confidence**: Max probability
- **Expected Return**: Estimated profit % based on signal

---

## Dashboard Modes

### 🔧 Training Mode
- Select training stocks
- Configure parameters
- Train model in real-time
- View performance metrics
- Save model

### 🔮 Prediction Mode
- Enter any stock symbol
- Get instant prediction
- See confidence score
- View timeline of predictions
- Generate trading signals

### 📊 Dashboard Mode
- Model validation metrics
- Feature importance ranking
- Training statistics
- Performance summary

---

## Validation Approach

### Walk-Forward Validation (5-Fold)

```
Fold 1: Train [0-80%]    → Test [80-84%]   ✓
Fold 2: Train [0-84%]    → Test [84-88%]   ✓
Fold 3: Train [0-88%]    → Test [88-92%]   ✓
Fold 4: Train [0-92%]    → Test [92-96%]   ✓
Fold 5: Train [0-96%]    → Test [96-100%]  ✓
```

**Why Important:**
- ✅ Prevents lookahead bias
- ✅ Shows realistic performance
- ✅ Detects overfitting
- ✅ Validates time-series separation

---

## Expected Performance

### Training Metrics
- **Accuracy**: 55-65% (better than 50% baseline)
- **Precision**: 55-65% (precision of positive predictions)
- **Recall**: 55-65% (recall of positives)
- **F1-Score**: 55-65% (balanced metric)

### Generalization
- ✅ Works on stocks NOT in training set
- ✅ Adapts to different price scales
- ✅ Handles different volatility regimes
- ✅ Works across market conditions

---

## Usage Examples

### Training

```bash
# Basic training
python multi_stock_main.py

# Custom parameters
python multi_stock_main.py --forward-bars 10 --threshold 0.01 --test-size 0.25
```

### Dashboard

```bash
# Launch
streamlit run multi_stock_dashboard.py

# Access
# Browser opens at http://localhost:8501
```

### Python API

```python
from stock_predictor import StockPredictor, SignalGenerator
from multi_stock_data_pipeline import MultiStockDataPipeline

# Load predictor
predictor = StockPredictor('generalized_momentum')

# Fetch data
pipeline = MultiStockDataPipeline()
df = pipeline.fetch_live_data('TSLA', period='3mo')

# Predict
results = predictor.predict_stock(df)
latest = predictor.predict_latest(df)

# Generate signal
signal = SignalGenerator.generate_signal(
    latest['prediction'],
    latest['confidence'],
    min_confidence=0.65
)

print(f"Signal: {signal['signal']}")
print(f"Confidence: {latest['confidence']:.1%}")
```

---

## Training Data Requirements

### Minimum
- **3 stocks** with 3+ months each = ~30K bars
- Format: 15-minute OHLCV data
- No missing data in OHLCV columns

### Recommended
- **5-10 stocks** with 6-12 months each = ~100K+ bars
- Mix of volatility (stable + volatile)
- Different sectors (improves generalization)

### Example Mix
```
AAPL.csv     - Mega-cap stable
MSFT.csv     - Mega-cap stable
NVDA.csv     - Mega-cap volatile
AMD.csv      - Large-cap volatile
GME.csv      - Mid-cap volatile
```

---

## How It Generalizes

### The Challenge
```
❌ Single-stock model trained on AAPL fails on MSFT
   - Different volatility profiles
   - Different volume scales
   - Different trading hours participation
```

### Our Solution
```
✅ Generalized model trained on multiple stocks works on any

Features normalize across stocks:
- Volume Imbalance (ratio, not absolute)
- Relative Volatility (vs market average)
- Momentum Rank (percentile position)
- Price Percentile (normalized range)

Result: Model sees patterns that generalize!
```

---

## Advanced Configuration

### Adjust Feature Weighting

Edit `generalized_feature_engineer.py`:
```python
def _stock_momentum_rank(self, window=20):
    # Increase window for more stable rank
    window = 50  # Instead of 20
```

### Change Model Hyperparameters

Edit `generalized_ml_trainer.py`:
```python
self.model = lgb.LGBMClassifier(
    n_estimators=800,      # More trees
    max_depth=6,           # More depth
    learning_rate=0.005,   # Slower learning
)
```

### Adjust Prediction Confidence Threshold

In dashboard or code:
```python
min_confidence = 0.75  # Higher = fewer but stronger signals
```

---

## Troubleshooting

### ❌ "No CSV files found"
**Solution:** Add CSV files to `data/` folder

### ❌ "Model accuracy <55%"
**Solution:** 
- Add more stocks to training data
- Use longer training period (6-12 months)
- Check data quality

### ❌ "No overlapping data for prediction"
**Solution:** 
- Check live data is being fetched
- Ensure yfinance can access the stock

### ❌ "Predictions very close to 50% confidence"
**Solution:**
- Add more feature engineering
- Use different forward_bars value
- Include additional stocks

---

## Performance Benchmarks

### Training (15-min data, 5 stocks, 6 months each)
- Feature engineering: ~5 seconds
- Walk-forward validation: ~30 seconds
- Full training: ~2 minutes

### Prediction
- Fetch live data: ~2 seconds
- Feature engineering: ~0.5 seconds
- Prediction generation: ~0.1 seconds
- Total: ~3 seconds per stock

---

## Next Steps

1. **Prepare Data** - Get CSVs for 5-10 stocks
2. **Train Model** - Run `multi_stock_main.py`
3. **Test Dashboard** - Launch and experiment
4. **Validate** - Check performance metrics
5. **Deploy** - Use in production or paper trading

---

## Best Practices

✅ **DO:**
- Train on multiple stocks
- Use 6-12 months of data minimum
- Validate walk-forward results
- Test on stocks NOT in training set
- Monitor live performance
- Retrain periodically (weekly/monthly)

❌ **DON'T:**
- Train on single stock (defeats generalization)
- Use data <3 months (overfitting risk)
- Skip walk-forward validation
- Test on training stocks only
- Trust backtesting without live data
- Deploy without paper trading

---

Good luck! 🚀
