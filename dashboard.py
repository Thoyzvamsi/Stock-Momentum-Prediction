import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import os

from data.pipeline import MultiStockDataPipeline
from features.engineering import GeneralizedFeatureEngineer, GeneralizedTargetEngineer
from training.model_training import GeneralizedMLTrainer, DataSplitter
from engine.backtest import CostAwareBacktester
from predictor.stock_predictor import StockPredictor, SignalGenerator

# ============================================================================
# Page Config
# ============================================================================

st.set_page_config(
    page_title="Multi-Stock Momentum Predictor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# SIDEBAR
# ============================================================================

st.sidebar.title("🎯 Control Panel")

mode = st.sidebar.radio(
    "Select Mode:",
    ["🔧 Training", "🔮 Prediction", "📊 Dashboard"]
)

# ============================================================================
# MODE 1: TRAINING
# ============================================================================

if mode == "🔧 Training":
    st.title("🔧 Model Training - Generalized Multi-Stock")
    
    st.markdown("""
    Train a generalized model on multiple stocks combined.
    This model learns market behavior across different stocks.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📁 Training Data")
        pipeline = MultiStockDataPipeline()
        df_train = pd.read_csv("data/raw_data.csv")
        tickers = pipeline.get_training_tickers(df_train)
        
        if len(tickers) == 0:
            st.error("❌ No CSV files found in data/ folder")
            st.info("Please add CSV files (AAPL.csv, MSFT.csv, etc.) to the data/ folder")
        else:
            st.success(f"✓ Found {len(tickers)} stocks: {', '.join(tickers)}")
    
    with col2:
        st.subheader("⚙️ Training Parameters")
        forward_bars = st.slider("Forward Look (bars):", 5, 20, 8)
        threshold = st.slider("Return Threshold (%):", 0.1, 1.0, 0.5) / 100
        test_size = st.slider("Test Size (%):", 10, 30, 20) / 100
    
    if st.button("🚀 Start Training", key="train_button"):
        with st.spinner("Loading training data..."):
            
            if df_train is not None and pipeline.validate_training_data(df_train):
                with st.spinner("Engineering features..."):
                    fe = GeneralizedFeatureEngineer(df_train)
                    features = fe.create_all_features()
                    target = GeneralizedTargetEngineer.create_forward_return_target(
                        df_train, 
                        forward_bars=forward_bars,
                        threshold=threshold
                    )
                
                with st.spinner("Preparing data..."):
                    trainer = GeneralizedMLTrainer()
                    X, y = trainer.prepare_data(features, target)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    with st.spinner("Walk-forward validation..."):
                        wfv_scores = trainer.walk_forward_validation(X, y, n_splits=5)
                        st.success(f"✓ WFV Average: {np.mean(wfv_scores):.3f}")
                
                with col2:
                    with st.spinner("Splitting data..."):
                        X_train, X_test, y_train, y_test = DataSplitter.train_test_split_timeseries(
                            X, y, test_size=test_size
                        )
                
                with st.spinner("Training model..."):
                    trainer.train_final_model(X_train, y_train)
                
                with st.spinner("Evaluating..."):
                    metrics = trainer.evaluate(X_test, y_test)
                
                st.success("✓ Model trained successfully!")
                
                # Display metrics
                st.subheader("📊 Model Performance")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Accuracy", f"{metrics['accuracy']:.1%}")
                with col2:
                    st.metric("Precision", f"{metrics['precision']:.1%}")
                with col3:
                    st.metric("Recall", f"{metrics['recall']:.1%}")
                with col4:
                    st.metric("F1-Score", f"{metrics['f1']:.1%}")
                
                # Feature importance
                st.subheader("🔍 Feature Importance")
                importance_df = trainer.feature_importance(top_n=15)
                
                fig_importance = px.bar(
                    importance_df.head(15),
                    x='importance',
                    y='feature',
                    orientation='h',
                    title="Top 15 Important Features"
                )
                fig_importance.update_layout(height=400, template='plotly_dark')
                st.plotly_chart(fig_importance, use_container_width=True)
                
                # Save model
                if st.button("💾 Save Model"):
                    trainer.save_model('generalized_momentum')
                    st.success("✓ Model saved!")


# ============================================================================
# MODE 2: PREDICTION
# ============================================================================

elif mode == "🔮 Prediction":
    st.title("🔮 Stock Prediction - Any Symbol")
    
    st.markdown("""
    Use the trained generalized model to predict any stock.
    """)
    
    # Load predictor
    try:
        predictor = StockPredictor('generalized_momentum')
        model_info = predictor.get_model_info()
        
        col1, col2 = st.columns(2)
        
        with col1:
            ticker = st.text_input("Stock Symbol:", value="TCS.NS", placeholder="e.g., TCS.NS,HDFCBANK.NS").upper()
            period = st.selectbox("Data Period:", ["30d", "45d", "60d"], index=2)
        
        with col2:
            st.subheader("Model Info")
            st.metric("Test Accuracy", f"{model_info['test_accuracy']:.1%}")
            st.metric("Samples", f"{model_info['n_samples']:,}")
        
        if st.button("🔍 Generate Prediction", key="predict_button"):
            with st.spinner(f"Fetching {ticker} data..."):
                pipeline = MultiStockDataPipeline()
                df = pipeline.fetch_live_data(ticker, period=period)
            
            if df is not None:
                with st.spinner("Generating predictions..."):
                    import traceback
                    try:
                        latest = predictor.predict_latest(df)
                    except Exception:
                        traceback.print_exc()
                
                if latest is not None:
                    # Current prediction
                    st.subheader(f"📊 {ticker} - Current Prediction")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        signal_color = "🟢" if latest['signal'] == 'BUY' else "🔴" if latest['signal'] == 'SELL' else "🟡"
                        st.metric(f"{signal_color} Signal", latest['signal'])
                    
                    with col2:
                        st.metric("Price", f"${latest['price']:.2f}")
                    
                    with col3:
                        st.metric("Confidence", f"{latest['confidence']:.1%}")
                    
                    with col4:
                        st.metric("Expected Return", f"{latest['expected_return']:.2%}")
                    
                    # Trading signal
                    st.divider()
                    signal = SignalGenerator.generate_signal(
                        latest['prediction'],
                        latest['confidence'],
                        min_confidence=0.65
                    )
                    
                    st.subheader("⚡ Trading Signal")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"**Signal:** {signal['signal']}")
                        st.write(f"**Strength:** {signal['strength']}")
                    with col2:
                        st.write(f"**Action:** {signal['action']}")
                    
                    # Predictions timeline
                    st.divider()
                    st.subheader("📈 Predictions Timeline (Last 50 Bars)")
                    
                    results = predictor.predict_stock(df)
                    
                    # Get last 50
                    last_50_idx = -50
                    pred_df = pd.DataFrame({
                        'timestamp': results['timestamps'][last_50_idx:],
                        'price': results['prices'][last_50_idx:],
                        'prediction': results['predictions'][last_50_idx:],
                        'confidence': results['confidence'][last_50_idx:]
                    })
                    
                    # Plot
                    fig = go.Figure()
                    
                    # Price
                    fig.add_trace(go.Scatter(
                        x=pred_df['timestamp'],
                        y=pred_df['price'],
                        name='Price',
                        yaxis='y',
                        line=dict(color='blue')
                    ))
                    
                    # Confidence
                    fig.add_trace(go.Scatter(
                        x=pred_df['timestamp'],
                        y=pred_df['confidence'],
                        name='Confidence',
                        yaxis='y2',
                        line=dict(color='orange', dash='dash')
                    ))
                    
                    fig.update_layout(
                        title=f"{ticker} - Predictions",
                        yaxis=dict(title="Price"),
                        yaxis2=dict(title="Confidence", overlaying="y", side="right"),
                        height=400,
                        template='plotly_dark',
                        hovermode='x unified'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
    
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        st.info("Please train a model first in the Training section")


# ============================================================================
# MODE 3: DASHBOARD
# ============================================================================

elif mode == "📊 Dashboard":
    st.title("📊 Model Dashboard - Complete Analysis")
    
    try:
        predictor = StockPredictor('generalized_momentum')
        model_info = predictor.get_model_info()
        
        # TAB 1: Model Validation
        st.subheader("📋 Model Validation Metrics")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Test Accuracy", f"{model_info['test_accuracy']:.1%}")
        with col2:
            st.metric("Precision", f"{model_info['test_precision']:.1%}")
        with col3:
            st.metric("Recall", f"{model_info['test_recall']:.1%}")
        with col4:
            st.metric("F1-Score", f"{model_info['test_f1']:.1%}")
        
        st.divider()
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("WFV Mean Accuracy", f"{model_info['wfv_mean']:.3f}")
        with col2:
            st.metric("WFV Std Dev", f"{model_info['wfv_std']:.3f}")
        
        st.divider()
        
        # Feature Importance
        st.subheader("🔍 Top Features")
        
        if model_info['top_features']:
            features = model_info['top_features']['feature']
            importances = model_info['top_features']['importance']
            
            feature_df = pd.DataFrame({
                'Feature': features,
                'Importance': importances
            })
            
            fig = px.bar(
                feature_df,
                x='Importance',
                y='Feature',
                orientation='h',
                title="Top 15 Feature Importance"
            )
            fig.update_layout(height=400, template='plotly_dark')
            st.plotly_chart(fig, use_container_width=True)
        
        st.divider()
        
        # Model Statistics
        st.subheader("📊 Model Statistics")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Training Samples", f"{model_info['n_samples']:,}")
        with col2:
            st.metric("Features", model_info['n_features'])
        with col3:
            st.metric("Feature Names Count", len(model_info['feature_names']))
        
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.info("Please train a model first in the Training section")

# ============================================================================
# Footer
# ============================================================================

st.divider()
st.markdown("""
    ---
    **Generalized Multi-Stock Momentum Predictor**
    
    ✅ Trained on multiple stocks combined
    ✅ Can predict any stock symbol
    ✅ Walk-forward validated (no lookahead bias)
    ✅ Transaction costs included in backtesting
    
    *For educational and research purposes only*
""")
