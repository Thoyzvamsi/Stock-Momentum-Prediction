import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import yfinance as yf
from datetime import datetime
import os

from data.pipeline import MultiStockDataPipeline

# Load .env if present (for local dev)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # pip install python-dotenv if needed
from predictor.stock_predictor import StockPredictor

# ============================================================================
# Page Config
# ============================================================================
st.set_page_config(
    page_title="Stock Momentum Predictor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# Helpers
# ============================================================================

@st.cache_resource
def load_predictor():
    return StockPredictor('generalized_momentum')

@st.cache_data(ttl=300)
def fetch_data(ticker, period):
    pipeline = MultiStockDataPipeline()
    return pipeline.fetch_live_data(ticker, period=period)

@st.cache_data(ttl=3600)
def fetch_fundamentals(ticker):
    """Fetch fundamental data via yfinance and compute key ratios."""
    try:
        info = yf.Ticker(ticker).info
        return info
    except Exception:
        return {}

def compute_fundamental_score(info):
    """
    Score a stock 0-100 for investment quality using forward-looking ratios.
    Returns (score, breakdown_dict).
    """
    score = 0
    breakdown = {}

    # 1. Forward P/E (lower is better, <15 ideal, <25 acceptable)
    fpe = info.get('forwardPE')
    if fpe and fpe > 0:
        if fpe < 15:   pts = 20
        elif fpe < 20: pts = 15
        elif fpe < 25: pts = 10
        elif fpe < 35: pts = 5
        else:           pts = 0
        score += pts
        breakdown['Forward P/E'] = (round(fpe, 1), pts, 20)
    else:
        breakdown['Forward P/E'] = ('N/A', 0, 20)

    # 2. PEG Ratio (price/earnings-to-growth; <1 = undervalued)
    peg = info.get('pegRatio')
    if peg and peg > 0:
        if peg < 1:    pts = 20
        elif peg < 1.5: pts = 15
        elif peg < 2:  pts = 8
        else:           pts = 0
        score += pts
        breakdown['PEG Ratio'] = (round(peg, 2), pts, 20)
    else:
        breakdown['PEG Ratio'] = ('N/A', 0, 20)

    # 3. Return on Equity (higher is better; >15% good)
    roe = info.get('returnOnEquity')
    if roe is not None:
        roe_pct = roe * 100
        if roe_pct > 20:   pts = 20
        elif roe_pct > 15: pts = 15
        elif roe_pct > 10: pts = 8
        elif roe_pct > 0:  pts = 3
        else:               pts = 0
        score += pts
        breakdown['Return on Equity'] = (f"{roe_pct:.1f}%", pts, 20)
    else:
        breakdown['Return on Equity'] = ('N/A', 0, 20)

    # 4. Debt-to-Equity (lower is better; <0.5 ideal)
    de = info.get('debtToEquity')
    if de is not None:
        if de < 30:    pts = 20
        elif de < 60:  pts = 15
        elif de < 100: pts = 8
        elif de < 150: pts = 3
        else:           pts = 0
        score += pts
        breakdown['Debt / Equity'] = (f"{de:.0f}%", pts, 20)
    else:
        breakdown['Debt / Equity'] = ('N/A', 0, 20)

    # 5. Revenue Growth (YoY; >10% strong)
    rev_growth = info.get('revenueGrowth')
    if rev_growth is not None:
        rg = rev_growth * 100
        if rg > 20:   pts = 20
        elif rg > 10: pts = 15
        elif rg > 5:  pts = 8
        elif rg > 0:  pts = 3
        else:          pts = 0
        score += pts
        breakdown['Revenue Growth (YoY)'] = (f"{rg:.1f}%", pts, 20)
    else:
        breakdown['Revenue Growth (YoY)'] = ('N/A', 0, 20)

    return score, breakdown

def score_to_rating(score):
    if score >= 80: return "⭐⭐⭐⭐⭐ Strong Buy"
    if score >= 65: return "⭐⭐⭐⭐ Buy"
    if score >= 50: return "⭐⭐⭐ Hold"
    if score >= 35: return "⭐⭐ Weak"
    return "⭐ Avoid"

def score_to_color(score):
    if score >= 65: return "#00e676"
    if score >= 50: return "#ffb74d"
    return "#ff1744"

def filter_market_hours(df, ticker):
    """
    Keep only rows within trading hours, weekdays only.
    yfinance returns NSE (.NS/.BO) data already in IST (Asia/Kolkata).
    NSE session: 09:15 – 15:30 IST.
    NYSE/NASDAQ: 09:30 – 16:00 ET.
    """
    is_nse = ticker.endswith('.NS') or ticker.endswith('.BO')
    idx = df.index

    # Ensure tz-aware index
    if idx.tz is None:
        idx = idx.tz_localize('UTC')

    if is_nse:
        local = idx.tz_convert('Asia/Kolkata')
        start_min, end_min = 9 * 60 + 15, 15 * 60 + 30
    else:
        local = idx.tz_convert('America/New_York')
        start_min, end_min = 9 * 60 + 30, 16 * 60

    minutes = local.hour * 60 + local.minute
    mask = (minutes >= start_min) & (minutes <= end_min) & (local.dayofweek < 5)
    filtered = df[mask]
    return filtered if len(filtered) > 10 else df

# ============================================================================
# Sidebar
# ============================================================================
st.sidebar.title("🎯 Control Panel")

ticker = st.sidebar.text_input("Stock Symbol", value="TCS.NS", placeholder="TCS.NS, RELIANCE.NS...").upper().strip()
period = st.sidebar.selectbox("Data Period", ["30d", "45d", "60d"], index=2)

st.sidebar.divider()
st.sidebar.caption("Model: generalized_momentum")

# ============================================================================
# Load model + data on every ticker/period change (no button needed)
# ============================================================================
try:
    predictor = load_predictor()
except Exception as e:
    st.error(f"Model not loaded: {e}. Train the model first via `python multi_stock_main.py`")
    st.stop()

with st.spinner(f"Loading {ticker}..."):
    df = fetch_data(ticker, period)

if df is None or len(df) == 0:
    st.error(f"Could not fetch data for **{ticker}**. Check the symbol.")
    st.stop()

# Run prediction
try:
    latest = predictor.predict_latest(df, min_confidence=0.65)
    pred_results = predictor.predict_stock(df)
except Exception as e:
    st.error(f"Prediction error: {e}")
    st.stop()

# ============================================================================
# HEADER ROW — title + model meta
# ============================================================================
h1, h2 = st.columns([3, 1])
with h1:
    st.title(f"📊 {ticker}")
with h2:
    model_info = predictor.get_model_info()
    st.metric("Model Accuracy", f"{model_info['test_accuracy']:.1%}")

st.divider()

# ============================================================================
# ROW 1: Signal card (left) + Candlestick chart (right)
# ============================================================================
signal_col, chart_col = st.columns([1, 2])

with signal_col:
    if latest:
        signal   = latest['signal']
        strength = latest['strength']
        conf     = latest['confidence']
        price    = latest['price']
        exp_ret  = latest['expected_return']

        emoji = "🟢" if signal == "BUY" else ("🔴" if signal == "SELL" else "🟡")
        action_map = {
            'BUY':  'ENTER_LONG'  if strength == 'STRONG' else 'CONSIDER_LONG',
            'SELL': 'ENTER_SHORT' if strength == 'STRONG' else 'CONSIDER_SHORT',
            'HOLD': 'NO_ACTION',
        }

        r1c1, r1c2 = st.columns(2)
        r1c1.metric(f"{emoji} Signal", signal)
        r1c2.metric("Strength", strength)

        r2c1, r2c2 = st.columns(2)
        r2c1.metric("Price", f"₹{price:,.2f}")
        r2c2.metric("Confidence", f"{conf:.1%}")

        r3c1, r3c2 = st.columns(2)
        r3c1.metric("Exp. Return", f"{exp_ret:.2%}")
        r3c2.metric("Action", "")
        st.markdown(
            f"<div style='font-size:13px;color:#888;margin-top:-18px;'>Action</div>"
            f"<div style='font-size:15px;font-weight:600;margin-top:2px;'>{action_map.get(signal,'NO_ACTION')}</div>",
            unsafe_allow_html=True
        )
    else:
        st.warning("Prediction unavailable")

with chart_col:
    chart_df = filter_market_hours(df, ticker).tail(120)  # ~2 trading days of 15m bars
    is_nse = ticker.endswith('.NS') or ticker.endswith('.BO')

    if is_nse:
        # NSE market hours 09:15-15:30 IST; remove gap 15:30->09:15
        hour_break = dict(bounds=[15.5, 9.25], pattern="hour")
    else:
        # NYSE 09:30-16:00 ET; remove gap 16:00->09:30
        hour_break = dict(bounds=[16.0, 9.5], pattern="hour")

    fig_c = go.Figure()
    fig_c.add_trace(go.Candlestick(
        x=chart_df.index,
        open=chart_df['Open'],
        high=chart_df['High'],
        low=chart_df['Low'],
        close=chart_df['Close'],
        increasing_line_color='#00e676',
        decreasing_line_color='#ff1744',
        increasing_fillcolor='#00e676',
        decreasing_fillcolor='#ff1744',
        name='OHLC',
        showlegend=True,
        line=dict(width=1),
    ))
    fig_c.add_trace(go.Bar(
        x=chart_df.index,
        y=chart_df['Volume'],
        name='Volume',
        yaxis='y2',
        marker_color='rgba(100,180,255,0.18)',
        showlegend=True,
    ))
    fig_c.update_layout(
        title=f"{ticker} — 15m Candles (Market Hours)",
        yaxis=dict(title='Price \u20b9', side='left', showgrid=True, gridcolor='#1e2329'),
        yaxis2=dict(
            title='Volume', overlaying='y', side='right',
            showgrid=False,
            range=[0, float(chart_df['Volume'].max()) * 5]
        ),
        xaxis=dict(
            type='date',
            rangebreaks=[
                dict(bounds=["sat", "mon"]),
                hour_break,
            ],
            tickformat='%d %b\n%H:%M',
            showgrid=True,
            gridcolor='#1e2329',
        ),
        xaxis_rangeslider_visible=False,
        height=360,
        template='plotly_dark',
        plot_bgcolor='#0e1117',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0, r=10, t=40, b=0),
        legend=dict(orientation='h', y=1.08),
    )
    st.plotly_chart(fig_c, use_container_width=True)


st.divider()

# ============================================================================
# ROW 2: Fundamental Analysis + Investment Score
# ============================================================================
st.subheader("🔬 Fundamental Analysis & Investment Score")

with st.spinner("Fetching fundamentals..."):
    info = fetch_fundamentals(ticker)

fund_score, breakdown = compute_fundamental_score(info)
rating = score_to_rating(fund_score)
score_color = score_to_color(fund_score)

fa_left, fa_right = st.columns([1, 2])

with fa_left:
    # Investment score gauge-style metric
    st.markdown(f"""
    <div style="
        background: #1e2329;
        border: 1px solid {score_color};
        border-radius: 12px;
        padding: 24px;
        text-align: center;
    ">
        <div style="font-size: 13px; color: #888; margin-bottom: 4px;">INVESTMENT SCORE</div>
        <div style="font-size: 52px; font-weight: 700; color: {score_color};">{fund_score}</div>
        <div style="font-size: 13px; color: #aaa;">/ 100</div>
        <div style="font-size: 16px; margin-top: 12px; color: {score_color};">{rating}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("")

    # Quick stats
    mcap = info.get('marketCap')
    sector = info.get('sector', 'N/A')
    industry = info.get('industry', 'N/A')
    pe = info.get('trailingPE')
    div_yield = info.get('dividendYield')
    beta = info.get('beta')

    st.markdown(f"**Sector:** {sector}")
    st.markdown(f"**Industry:** {industry}")
    if mcap:
        st.markdown(f"**Market Cap:** ₹{mcap/1e9:.1f}B")
    if pe:
        st.markdown(f"**Trailing P/E:** {pe:.1f}")
    if div_yield:
        st.markdown(f"**Dividend Yield:** {div_yield*100:.2f}%")
    if beta:
        st.markdown(f"**Beta:** {beta:.2f}")

with fa_right:
    # Breakdown table
    rows = []
    for metric, (value, pts, max_pts) in breakdown.items():
        fill = pts / max_pts if max_pts else 0
        bar_color = "#00e676" if fill > 0.7 else ("#ffb74d" if fill > 0.4 else "#ff1744")
        bar_html = f"""
        <div style="background:#2a2d35; border-radius:4px; height:8px; margin-top:4px;">
            <div style="width:{fill*100:.0f}%; background:{bar_color}; height:8px; border-radius:4px;"></div>
        </div>"""
        rows.append((metric, value, f"{pts}/{max_pts}", bar_html))

    st.markdown("""
    <style>
    .fund-table { width:100%; border-collapse: collapse; font-size: 14px; }
    .fund-table th { color: #888; font-weight: 500; padding: 6px 10px; text-align: left; border-bottom: 1px solid #333; }
    .fund-table td { padding: 8px 10px; border-bottom: 1px solid #1e2329; vertical-align: middle; }
    </style>
    <table class="fund-table">
      <tr><th>Metric</th><th>Value</th><th>Score</th><th style="width:35%">Rating</th></tr>
    """ + "".join(
        f"<tr><td>{m}</td><td>{v}</td><td>{s}</td><td>{b}</td></tr>"
        for m, v, s, b in rows
    ) + "</table>", unsafe_allow_html=True)

    # Analyst targets
    st.markdown("")
    target_low  = info.get('targetLowPrice')
    target_mean = info.get('targetMeanPrice')
    target_high = info.get('targetHighPrice')
    rec         = info.get('recommendationKey', 'N/A').upper()
    num_analysts= info.get('numberOfAnalystOpinions')

    if target_mean and latest:
        current = latest['price']
        upside = ((target_mean - current) / current) * 100
        col_a, col_b, col_c, col_d = st.columns(4)
        col_a.metric("Analyst Consensus", rec)
        col_b.metric("Price Target", f"₹{target_mean:.0f}")
        col_c.metric("Upside", f"{upside:.1f}%")
        if num_analysts:
            col_d.metric("# Analysts", num_analysts)

st.divider()

# ============================================================================
# ROW 3: Predictions Timeline
# ============================================================================
st.subheader("📈 Predictions Timeline (Last 50 Bars)")

if pred_results:
    last_n = -50
    pred_df = pd.DataFrame({
        'timestamp': pred_results['timestamps'][last_n:],
        'price':     pred_results['prices'][last_n:],
        'confidence':pred_results['confidence'][last_n:],
    })

    fig_t = go.Figure()
    fig_t.add_trace(go.Scatter(
        x=pred_df['timestamp'], y=pred_df['price'],
        name='Price', yaxis='y',
        line=dict(color='#4fc3f7', width=1.5)
    ))
    fig_t.add_trace(go.Scatter(
        x=pred_df['timestamp'], y=pred_df['confidence'],
        name='Confidence', yaxis='y2',
        line=dict(color='#ffb74d', dash='dash', width=1.5)
    ))
    fig_t.add_hline(
        y=0.65, line_dash='dot', line_color='rgba(255,255,255,0.25)',
        annotation_text='Signal threshold (0.65)', annotation_position='bottom right',
        yref='y2'
    )
    fig_t.update_layout(
        yaxis=dict(title='Price ₹'),
        yaxis2=dict(title='Confidence', overlaying='y', side='right', range=[0.4, 1.0]),
        height=320,
        template='plotly_dark',
        hovermode='x unified',
        margin=dict(l=0, r=0, t=10, b=0),
    )
    st.plotly_chart(fig_t, use_container_width=True)


# ============================================================================
# ROW 3: AI News Sentiment
# ============================================================================
st.divider()
st.subheader("📰 AI News Sentiment Analysis")

from groq import Groq
import httpx
from urllib.parse import quote

@st.cache_data(ttl=900)  # 15-min cache
def fetch_and_analyze_news(ticker_sym):
    """
    Scrape recent headlines via Google News RSS, then use Claude to
    analyze sentiment and produce a market outlook.
    """
    # Strip exchange suffix for cleaner search (TCS.NS -> TCS)
    base = ticker_sym.split('.')[0]
    company_name = yf.Ticker(ticker_sym).info.get('longName', base)

    # Fetch Google News RSS
    rss_url = f"https://news.google.com/rss/search?q={quote(company_name + ' stock')}&hl=en-IN&gl=IN&ceid=IN:en"
    headlines = []
    try:
        resp = httpx.get(rss_url, timeout=10, follow_redirects=True)
        import re
        # Extract <title> tags from RSS (skip first which is feed title)
        titles = re.findall(r'<title><!\[CDATA\[(.*?)\]\]></title>', resp.text)
        if not titles:
            titles = re.findall(r'<title>(.*?)</title>', resp.text)[1:]  # skip feed title
        headlines = titles[:15]
    except Exception as e:
        headlines = [f"Could not fetch news: {e}"]

    if not headlines:
        return None, []

    headlines_text = "\n".join(f"- {h}" for h in headlines)


    # Call Groq (free tier — llama-3.3-70b)
    client = Groq()
    prompt = (
        f"You are a financial analyst. Analyze these recent news headlines for "
        f"{company_name} ({ticker_sym}) and provide:\n\nHeadlines:\n{headlines_text}\n\n"
        "Respond in this EXACT JSON structure (no markdown, no extra text):\n"
        '{"overall_sentiment": "BULLISH or BEARISH or NEUTRAL or MIXED",'
        '"sentiment_score": <integer -100 to +100>,'
        '"key_themes": ["theme1","theme2","theme3"],'
        '"bullish_factors": ["factor1","factor2"],'
        '"bearish_factors": ["factor1","factor2"],'
        '"short_term_outlook": "<2 sentences on next 1-2 weeks>",'
        '"risk_factors": ["risk1","risk2"],'
        '"confidence": "HIGH or MEDIUM or LOW"}'
    )

    try:
        import json
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=800,
            temperature=0.3,
        )
        raw = response.choices[0].message.content.strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        analysis = json.loads(raw)
        return analysis, headlines
    except Exception as e:
        return {"error": str(e)}, headlines


with st.spinner("Fetching news & running AI analysis..."):
    sentiment_data, headlines = fetch_and_analyze_news(ticker)

if sentiment_data and "error" not in sentiment_data:
    sent_left, sent_right = st.columns([1, 2])

    with sent_left:
        overall = sentiment_data.get("overall_sentiment", "NEUTRAL")
        score   = sentiment_data.get("sentiment_score", 0)
        conf    = sentiment_data.get("confidence", "MEDIUM")

        sent_color = {
            "BULLISH": "#00e676",
            "BEARISH": "#ff1744",
            "NEUTRAL": "#ffb74d",
            "MIXED":   "#90caf9",
        }.get(overall, "#aaa")

        sent_emoji = {
            "BULLISH": "📈", "BEARISH": "📉",
            "NEUTRAL": "➡️", "MIXED": "↔️"
        }.get(overall, "")

        st.markdown(f"""
        <div style="
            background:#1e2329;
            border:1px solid {sent_color};
            border-radius:12px;
            padding:20px;
            text-align:center;
        ">
            <div style="font-size:13px;color:#888;margin-bottom:4px;">NEWS SENTIMENT</div>
            <div style="font-size:36px;">{sent_emoji}</div>
            <div style="font-size:28px;font-weight:700;color:{sent_color};">{overall}</div>
            <div style="margin:10px 0;background:#2a2d35;border-radius:6px;height:10px;">
                <div style="width:{min(100, max(0, (score+100)//2))}%;background:{sent_color};height:10px;border-radius:6px;"></div>
            </div>
            <div style="font-size:12px;color:#888;">Score: {score:+d} / Confidence: {conf}</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("")
        st.markdown("**🔑 Key Themes**")
        for theme in sentiment_data.get("key_themes", []):
            st.markdown(f"• {theme}")

    with sent_right:
        outlook = sentiment_data.get("short_term_outlook", "")
        if outlook:
            st.info(f"**Short-Term Outlook (1–2 weeks):** {outlook}")

        bull = sentiment_data.get("bullish_factors", [])
        bear = sentiment_data.get("bearish_factors", [])
        risks = sentiment_data.get("risk_factors", [])

        c1, c2 = st.columns(2)
        with c1:
            if bull:
                st.markdown("**🟢 Bullish Factors**")
                for f in bull:
                    st.markdown(f"✓ {f}")
        with c2:
            if bear:
                st.markdown("**🔴 Bearish Factors**")
                for f in bear:
                    st.markdown(f"✗ {f}")

        if risks:
            st.markdown("**⚠️ Risk Factors**")
            for r in risks:
                st.markdown(f"• {r}")

        with st.expander("📋 Raw Headlines Analyzed"):
            for h in headlines:
                st.markdown(f"- {h}")

elif sentiment_data and "error" in sentiment_data:
    st.warning(f"News analysis unavailable: {sentiment_data['error']}")
else:
    st.info("No news data available for this ticker.")

# ============================================================================
# Footer
# ============================================================================
st.divider()
st.caption("For research purposes only. Not financial advice.")

