import streamlit as st

st.set_page_config(
    page_title="Stock Momentum Predictor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import yfinance as yf
import json
import re
import time
import httpx
from datetime import datetime
from urllib.parse import quote
from dotenv import load_dotenv
import os

from data.pipeline import MultiStockDataPipeline
from fundamentals.scoring import Fundamentals
from predictor.stock_predictor import StockPredictor
from engine.indicators import Indicators

# ============================================================================
# Helper — safe single-box HTML (no layout CSS, no flex, no grid)
# ============================================================================
def card(content_html: str, border_color: str = "#1e2330", extra: str = "") -> str:
    return (
        f"<div style='background:#12151c;border:1px solid {border_color};"
        f"border-radius:14px;padding:24px;{extra}'>"
        f"{content_html}</div>"
    )

def label(text: str) -> str:
    return (
        f"<div style='font-size:11px;font-weight:600;letter-spacing:0.1em;"
        f"color:#555e7a;text-transform:uppercase;margin-bottom:6px;'>{text}</div>"
    )

def mono(text: str, size: str = "20px", color: str = "#e8eaf6") -> str:
    return f"<div style='font-size:{size};font-weight:700;color:{color};font-family:monospace;line-height:1;'>{text}</div>"

def divider_line() -> str:
    return "<div style='height:1px;background:#1e2330;margin:16px 0;'></div>"

# ============================================================================
# Secrets
# ============================================================================
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    try:
        GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
    except Exception:
        pass
if not GROQ_API_KEY:
    st.warning("⚠️ GROQ_API_KEY not set — news sentiment disabled.")

# ============================================================================
# Cached loaders
# ============================================================================
@st.cache_resource
def load_predictor():
    return StockPredictor('generalized_momentum')


@st.cache_data(ttl=300)
def fetch_data(ticker: str, period: str):
    try:
        pipeline = MultiStockDataPipeline()
        df = pipeline.fetch_live_data(ticker, period=period)
        if df is None or len(df) == 0:
            return None
        return df
    except Exception:
        return None


@st.cache_data(ttl=7200)
def fetch_fundamentals(ticker: str) -> dict:
    """
    Multi-layer fundamentals fetch.
    Layer 1: yf.Ticker.info (full, but rate-limited on cloud IPs)
    Layer 2: fast_info + financials stitched together
    Layer 3: fast_info only (price/mcap at minimum)
    """
    t = yf.Ticker(ticker)

    # Layer 1 — try .info with retries
    for attempt in range(3):
        try:
            info = t.info
            # yfinance sometimes returns a nearly-empty dict with just a message key
            if info and len(info) > 5 and info.get('regularMarketPrice') is not None:
                return info
        except Exception:
            pass
        time.sleep(1.5 * (attempt + 1))

    # Layer 2 — stitch from fast_info + financials + balance_sheet
    result = {}
    try:
        fi = t.fast_info
        result['regularMarketPrice'] = getattr(fi, 'last_price', None)
        result['marketCap']          = getattr(fi, 'market_cap', None)
        result['sector']             = 'N/A'
        result['industry']           = 'N/A'

        # trailing P/E from fast_info
        pe = getattr(fi, 'pe_forward', None) or getattr(fi, 'pe_trailing', None)
        if pe:
            result['trailingPE'] = pe

        # financials — ROE, revenue growth
        try:
            fin = t.financials
            bs  = t.balance_sheet
            if fin is not None and not fin.empty:
                rev_rows = [r for r in fin.index if 'Total Revenue' in str(r)]
                if rev_rows and fin.shape[1] >= 2:
                    rev = fin.loc[rev_rows[0]]
                    if rev.iloc[0] and rev.iloc[1] and rev.iloc[1] != 0:
                        result['revenueGrowth'] = (rev.iloc[0] - rev.iloc[1]) / abs(rev.iloc[1])
            if bs is not None and not bs.empty:
                eq_rows = [r for r in bs.index if 'Stockholders' in str(r) or 'Equity' in str(r)]
                ni_rows = [r for r in fin.index if 'Net Income' in str(r)] if fin is not None else []
                if eq_rows and ni_rows:
                    equity = bs.loc[eq_rows[0]].iloc[0]
                    net_inc = t.financials.loc[ni_rows[0]].iloc[0]
                    if equity and equity != 0:
                        result['returnOnEquity'] = net_inc / equity
        except Exception:
            pass

    except Exception:
        pass

    return result


@st.cache_data(ttl=1800)
def fetch_company_name(ticker_sym: str) -> str:
    # fast_info only — no .info call to avoid extra rate-limit hit
    try:
        name = getattr(yf.Ticker(ticker_sym).fast_info, 'long_name', None)
        if name:
            return name
    except Exception:
        pass
    return ticker_sym.split('.')[0].replace('-', ' ').title()


@st.cache_data(ttl=3600)
def fetch_and_analyze_news(ticker_sym: str):
    if not GROQ_API_KEY:
        return None, []

    company_name = fetch_company_name(ticker_sym)

    rss_url = (
        f"https://news.google.com/rss/search?"
        f"q={quote(company_name + ' stock NSE')}&hl=en-IN&gl=IN&ceid=IN:en"
    )
    headlines = []
    try:
        resp = httpx.get(rss_url, timeout=10, follow_redirects=True)
        resp.raise_for_status()
        titles = re.findall(r'<title><!\[CDATA\[(.*?)\]\]></title>', resp.text)
        if not titles:
            titles = re.findall(r'<title>(.*?)</title>', resp.text)[1:]
        headlines = [h.strip() for h in titles[:15] if h.strip()]
    except Exception as e:
        return {"error": f"News fetch failed: {e}"}, []

    if not headlines:
        return None, []

    headlines_text = "\n".join(f"- {h}" for h in headlines)
    prompt = f"""You are a financial analyst. Analyze these news headlines for {company_name} ({ticker_sym}).

Headlines:
{headlines_text}

Respond ONLY with a valid JSON object. No markdown, no explanation, no extra text.
{{"overall_sentiment":"BULLISH","sentiment_score":45,"key_themes":["theme1","theme2","theme3"],"bullish_factors":["factor1","factor2"],"bearish_factors":["factor1","factor2"],"short_term_outlook":"Two sentence outlook.","risk_factors":["risk1","risk2"],"confidence":"HIGH"}}

Rules: overall_sentiment must be BULLISH/BEARISH/NEUTRAL/MIXED, sentiment_score integer -100 to +100, confidence HIGH/MEDIUM/LOW, no trailing commas."""

    try:
        from groq import Groq
        client = Groq(api_key=GROQ_API_KEY)
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1024,
            temperature=0.1,
        )
        raw = response.choices[0].message.content.strip()
        raw = re.sub(r'```json|```', '', raw).strip()
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        if not match:
            return {"error": "Model returned no JSON"}, headlines
        raw = match.group(0)
        raw = re.sub(r',\s*}', '}', raw)
        raw = re.sub(r',\s*]', ']', raw)
        return json.loads(raw), headlines
    except json.JSONDecodeError as e:
        return {"error": f"JSON parse error: {e}"}, headlines
    except Exception as e:
        return {"error": str(e)}, headlines


# ============================================================================
# Sidebar
# ============================================================================
st.sidebar.title("🎯 Control Panel")
ticker = st.sidebar.text_input(
    "Stock Symbol", value="TCS.NS",
    placeholder="TCS.NS, RELIANCE.NS, INFY.NS"
).upper().strip()
period = st.sidebar.selectbox("Data Period", ["30d", "45d", "60d"], index=2)
st.sidebar.divider()
st.sidebar.caption("Model: generalized_momentum")

if not ticker:
    st.warning("Enter a stock symbol in the sidebar.")
    st.stop()

# ============================================================================
# Load model
# ============================================================================
try:
    predictor = load_predictor()
except Exception as e:
    st.error(f"Model not loaded: {e}. Run `python multi_stock_main.py` first.")
    st.stop()

# ============================================================================
# Fetch & process data
# ============================================================================
with st.spinner(f"Loading {ticker}…"):
    df = fetch_data(ticker, period)

if df is None or len(df) == 0:
    st.error(f"No data for **{ticker}**. Check the symbol and try again.")
    st.stop()

try:
    df = Indicators.add_all(df)
except Exception as e:
    st.error(f"Indicator computation failed: {e}")
    st.stop()

try:
    signal_result = predictor.predict_latest(df, min_confidence=0.65)
except Exception as e:
    st.error(f"Prediction error: {e}")
    st.stop()

try:
    latest_bar = df.iloc[-1]
    adx_sig = Indicators.adx_signal(
        latest_bar['adx'], latest_bar['di_plus'], latest_bar['di_minus']
    )
except Exception as e:
    st.warning(f"ADX signal error: {e}")
    adx_sig = {'adx': 0.0, 'di_plus': 0.0, 'di_minus': 0.0,
                'strength': 'N/A', 'bias': 'Neutral', 'trending': False, 'color': '#555e7a'}

# ============================================================================
# Header
# ============================================================================
company_display = fetch_company_name(ticker)
st.title(f"📊 {company_display}")
st.caption(f"`{ticker}` · {period} · Updated {datetime.now().strftime('%H:%M:%S')}")
st.divider()

# ============================================================================
# ROW 1 — Signal card | Candlestick chart
# ============================================================================
signal_col, chart_col = st.columns([1, 2.2], gap="large")

with signal_col:
    if signal_result:
        signal   = signal_result.get('signal', 'HOLD')
        strength = signal_result.get('strength', 'WEAK')
        conf     = signal_result.get('confidence', 0.0)
        price    = signal_result.get('price', 0.0)
        exp_ret  = signal_result.get('expected_return', 0.0)

        sig_color = {'BUY': '#00e676', 'SELL': '#ff4466', 'HOLD': '#ffb74d'}.get(signal, '#aaa')
        action_map = {
            ('BUY',  'STRONG'): 'ENTER LONG',  ('BUY',  'WEAK'): 'CONSIDER LONG',
            ('SELL', 'STRONG'): 'ENTER SHORT', ('SELL', 'WEAK'): 'CONSIDER SHORT',
            ('HOLD', 'STRONG'): 'HOLD',        ('HOLD', 'WEAK'): 'STAND ASIDE',
        }
        action    = action_map.get((signal, strength), 'NO ACTION')
        ret_color = '#00e676' if exp_ret >= 0 else '#ff4466'

        # Signal + Action (single box)
        st.markdown(card(
            label("Signal") +
            mono(signal, "46px", sig_color) +
            f"<div style='font-size:13px;color:{sig_color};opacity:0.85;margin-top:6px;'>{action}</div>" +
            divider_line(),
            border_color=sig_color + "44"
        ), unsafe_allow_html=True)

        # 4 metric tiles using st.columns — NO html layout
        m1, m2 = st.columns(2)
        m3, m4 = st.columns(2)
        m1.metric("Price",       f"₹{price:,.2f}")
        m2.metric("Confidence",  f"{conf:.0%}")
        m3.metric("Strength",    strength)
        m4.metric("Exp. Return", f"{exp_ret:+.2%}")

    else:
        st.markdown(card(
            label("Signal") +
            "<div style='color:#555e7a;font-size:14px;margin-top:12px;line-height:1.7;'>"
            "Confidence below 65%<br>No actionable signal.</div>"
        ), unsafe_allow_html=True)

with chart_col:
    try:
        pipeline   = MultiStockDataPipeline()
        chart_df   = pipeline.filter_market_hours(df, ticker).tail(120)
        is_nse     = ticker.endswith('.NS') or ticker.endswith('.BO')
        hour_break = (
            dict(bounds=[15.5, 9.25], pattern="hour") if is_nse
            else dict(bounds=[16.0, 9.5],  pattern="hour")
        )
        fig_c = go.Figure()
        fig_c.add_trace(go.Candlestick(
            x=chart_df.index,
            open=chart_df['Open'], high=chart_df['High'],
            low=chart_df['Low'],   close=chart_df['Close'],
            increasing_line_color='#00e676', decreasing_line_color='#ff4466',
            increasing_fillcolor='#00e676',  decreasing_fillcolor='#ff4466',
            name='OHLC', line=dict(width=1),
        ))
        fig_c.add_trace(go.Bar(
            x=chart_df.index, y=chart_df['Volume'],
            name='Volume', yaxis='y2',
            marker_color='rgba(100,160,255,0.15)',
        ))
        fig_c.update_layout(
            title=dict(text=f"{ticker} · 15m Candles", font=dict(size=13, color='#555e7a')),
            yaxis=dict(title='Price ₹', showgrid=True, gridcolor='#1a1d27', tickfont=dict(size=11)),
            yaxis2=dict(overlaying='y', side='right', showgrid=False,
                        range=[0, float(chart_df['Volume'].max()) * 5]),
            xaxis=dict(type='date',
                       rangebreaks=[dict(bounds=["sat", "mon"]), hour_break],
                       tickformat='%d %b\n%H:%M',
                       showgrid=True, gridcolor='#1a1d27', tickfont=dict(size=10)),
            xaxis_rangeslider_visible=False,
            height=400, template='plotly_dark',
            plot_bgcolor='#0c0e14', paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=0, r=10, t=36, b=0),
            legend=dict(orientation='h', y=1.08, font=dict(size=11)),
        )
        st.plotly_chart(fig_c, use_container_width=True)
    except Exception as e:
        st.warning(f"Chart render error: {e}")

st.divider()

# ============================================================================
# ROW 2 — ADX / DMI (all layout via st.columns)
# ============================================================================
st.markdown(
    "<p style='font-size:13px;font-weight:600;letter-spacing:0.1em;color:#555e7a;"
    "text-transform:uppercase;margin-bottom:4px;'>📡 Trend Strength · ADX / DMI</p>",
    unsafe_allow_html=True
)

adx_val        = float(adx_sig.get('adx', 0))
di_plus        = float(adx_sig.get('di_plus', 0))
di_minus       = float(adx_sig.get('di_minus', 0))
trending       = adx_sig.get('trending', False)
bias           = adx_sig.get('bias', 'Neutral')
strength_label = adx_sig.get('strength', 'N/A')

bias_color   = '#00e676' if bias == 'Bullish' else '#ff4466'
adx_color    = '#00e676' if adx_val >= 40 else ('#ffb74d' if adx_val >= 25 else '#555e7a')
adx_bar_pct  = min(100, adx_val / 60 * 100)
di_max       = max(di_plus, di_minus, 1)
dip_pct      = min(100, (di_plus  / di_max) * 100)
dim_pct      = min(100, (di_minus / di_max) * 100)
trend_color  = '#00e676' if trending else '#ff4466'
trend_label  = 'TRENDING' if trending else 'RANGING'
bias_arrow   = '▲' if bias == 'Bullish' else '▼'

left_adx, mid_adx, right_adx = st.columns([1, 1, 2.5], gap="large")

with left_adx:
    # ADX number + bar — single self-contained box, no layout CSS
    st.markdown(card(
        label("ADX") +
        mono(f"{adx_val:.1f}", "56px", adx_color) +
        f"<div style='font-size:12px;color:#555e7a;margin-top:8px;'>{strength_label}</div>"
        f"<div style='margin-top:14px;background:#1e2330;border-radius:4px;height:4px;overflow:hidden;'>"
        f"<div style='width:{adx_bar_pct:.0f}%;background:{adx_color};height:4px;'></div></div>"
        f"<div style='font-size:10px;color:#444;margin-top:4px;'>0 ——— 25 ——— 60+</div>",
        border_color=adx_color + "33"
    ), unsafe_allow_html=True)

with mid_adx:
    # Bias label + DI bars — each DI row is its own simple box, no flex
    st.markdown(card(
        label("Bias") +
        mono(f"{bias_arrow} {bias}", "26px", bias_color),
        border_color="#1e2330"
    ), unsafe_allow_html=True)

    st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)

    # DI+ row
    st.markdown(
        f"<div style='background:#12151c;border:1px solid #1e2330;border-radius:10px;padding:12px 14px;margin-bottom:6px;'>"
        f"<div style='font-size:11px;font-weight:600;color:#00e676;margin-bottom:6px;'>DI+  {di_plus:.1f}</div>"
        f"<div style='background:#1e2330;border-radius:3px;height:6px;overflow:hidden;'>"
        f"<div style='width:{dip_pct:.0f}%;background:#00e676;height:6px;'></div></div>"
        f"</div>",
        unsafe_allow_html=True
    )
    # DI- row
    st.markdown(
        f"<div style='background:#12151c;border:1px solid #1e2330;border-radius:10px;padding:12px 14px;margin-bottom:6px;'>"
        f"<div style='font-size:11px;font-weight:600;color:#ff4466;margin-bottom:6px;'>DI−  {di_minus:.1f}</div>"
        f"<div style='background:#1e2330;border-radius:3px;height:6px;overflow:hidden;'>"
        f"<div style='width:{dim_pct:.0f}%;background:#ff4466;height:6px;'></div></div>"
        f"</div>",
        unsafe_allow_html=True
    )
    # Trend badge
    st.markdown(
        f"<div style='background:{trend_color}11;border:1px solid {trend_color};"
        f"border-radius:20px;padding:5px 14px;text-align:center;margin-top:4px;'>"
        f"<span style='color:{trend_color};font-size:11px;font-weight:600;letter-spacing:0.1em;'>"
        f"{'✦' if trending else '◌'} {trend_label}</span></div>",
        unsafe_allow_html=True
    )

with right_adx:
    adx_plot = df.tail(100).copy()
    fig_adx  = go.Figure()
    fig_adx.add_trace(go.Scatter(
        x=adx_plot.index, y=adx_plot['adx'],
        name='ADX', line=dict(color='#e8eaf6', width=2),
        fill='tozeroy', fillcolor='rgba(232,234,246,0.04)',
    ))
    fig_adx.add_trace(go.Scatter(
        x=adx_plot.index, y=adx_plot['di_plus'],
        name='DI+', line=dict(color='#00e676', width=1.5, dash='dot'),
    ))
    fig_adx.add_trace(go.Scatter(
        x=adx_plot.index, y=adx_plot['di_minus'],
        name='DI−', line=dict(color='#ff4466', width=1.5, dash='dot'),
    ))
    fig_adx.add_trace(go.Scatter(
        x=[adx_plot.index[0], adx_plot.index[-1]], y=[25, 25],
        mode='lines', line=dict(color='rgba(255,183,77,0.4)', width=1, dash='dash'),
        name='Threshold (25)', showlegend=True,
    ))
    fig_adx.update_layout(
        height=240, template='plotly_dark',
        plot_bgcolor='#0c0e14', paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0, r=0, t=10, b=0),
        yaxis=dict(showgrid=True, gridcolor='#1a1d27', range=[0, 65], tickfont=dict(size=10)),
        xaxis=dict(showgrid=False, tickfont=dict(size=10)),
        legend=dict(orientation='h', y=1.18, font=dict(size=11)),
        hovermode='x unified',
    )
    st.plotly_chart(fig_adx, use_container_width=True)

st.divider()

# ============================================================================
# ROW 3 — Fundamental Analysis (layout via st.columns only)
# ============================================================================
st.markdown(
    "<p style='font-size:13px;font-weight:600;letter-spacing:0.1em;color:#555e7a;"
    "text-transform:uppercase;margin-bottom:4px;'>🔬 Fundamental Analysis · Investment Score</p>",
    unsafe_allow_html=True
)

with st.spinner("Fetching fundamentals…"):
    info = fetch_fundamentals(ticker)

# Show warning if data is sparse (cloud rate-limit fallback)
if not info or len([v for v in info.values() if v is not None]) < 3:
    st.warning("⚠️ Yahoo Finance returned limited data (cloud IP rate-limit). Showing available data only.")

fund = Fundamentals()
try:
    fund_score, breakdown = fund.compute_fundamental_score(info)
    rating      = fund.score_to_rating(fund_score)
    score_color = fund.score_to_color(fund_score)
except Exception as e:
    st.warning(f"Fundamental scoring error: {e}")
    fund_score, breakdown, rating, score_color = 0, {}, "N/A", "#555e7a"

fa_left, fa_right = st.columns([1, 2.2], gap="large")

with fa_left:
    # Score box — single contained div, no layout CSS
    st.markdown(card(
        label("Investment Score") +
        mono(str(fund_score), "64px", score_color) +
        f"<div style='font-size:12px;color:#555e7a;margin-top:2px;'>/ 100</div>"
        f"<div style='font-size:15px;font-weight:600;color:{score_color};margin-top:12px;'>{rating}</div>",
        border_color=score_color + "44",
        extra="text-align:center;"
    ), unsafe_allow_html=True)

    st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)

    meta_fields = [
        ("Sector",       info.get('sector', 'N/A')),
        ("Industry",     info.get('industry', 'N/A')),
        ("Market Cap",   f"₹{info['marketCap']/1e9:.1f}B" if info.get('marketCap') else None),
        ("P/E",          f"{info['trailingPE']:.1f}"       if info.get('trailingPE') else None),
        ("Div. Yield",   f"{info['dividendYield']:.2f}%"   if info.get('dividendYield') else None),
        ("Beta",         f"{info['beta']:.2f}"             if info.get('beta') else None),
    ]
    for lbl, val in meta_fields:
        if val:
            # Each row is its own simple div — no flex, no grid
            st.markdown(
                f"<div style='padding:7px 0;border-bottom:1px solid #1e2330;font-size:13px;'>"
                f"<span style='color:#555e7a;'>{lbl}:&nbsp;</span>"
                f"<span style='font-weight:500;color:#e8eaf6;'>{val}</span></div>",
                unsafe_allow_html=True
            )

with fa_right:
    if breakdown:
        table_rows = ""
        for metric, (value, pts, max_pts) in breakdown.items():
            fill      = pts / max_pts if max_pts else 0
            bar_color = "#00e676" if fill > 0.7 else ("#ffb74d" if fill > 0.4 else "#ff4466")
            bar_html  = (
                f"<div style='background:#1e2330;border-radius:3px;height:5px;'>"
                f"<div style='width:{fill*100:.0f}%;background:{bar_color};height:5px;border-radius:3px;'>"
                f"</div></div>"
            )
            table_rows += (
                f"<tr>"
                f"<td style='padding:10px 10px;border-bottom:1px solid #12151c;font-size:12px;color:#aab;'>{metric}</td>"
                f"<td style='padding:10px 10px;border-bottom:1px solid #12151c;font-size:12px;font-family:monospace;'>{value}</td>"
                f"<td style='padding:10px 10px;border-bottom:1px solid #12151c;font-size:12px;color:#777;'>{pts}/{max_pts}</td>"
                f"<td style='padding:10px 10px;border-bottom:1px solid #12151c;width:30%;'>{bar_html}</td>"
                f"</tr>"
            )
        st.markdown(
            f"<table style='width:100%;border-collapse:collapse;'>"
            f"<tr>"
            f"<th style='color:#555e7a;font-size:11px;font-weight:600;letter-spacing:0.08em;text-transform:uppercase;padding:8px 10px;text-align:left;border-bottom:1px solid #1e2330;'>Metric</th>"
            f"<th style='color:#555e7a;font-size:11px;font-weight:600;letter-spacing:0.08em;text-transform:uppercase;padding:8px 10px;text-align:left;border-bottom:1px solid #1e2330;'>Value</th>"
            f"<th style='color:#555e7a;font-size:11px;font-weight:600;letter-spacing:0.08em;text-transform:uppercase;padding:8px 10px;text-align:left;border-bottom:1px solid #1e2330;'>Score</th>"
            f"<th style='color:#555e7a;font-size:11px;font-weight:600;letter-spacing:0.08em;text-transform:uppercase;padding:8px 10px;text-align:left;border-bottom:1px solid #1e2330;width:30%;'>Rating</th>"
            f"</tr>{table_rows}</table>",
            unsafe_allow_html=True
        )

    st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)

    target_mean  = info.get('targetMeanPrice')
    rec          = (info.get('recommendationKey') or 'N/A').replace('_', ' ').upper()
    num_analysts = info.get('numberOfAnalystOpinions')

    if target_mean and signal_result:
        current = signal_result.get('price', 0)
        upside  = ((target_mean - current) / current) * 100 if current else 0
        col_a, col_b, col_c, col_d = st.columns(4)
        col_a.metric("Consensus",    rec)
        col_b.metric("Price Target", f"₹{target_mean:.0f}")
        col_c.metric("Upside",       f"{upside:.1f}%")
        if num_analysts:
            col_d.metric("# Analysts", num_analysts)

st.divider()

# ============================================================================
# ROW 4 — AI News Sentiment
# ============================================================================
st.markdown(
    "<p style='font-size:13px;font-weight:600;letter-spacing:0.1em;color:#555e7a;"
    "text-transform:uppercase;margin-bottom:4px;'>📰 AI News Sentiment</p>",
    unsafe_allow_html=True
)

if not GROQ_API_KEY:
    st.info("Set GROQ_API_KEY to enable AI news analysis.")
else:
    with st.spinner("Fetching news & running analysis…"):
        sentiment_data, headlines = fetch_and_analyze_news(ticker)

    if not sentiment_data:
        st.info("No news data available for this ticker.")
    elif "error" in sentiment_data:
        st.warning(f"News analysis unavailable: {sentiment_data['error']}")
    else:
        sent_left, sent_right = st.columns([1, 2.2], gap="large")

        with sent_left:
            overall = sentiment_data.get("overall_sentiment", "NEUTRAL")
            score   = int(sentiment_data.get("sentiment_score", 0))
            conf    = sentiment_data.get("confidence", "MEDIUM")

            sent_colors = {"BULLISH": "#00e676", "BEARISH": "#ff4466", "NEUTRAL": "#ffb74d", "MIXED": "#90caf9"}
            sent_icons  = {"BULLISH": "▲", "BEARISH": "▼", "NEUTRAL": "→", "MIXED": "↔"}
            sent_color  = sent_colors.get(overall, "#aaa")
            sent_icon   = sent_icons.get(overall, "")
            bar_width   = min(100, max(0, (score + 100) / 2))

            # Sentiment box — single div, safe CSS only
            st.markdown(card(
                label("News Sentiment") +
                mono(f"{sent_icon} {overall}", "32px", sent_color) +
                f"<div style='margin:14px 0 6px;background:#1e2330;border-radius:6px;height:8px;overflow:hidden;'>"
                f"<div style='width:{bar_width:.0f}%;background:{sent_color};height:8px;border-radius:6px;'></div></div>"
                f"<div style='font-size:11px;color:#555e7a;'>Score: <span style='color:{sent_color};font-weight:600;'>{score:+d}</span>"
                f" &nbsp;·&nbsp; Confidence: <span style='font-weight:600;color:#e8eaf6;'>{conf}</span></div>",
                border_color=sent_color + "33",
                extra="text-align:center;"
            ), unsafe_allow_html=True)

            themes = sentiment_data.get("key_themes", [])
            if themes:
                st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)
                st.markdown(
                    "<p style='font-size:11px;font-weight:600;letter-spacing:0.1em;"
                    "color:#555e7a;text-transform:uppercase;margin-bottom:4px;'>Key Themes</p>",
                    unsafe_allow_html=True
                )
                for theme in themes:
                    st.markdown(
                        f"<div style='font-size:13px;padding:5px 0;border-bottom:1px solid #1e2330;color:#e8eaf6;'>· {theme}</div>",
                        unsafe_allow_html=True
                    )

        with sent_right:
            outlook = sentiment_data.get("short_term_outlook", "")
            if outlook:
                st.info(f"**Outlook (1–2 weeks):** {outlook}")

            bull  = sentiment_data.get("bullish_factors", [])
            bear  = sentiment_data.get("bearish_factors", [])
            risks = sentiment_data.get("risk_factors", [])

            c1, c2 = st.columns(2)
            with c1:
                if bull:
                    st.markdown("**🟢 Bullish Factors**")
                    for f in bull:
                        st.markdown(f"<div style='font-size:13px;padding:4px 0;color:#ccc;'>✓ {f}</div>", unsafe_allow_html=True)
            with c2:
                if bear:
                    st.markdown("**🔴 Bearish Factors**")
                    for f in bear:
                        st.markdown(f"<div style='font-size:13px;padding:4px 0;color:#ccc;'>✗ {f}</div>", unsafe_allow_html=True)

            if risks:
                st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)
                st.markdown("**⚠️ Risk Factors**")
                for r in risks:
                    st.markdown(f"<div style='font-size:13px;padding:4px 0;color:#ccc;'>· {r}</div>", unsafe_allow_html=True)

            if headlines:
                with st.expander("📋 Raw Headlines Analyzed"):
                    for h in headlines:
                        st.markdown(
                            f"<div style='font-size:12px;padding:4px 0;color:#888;border-bottom:1px solid #1e2330;'>— {h}</div>",
                            unsafe_allow_html=True
                        )

# ============================================================================
# Footer
# ============================================================================
st.divider()
st.caption("For research purposes only. Not financial advice.")