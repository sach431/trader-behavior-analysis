import streamlit as st
import pandas as pd
import plotly.express as px
import os
from src.modeling import train_model

# =====================
# Page Config
# =====================
st.set_page_config(
    page_title="Trader Behavior vs Market Sentiment",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Trader Performance vs Market Sentiment")

# =====================
# Load & Merge Data
# =====================
@st.cache_data
def load_data():
    base_path = os.path.dirname(__file__)
    data_path = os.path.join(base_path, "data")

    trades_path = os.path.join(data_path, "historical_data.csv")
    sentiment_path = os.path.join(data_path, "fear_greed_index.csv")

    if not os.path.exists(trades_path) or not os.path.exists(sentiment_path):
        st.error("❌ Data files not found. Make sure they are inside the 'data' folder.")
        st.stop()

    trades = pd.read_csv(trades_path)
    sentiment = pd.read_csv(sentiment_path)

    trades.columns = trades.columns.str.strip().str.lower()
    sentiment.columns = sentiment.columns.str.strip().str.lower()

    # Numeric conversion
    if "closed pnl" in trades.columns:
        trades["closed pnl"] = pd.to_numeric(trades["closed pnl"], errors="coerce")

    # Date handling
    if "timestamp ist" in trades.columns:
        trades["timestamp ist"] = pd.to_datetime(trades["timestamp ist"], errors="coerce")
        trades["date"] = trades["timestamp ist"].dt.date

    if "date" in sentiment.columns:
        sentiment["date"] = pd.to_datetime(sentiment["date"], errors="coerce").dt.date

    df = trades.merge(sentiment, on="date", how="left")

    # Create volume_change feature
    if "value" in df.columns:
        df["volume_change"] = df["value"].pct_change()

    return df


df = load_data()

# =====================
# Sidebar Filters
# =====================
st.sidebar.header("Filters")

if "timestamp ist" in df.columns:
    min_date = df["timestamp ist"].min().date()
    max_date = df["timestamp ist"].max().date()

    start_date = st.sidebar.date_input("Start Date", min_date)
    end_date = st.sidebar.date_input("End Date", max_date)

    start_datetime = pd.to_datetime(start_date)
    end_datetime = pd.to_datetime(end_date) + pd.Timedelta(days=1)

    df = df[
        (df["timestamp ist"] >= start_datetime) &
        (df["timestamp ist"] < end_datetime)
    ].copy()

# Sentiment filter
if "classification" in df.columns:
    sentiments = df["classification"].dropna().unique()
    selected_sentiment = st.sidebar.multiselect(
        "Select Sentiment",
        sentiments,
        default=sentiments
    )

    df = df[df["classification"].isin(selected_sentiment)].copy()

# =====================
# KPI Section
# =====================
st.markdown("## 📌 Overall Performance")

col1, col2, col3 = st.columns(3)

total_trades = len(df)
col1.metric("Total Trades", total_trades)

if total_trades > 0 and "closed pnl" in df.columns:
    total_pnl = df["closed pnl"].sum()
    win_rate = (df["closed pnl"] > 0).mean() * 100

    col2.metric("Total PnL", f"{total_pnl:,.2f}")
    col3.metric("Win Rate (%)", f"{win_rate:.2f}")
else:
    col2.metric("Total PnL", "N/A")
    col3.metric("Win Rate (%)", "N/A")

st.markdown("---")

# =====================
# Daily PnL Trend
# =====================
if total_trades > 0 and "closed pnl" in df.columns:

    st.markdown("## 📈 Daily PnL Trend")

    daily_pnl = df.groupby("date")["closed pnl"].sum().reset_index()

    fig_daily = px.line(
        daily_pnl,
        x="date",
        y="closed pnl",
        title="Daily Total PnL"
    )
    st.plotly_chart(fig_daily, width="stretch")

# =====================
# Performance by Sentiment
# =====================
if total_trades > 0 and "classification" in df.columns:

    st.markdown("## 📊 Performance by Sentiment")

    sentiment_summary = (
        df.groupby("classification")["closed pnl"]
        .mean()
        .reset_index()
    )

    fig_pnl = px.bar(
        sentiment_summary,
        x="classification",
        y="closed pnl",
        color="classification",
        title="Average PnL by Sentiment"
    )
    st.plotly_chart(fig_pnl, width="stretch")

    win_summary = (
        df.groupby("classification")["closed pnl"]
        .apply(lambda x: (x > 0).mean() * 100)
        .reset_index(name="win_rate")
    )

    fig_win = px.bar(
        win_summary,
        x="classification",
        y="win_rate",
        color="classification",
        title="Win Rate (%) by Sentiment"
    )
    st.plotly_chart(fig_win, width="stretch")

# =====================
# Trade Frequency
# =====================
if total_trades > 0 and "classification" in df.columns:

    st.markdown("## 📅 Daily Trade Frequency")

    trade_freq = (
        df.groupby(["date", "classification"])
        .size()
        .reset_index(name="trade_count")
    )

    fig_freq = px.line(
        trade_freq,
        x="date",
        y="trade_count",
        color="classification",
        title="Trade Frequency Trend"
    )
    st.plotly_chart(fig_freq, width="stretch")

# =====================
# Long / Short Behaviour
# =====================
if total_trades > 0 and "classification" in df.columns and "side" in df.columns:

    st.markdown("## ⚖ Long / Short Behaviour")

    side_counts = (
        df.groupby(["classification", "side"])
        .size()
        .reset_index(name="count")
    )

    side_counts["ratio"] = side_counts.groupby("classification")["count"].transform(
        lambda x: x / x.sum()
    )

    fig_side = px.bar(
        side_counts,
        x="classification",
        y="ratio",
        color="side",
        barmode="group",
        title="Long/Short Ratio by Sentiment"
    )
    st.plotly_chart(fig_side, width="stretch")

# =====================
# Volatility (Risk Proxy)
# =====================
if total_trades > 0 and "classification" in df.columns:

    st.markdown("## 📉 Risk (PnL Volatility)")

    volatility = df.groupby("classification")["closed pnl"].std().reset_index()

    fig_vol = px.bar(
        volatility,
        x="classification",
        y="closed pnl",
        color="classification",
        title="PnL Volatility by Sentiment"
    )
    st.plotly_chart(fig_vol, width="stretch")

# =====================
# Machine Learning Model
# =====================
from src.modeling import train_model

st.markdown("## -- Predictive Model Performance")

if total_trades > 0:

    model, accuracy, report, X_test, y_test = train_model(df)

    if model is not None:

        st.metric("Model Accuracy", f"{accuracy:.2f}")

        st.markdown("### 📋 Classification Report")
        st.text(report)

    else:
        st.warning("Required features not available for model training.")
        
# Strategic Recommendations
# =====================
st.markdown("## 🚀 Strategic Recommendations")

st.success("""
### 📊 Regime-Based Trading Strategy Framework

**1️⃣ Extreme Greed Regime**
- Highest average PnL observed but accompanied by elevated volatility.
- Momentum-driven trades perform well, but reversal risk increases sharply.
- Recommendation: Deploy momentum strategies with strict stop-loss and predefined profit booking.
- Avoid over-leveraging during late-stage greed cycles.

**2️⃣ Greed Regime**
- Moderate win rate with controlled volatility.
- Directional bias present (Long dominance observed).
- Recommendation: Use trend-following setups with disciplined position sizing.

**3️⃣ Fear Regime**
- Reduced win rate compared to greed phases.
- Volatility remains elevated with inconsistent profitability.
- Recommendation: Reduce exposure, tighten risk controls, and prioritize capital protection.

**4️⃣ Extreme Fear Regime**
- Lowest stability in performance and suppressed win rate.
- Emotional trading and uncertainty dominate.
- Recommendation: Avoid aggressive entries and limit trade frequency.
  Focus on defensive positioning or stay partially sidelined.

**5️⃣ Neutral Regime**
- Most stable and balanced performance.
- Risk-adjusted behavior comparatively smoother.
- Recommendation: Ideal regime for systematic strategy deployment.

---

### 🎯 Executive Insight

The analysis confirms that trader profitability is strongly regime-dependent.
Performance, volatility, and directional bias shift significantly across sentiment states.

Integrating market sentiment as a dynamic regime filter can:
• Improve risk-adjusted returns  
• Reduce drawdowns during extreme phases  
• Enhance trade timing and exposure control  
• Support disciplined behavioral trading  

Sentiment-aware strategy design provides a structural edge over static trading systems.
""")