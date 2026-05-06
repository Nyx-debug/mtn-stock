import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json

def parse_volume(value):
    if pd.isna(value):
        return np.nan
    if isinstance(value, str):
        value = value.strip().upper().replace(",", "")
        if value.endswith("M"):
            return float(value[:-1]) * 1_000_000
        if value.endswith("K"):
            return float(value[:-1]) * 1_000
        if value.endswith("B"):
            return float(value[:-1]) * 1_000_000_000
        return float(value)
    return float(value)

# ── Load model ──
model = joblib.load("models/mtn_model.pkl")

# ── Load dataset ──
raw_df = pd.read_csv("data/mtn_data.csv")
raw_df = raw_df.loc[:, ~raw_df.columns.str.contains('^Unnamed')]
raw_df["Date"] = pd.to_datetime(raw_df["Date"], format='%d-%b-%y', errors='coerce')
missing = raw_df["Date"].isna()
if missing.any():
    raw_df.loc[missing, "Date"] = pd.to_datetime(raw_df.loc[missing, "Date"], format='%b-%y', errors='coerce')
if raw_df["Date"].isna().any():
    raise ValueError("Unable to parse some Date values in data/mtn_data.csv")

# Normalize volume column names for display
raw_df = raw_df.rename(columns={"Vol.": "Vol", "Vol": "Vol"})

# Keep the original file order for display and latest market snapshots.
# Use a sorted, renamed copy only for feature engineering and model input.
df = raw_df.copy()
df = df.rename(columns={"Price": "Close", "Vol": "Volume"})
df["Volume"] = df["Volume"].apply(parse_volume)
df = df.sort_values("Date").reset_index(drop=True)

# ── Load metrics ──
metrics = None
try:
    with open("results/metrics.json") as f:
        metrics = json.load(f)
except FileNotFoundError:
    pass

st.set_page_config(layout="wide")
st.title("📈 MTN Nigeria Stock Prediction Dashboard")

# ═══════════════════════════════════════════════════════
# 📊 CURRENT MARKET SNAPSHOT
# ═══════════════════════════════════════════════════════

st.subheader("📊 Current Market Snapshot")

latest = raw_df.iloc[0]
previous = raw_df.iloc[1]

change = latest["Close"] - previous["Close"]
pct_change = (change / previous["Close"]) * 100

col1, col2, col3 = st.columns(3)

col1.metric(
    "Current Price (₦)",
    f"{latest['Close']:.2f}",
    f"{change:.2f} ({pct_change:.2f}%)"
)

col2.metric("Daily High", f"{latest['High']:.2f}")
col3.metric("Volume", latest["Vol"])

st.markdown("---")

# ═══════════════════════════════════════════════════════
# 🤖 MODEL PERFORMANCE
# ═══════════════════════════════════════════════════════

st.subheader("🤖 Model Performance")

if metrics:
    col1, col2, col3 = st.columns(3)
    col1.metric("MAE (₦)", f"{metrics['MAE']:.2f}")
    col2.metric("RMSE (₦)", f"{metrics['RMSE']:.2f}")
    col3.metric("R² Score", f"{metrics['R2']:.3f}")
else:
    st.info("📊 Metrics not available. Run `python main.py` locally to generate evaluation metrics.")

st.markdown("---")

# ═══════════════════════════════════════════════════════
# 🔄 MODE SELECTOR
# ═══════════════════════════════════════════════════════

mode = st.radio(
    "Select Prediction Mode",
    ["Use Latest Market Data", "Manual Input"]
)

st.markdown("---")

# ═══════════════════════════════════════════════════════
# 🔮 INPUT SECTION
# ═══════════════════════════════════════════════════════

st.subheader("🔮 Predict Tomorrow's Closing Price")

if mode == "Use Latest Market Data":
    open_price = latest["Open"]
    high_price = latest["High"]
    low_price  = latest["Low"]
    close_price = latest["Close"]
    volume = parse_volume(latest["Vol"])

    st.info("Using latest real market data")

else:
    open_price = st.number_input("Open Price", value=float(latest["Open"]))
    high_price = st.number_input("High Price", value=float(latest["High"]))
    low_price  = st.number_input("Low Price", value=float(latest["Low"]))
    close_price = st.number_input("Close Price", value=float(latest["Close"]))
    volume = st.number_input("Volume", value=parse_volume(latest["Vol"]))


# ═══════════════════════════════════════════════════════
# 🧠 FEATURE ENGINEERING (MATCHES TRAINING)
# ═══════════════════════════════════════════════════════

def prepare_features():
    temp_df = df.copy()

    if mode == "Manual Input":
        new_row = pd.DataFrame({
            "Date": [pd.Timestamp.today()],
            "Open": [open_price],
            "High": [high_price],
            "Low": [low_price],
            "Close": [close_price],
            "Volume": [volume]
        })

        temp_df = pd.concat([temp_df, new_row], ignore_index=True)

    temp_df["MA10"] = temp_df["Close"].rolling(10).mean()
    temp_df["MA50"] = temp_df["Close"].rolling(50).mean()
    temp_df["Return"] = temp_df["Close"].pct_change()
    temp_df["Volatility"] = temp_df["Return"].rolling(10).std()
    temp_df["Lag1"] = temp_df["Close"].shift(1)
    temp_df["Lag2"] = temp_df["Close"].shift(2)

    latest_row = temp_df.iloc[-1]

    features = np.array([[
        latest_row["Open"],
        latest_row["High"],
        latest_row["Low"],
        latest_row["Volume"],
        latest_row["MA10"],
        latest_row["MA50"],
        latest_row["Volatility"],
        latest_row["Lag1"],
        latest_row["Lag2"]
    ]])

    return features


# ═══════════════════════════════════════════════════════
# 🔮 PREDICTION
# ═══════════════════════════════════════════════════════

if st.button("Predict Tomorrow's Close"):
    features = prepare_features()
    prediction = model.predict(features)[0]

    diff = prediction - latest["Close"]
    pct = (diff / latest["Close"]) * 100

    st.success(f"Predicted Price: ₦{prediction:.2f}")
    st.info(f"Expected Change: ₦{diff:.2f} ({pct:.2f}%)")

st.markdown("---")

# ═══════════════════════════════════════════════════════
# 📉 PRICE CHART
# ═══════════════════════════════════════════════════════

st.subheader("📉 Historical Price Trend")
plot_df = raw_df.copy()
plot_df["Close_MA10"] = plot_df["Close"].rolling(10).mean()

st.line_chart(plot_df[["Close", "Close_MA10"]])

st.markdown("---")
st.subheader("📋 Raw Data (file order)")
st.dataframe(raw_df)