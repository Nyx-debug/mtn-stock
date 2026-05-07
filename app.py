import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os

from data_collector import fetch_mtn_data
from preprocess import preprocess, split_data
from model import train_model, predict
from evaluate import evaluate

# ═══════════════════════════════════════════════════════
# 📁 Ensure folders exist
# ═══════════════════════════════════════════════════════

os.makedirs("results", exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("data", exist_ok=True)

# ═══════════════════════════════════════════════════════
# 🔧 Helper Function
# ═══════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════
# 📊 STREAMLIT CONFIG
# ═══════════════════════════════════════════════════════

st.set_page_config(
    page_title="MTN Nigeria Stock Predictor",
    page_icon="📈",
    layout="wide"
)

st.title("📈 MTN Nigeria Stock Prediction Dashboard")

st.caption(
    "AI-powered stock forecasting system using Random Forest Regression "
    "and MTN Nigeria historical market data."
)

# ═══════════════════════════════════════════════════════
# 📂 LOAD DATA
# ═══════════════════════════════════════════════════════

data_path = "data/mtn_data.csv"

raw_df = pd.read_csv(data_path)

# remove unnamed columns
raw_df = raw_df.loc[:, ~raw_df.columns.str.contains("^Unnamed")]

# parse dates
raw_df["Date"] = pd.to_datetime(
    raw_df["Date"],
    format="%d-%b-%y",
    errors="coerce"
)

# fallback parser
missing = raw_df["Date"].isna()

if missing.any():
    raw_df.loc[missing, "Date"] = pd.to_datetime(
        raw_df.loc[missing, "Date"],
        format="%b-%y",
        errors="coerce"
    )

if raw_df["Date"].isna().any():
    raise ValueError("Unable to parse some Date values")

# normalize columns
raw_df = raw_df.rename(columns={
    "Price": "Close",
    "Vol.": "Vol"
})

# working dataframe
df = raw_df.copy()

df = df.rename(columns={
    "Vol": "Volume"
})

df["Volume"] = df["Volume"].apply(parse_volume)

# IMPORTANT: sort for ML
df = df.sort_values("Date").reset_index(drop=True)

# ═══════════════════════════════════════════════════════
# 🤖 AUTO MODEL TRAINING + METRICS
# ═══════════════════════════════════════════════════════

metrics_path = "results/metrics.json"
model_path = "models/mtn_model.pkl"

metrics = None
model = None

if (
    not os.path.exists(metrics_path)
    or not os.path.exists(model_path)
):

    st.warning("Model files not found. Training automatically...")

    # preprocess
    processed_df, features = preprocess(df)

    # split
    X_train, X_test, y_train, y_test = split_data(
        processed_df,
        features
    )

    # train
    model = train_model(X_train, y_train)

    # predict
    y_pred = predict(model, X_test)

    # evaluate
    metrics = evaluate(
        y_test,
        y_pred,
        model=model,
        features=features
    )

    st.success("Model trained successfully.")

else:

    # load model
    model = joblib.load(model_path)

    # load metrics
    with open(metrics_path) as f:
        metrics = json.load(f)

# ═══════════════════════════════════════════════════════
# 📊 CURRENT MARKET SNAPSHOT
# ═══════════════════════════════════════════════════════

st.markdown("---")

st.subheader("📊 Current Market Snapshot")

latest = raw_df.iloc[0]
previous = raw_df.iloc[1]

change = latest["Close"] - previous["Close"]

pct_change = (
    change / previous["Close"]
) * 100

col1, col2, col3 = st.columns(3)

col1.metric(
    "Current Price (₦)",
    f"{latest['Close']:.2f}",
    f"{change:.2f} ({pct_change:.2f}%)"
)

col2.metric(
    "Daily High",
    f"{latest['High']:.2f}"
)

col3.metric(
    "Volume",
    latest["Vol"]
)

# ═══════════════════════════════════════════════════════
# 🤖 MODEL PERFORMANCE
# ═══════════════════════════════════════════════════════

st.markdown("---")

st.subheader("🤖 Model Performance")

col1, col2, col3 = st.columns(3)

col1.metric(
    "MAE (₦)",
    f"{metrics['MAE']:.2f}"
)

col2.metric(
    "RMSE (₦)",
    f"{metrics['RMSE']:.2f}"
)

col3.metric(
    "R² Score",
    f"{metrics['R2']:.4f}"
)

# ═══════════════════════════════════════════════════════
# 🔄 MODE SELECTOR
# ═══════════════════════════════════════════════════════

st.markdown("---")

mode = st.radio(
    "Select Prediction Mode",
    [
        "Use Latest Market Data",
        "Manual Input"
    ]
)

# ═══════════════════════════════════════════════════════
# 🔮 PREDICTION INPUTS
# ═══════════════════════════════════════════════════════

st.markdown("---")

st.subheader("🔮 Predict Tomorrow's Closing Price")

if mode == "Use Latest Market Data":

    open_price = latest["Open"]
    high_price = latest["High"]
    low_price = latest["Low"]
    close_price = latest["Close"]
    volume = parse_volume(latest["Vol"])

    st.info("Using latest real market data")

else:

    open_price = st.number_input(
        "Open Price",
        value=float(latest["Open"])
    )

    high_price = st.number_input(
        "High Price",
        value=float(latest["High"])
    )

    low_price = st.number_input(
        "Low Price",
        value=float(latest["Low"])
    )

    close_price = st.number_input(
        "Close Price",
        value=float(latest["Close"])
    )

    volume = st.number_input(
        "Volume",
        value=parse_volume(latest["Vol"])
    )

# ═══════════════════════════════════════════════════════
# 🧠 FEATURE ENGINEERING
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

        temp_df = pd.concat(
            [temp_df, new_row],
            ignore_index=True
        )

    # moving averages
    temp_df["MA10"] = (
        temp_df["Close"]
        .rolling(10)
        .mean()
    )

    temp_df["MA50"] = (
        temp_df["Close"]
        .rolling(50)
        .mean()
    )

    # returns
    temp_df["Return"] = (
        temp_df["Close"]
        .pct_change()
    )

    # volatility
    temp_df["Volatility"] = (
        temp_df["Return"]
        .rolling(10)
        .std()
    )

    # lag features
    temp_df["Lag1"] = (
        temp_df["Close"]
        .shift(1)
    )

    temp_df["Lag2"] = (
        temp_df["Close"]
        .shift(2)
    )

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
# 🔮 PREDICT
# ═══════════════════════════════════════════════════════

if st.button("Predict Tomorrow's Close"):

    if model is None:

        st.error("Model unavailable.")

    else:

        features = prepare_features()

        prediction = model.predict(features)[0]

        diff = prediction - latest["Close"]

        pct = (
            diff / latest["Close"]
        ) * 100

        st.success(
            f"Predicted Price: ₦{prediction:.2f}"
        )

        st.info(
            f"Expected Change: ₦{diff:.2f} ({pct:.2f}%)"
        )

# ═══════════════════════════════════════════════════════
# 📉 HISTORICAL PRICE TREND
# ═══════════════════════════════════════════════════════

st.markdown("---")

st.subheader("📉 Historical Price Trend")

plot_df = df.copy()

plot_df = plot_df.sort_values("Date")

plot_df["MA10"] = (
    plot_df["Close"]
    .rolling(10)
    .mean()
)

st.line_chart(
    plot_df.set_index("Date")[[
        "Close",
        "MA10"
    ]]
)

# ═══════════════════════════════════════════════════════
# 📊 VOLUME CHART
# ═══════════════════════════════════════════════════════

st.markdown("---")

st.subheader("📊 Trading Volume")

volume_chart = plot_df.set_index("Date")[["Volume"]]

st.bar_chart(volume_chart)

# ═══════════════════════════════════════════════════════
# 📋 RAW DATA
# ═══════════════════════════════════════════════════════

st.markdown("---")

st.subheader("📋 Raw Dataset")

st.dataframe(raw_df)