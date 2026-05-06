import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def preprocess(df):
    """
    Feature engineering for MTN stock prediction
    """

    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], format='%d-%b-%y', errors='coerce')
    missing = df["Date"].isna()
    if missing.any():
        df.loc[missing, "Date"] = pd.to_datetime(df.loc[missing, "Date"], format='%b-%y', errors='coerce')
    if df["Date"].isna().any():
        raise ValueError("Unable to parse some Date values in the dataset")
    df = df.sort_values("Date").reset_index(drop=True)

    # ── TARGET (Next Day Price) ──
    df["Next_Close"] = df["Close"].shift(-1)

    # ── MOVING AVERAGES ──
    df["MA10"] = df["Close"].rolling(10).mean()
    df["MA50"] = df["Close"].rolling(50).mean()

    # ── RETURNS ──
    df["Return"] = df["Close"].pct_change()

    # ── VOLATILITY ──
    df["Volatility"] = df["Return"].rolling(10).std()

    # ── LAG FEATURES ──
    df["Lag1"] = df["Close"].shift(1)
    df["Lag2"] = df["Close"].shift(2)

    # ── DROP MISSING VALUES ──
    df = df.dropna().reset_index(drop=True)

    # ── FEATURES ──
    features = [
        "Open", "High", "Low", "Volume",
        "MA10", "MA50", "Volatility",
        "Lag1", "Lag2"
    ]

    print(f"[Preprocess] Rows: {len(df)} | Features: {len(features)}")

    return df, features


def split_data(df, features, test_size=0.2, random_state=42):
    """
    Random train/test split for validation.
    """

    X = df[features].values
    y = df["Next_Close"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        shuffle=True
    )

    print(f"[Preprocess] Train: {len(X_train)} | Test: {len(X_test)}")

    return X_train, X_test, y_train, y_test