import pandas as pd
import os


def fetch_mtn_data(file_path="data/mtn_data.csv"):
    """
    Load and clean MTN Nigeria dataset
    Expected columns:
    Date | Close | Open | High | Low | Vol. | Change %
    """

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    print("[DataCollector] Loading dataset...")

    df = pd.read_csv(file_path)
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]  # drop extra trailing empty column

    # Rename columns to the internal names expected by the model pipeline.
    df.rename(columns={
        "Price": "Close",
        "Vol.": "Volume",
        "Vol": "Volume"
    }, inplace=True)

    # Convert Date with mixed day-month-year and month-year formats
    df["Date"] = pd.to_datetime(df["Date"], format='%d-%b-%y', errors='coerce')
    missing = df["Date"].isna()
    if missing.any():
        df.loc[missing, "Date"] = pd.to_datetime(df.loc[missing, "Date"], format='%b-%y', errors='coerce')
    if df["Date"].isna().any():
        raise ValueError("Unable to parse some Date values in the CSV")

    # Clean Volume
    def convert_volume(v):
        if isinstance(v, str):
            v = v.replace(",", "")
            if "M" in v:
                return float(v.replace("M", "")) * 1_000_000
            elif "K" in v:
                return float(v.replace("K", "")) * 1_000
            elif "B" in v:
                return float(v.replace("B", "")) * 1_000_000_000
            elif v.strip() == "-" or v.strip() == "":
                return 0.0
        return float(v)

    df["Volume"] = df["Volume"].apply(convert_volume)

    # Drop Change %
    if "Change %" in df.columns:
        df.drop(columns=["Change %"], inplace=True)

    # Sort and clean
    df = df.sort_values("Date").reset_index(drop=True)
    df = df[df["Close"] > 0]

    print(f"[DataCollector] Loaded {len(df)} rows")

    return df


if __name__ == "__main__":
    df = fetch_mtn_data()
    print(df.head())