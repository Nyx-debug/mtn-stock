from data_collector import fetch_mtn_data
from preprocess import preprocess, split_data
from model import train_model, predict
from evaluate import evaluate


def generate_additional_plots(df):
    """
    Generate additional plots for the report
    """
    import matplotlib.pyplot as plt
    import os

    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)

    # ── Price with Moving Average ──
    plt.figure(figsize=(12, 6))
    plt.plot(df['Date'], df['Close'], label='Close Price', alpha=0.7)
    plt.plot(df['Date'], df['MA10'], label='10-Day MA', linewidth=2)
    plt.plot(df['Date'], df['MA50'], label='50-Day MA', linewidth=2)
    # Add smoothed price
    df['Smoothed'] = df['Close'].rolling(30).mean()
    plt.plot(df['Date'], df['Smoothed'], label='30-Day Smoothed', linewidth=2, linestyle='--')
    plt.xlabel('Date')
    plt.ylabel('Price (NGN)')
    plt.title('MTN Stock Price with Moving Averages and Smoothed Line')
    plt.legend()
    plt.grid(True)
    price_ma_path = os.path.join(results_dir, "price_with_ma.png")
    plt.savefig(price_ma_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[Additional] Price with MA and smoothed plot saved to {price_ma_path}")

    # ── Trading Volume Over Time ──
    plt.figure(figsize=(12, 6))
    plt.bar(df['Date'], df['Volume'], color='green', alpha=0.7, width=1)
    plt.xlabel('Date')
    plt.ylabel('Volume')
    plt.title('MTN Trading Volume Over Time')
    plt.grid(True)
    volume_path = os.path.join(results_dir, "volume_over_time.png")
    plt.savefig(volume_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[Additional] Volume over time bar chart saved to {volume_path}")


def main():
    print("\n=== MTN Nigeria Stock Prediction System ===\n")

    # ── STEP 1: Load & Clean Data ──
    print("STEP 1: DATA LOADING")
    df = fetch_mtn_data()

    # ── STEP 2: Preprocess ──
    print("\nSTEP 2: PREPROCESSING")
    df, features = preprocess(df)

    X_train, X_test, y_train, y_test = split_data(df, features)

    # ── STEP 3: Train Model ──
    print("\nSTEP 3: TRAINING MODEL")
    model = train_model(X_train, y_train)

    # ── STEP 4: Predict ──
    print("\nSTEP 4: PREDICTION")
    y_pred = predict(model, X_test)

    # ── STEP 5: Evaluate ──
    print("\nSTEP 5: EVALUATION")
    metrics = evaluate(y_test, y_pred, model=model, features=features)

    # ── Additional Plots ──
    print("\nSTEP 6: ADDITIONAL PLOTS")
    generate_additional_plots(df)

    print("\n=== DONE ===")


if __name__ == "__main__":
    main()