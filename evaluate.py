import os
import json
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def evaluate(y_test, y_pred, model=None, features=None, results_dir="results"):
    """
    Evaluate model performance and save outputs
    """

    os.makedirs(results_dir, exist_ok=True)

    # ── Metrics ──
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    metrics = {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "R2": float(r2)
    }

    print("\n[Evaluation] Results:")
    print(f"MAE  : NGN {mae:.2f}")
    print(f"RMSE : NGN {rmse:.2f}")
    print(f"R2   : {r2:.4f}")

    # ── Save metrics ──
    with open(os.path.join(results_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    # ── Plot Actual vs Predicted ──
    if HAS_MATPLOTLIB:
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred, alpha=0.5)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2)
        plt.xlabel('Actual Price (NGN)')
        plt.ylabel('Predicted Price (NGN)')
        plt.title('Actual vs Predicted Stock Prices')
        plt.grid(True)
        plot_path = os.path.join(results_dir, "actual_vs_predicted.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[Evaluation] Actual vs Predicted plot saved to {plot_path}")

        # ── Residual Plot ──
        residuals = y_pred - y_test
        plt.figure(figsize=(10, 6))
        plt.scatter(y_pred, residuals, alpha=0.5)
        plt.axhline(y=0, color='r', linestyle='--', linewidth=2)
        plt.xlabel('Predicted Price (NGN)')
        plt.ylabel('Residuals (NGN)')
        plt.title('Residual Plot')
        plt.grid(True)
        residual_path = os.path.join(results_dir, "residual_plot.png")
        plt.savefig(residual_path, dpi=300, bbox_inches='tight')
        plt.close()
        # ── Residual Distribution (Histogram) ──
        plt.figure(figsize=(10, 6))
        plt.hist(residuals, bins=30, alpha=0.7, color='blue', edgecolor='black')
        plt.xlabel('Residuals (NGN)')
        plt.ylabel('Frequency')
        plt.title('Residual Distribution')
        plt.grid(True)
        hist_path = os.path.join(results_dir, "residual_distribution.png")
        plt.savefig(hist_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[Evaluation] Residual distribution plot saved to {hist_path}")

        # ── Save Actual vs Predicted Table ──
        import pandas as pd
        table_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
        table_path = os.path.join(results_dir, "actual_vs_predicted.csv")
        table_df.to_csv(table_path, index=False)
        print(f"[Evaluation] Actual vs Predicted table saved to {table_path}")
        if model is not None and features is not None and hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1]
            plt.figure(figsize=(10, 6))
            plt.title('Feature Importances')
            plt.bar(range(len(features)), importances[indices], align='center')
            plt.xticks(range(len(features)), [features[i] for i in indices], rotation=45)
            plt.xlabel('Features')
            plt.ylabel('Importance')
            plt.tight_layout()
            fi_path = os.path.join(results_dir, "feature_importance.png")
            plt.savefig(fi_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"[Evaluation] Feature importance plot saved to {fi_path}")
        else:
            print("[Evaluation] Feature importance plot requires model with feature_importances_ and features list.")

    else:
        print("[Evaluation] matplotlib not installed; skipping plot generation.")

    return metrics