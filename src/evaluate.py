
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def main():
    root = Path(__file__).resolve().parents[1]
    metrics = pd.read_csv(root / "reports" / "model_metrics.csv")
    pivot = metrics.pivot_table(index="model", columns="dataset", values="rmse")
    ax = pivot.plot(kind="bar", figsize=(8,4))
    ax.set_title("RMSE por modelo y dataset"); ax.set_ylabel("RMSE")
    (root / "reports" / "figures").mkdir(parents=True, exist_ok=True)
    plt.tight_layout(); plt.savefig(root / "reports" / "figures" / "rmse_by_model.png"); plt.close()

if __name__ == "__main__":
    main()
