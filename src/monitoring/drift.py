"""Utilities that simulate a data drift scenario and flag performance loss."""

from __future__ import annotations

import json
from math import sqrt
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

FEATURE_COLUMNS = [
    "temperature",
    "humidity",
    "wind_speed",
    "general_diffuse_flows",
    "diffuse_flows",
]
TARGET_COLUMN = "zone_1_power_consumption"
TIME_COLUMN = "timestamp"


def build_baseline_monitoring_data(records: int = 720, seed: int = 42) -> pd.DataFrame:
    """Construye un conjunto ordenado con alta correlación entre features y target."""
    rng = np.random.RandomState(seed)
    timestamps = pd.date_range("2025-01-01", periods=records, freq="10min")

    base_temperature = 25 + np.sin(np.linspace(0, 4 * np.pi, records))
    df = pd.DataFrame(
        {
            TIME_COLUMN: timestamps,
            "temperature": base_temperature + rng.normal(0, 1.5, records),
            "humidity": 50 + rng.normal(0, 5, records),
            "wind_speed": 5 + rng.normal(0, 1, records),
            "general_diffuse_flows": 150 + np.linspace(0, 20, records) + rng.normal(0, 3, records),
            "diffuse_flows": 80 + 0.5 * base_temperature + rng.normal(0, 2, records),
        }
    )

    df[TARGET_COLUMN] = (
        1000
        + df["temperature"] * 20
        + df["humidity"] * 3
        - df["wind_speed"] * 10
        + df["general_diffuse_flows"] * 2
        + rng.normal(0, 10, records)
    )
    return df


def apply_drift(df: pd.DataFrame) -> pd.DataFrame:
    """Introduce cambios de media/envestimiento y filas con features faltantes."""
    drifted = df.copy()
    drifted["temperature"] += 5
    drifted["humidity"] *= 0.85
    drifted["wind_speed"] *= 1.25
    seasonal_shift = 15 * np.sin(np.linspace(0, 8 * np.pi, len(df)))
    drifted["general_diffuse_flows"] += seasonal_shift
    drifted["diffuse_flows"] *= 0.9

    # Simula feature faltantes durante periodos puntuales
    drifted.loc[drifted.index % 15 == 0, "wind_speed"] = np.nan
    drifted = drifted.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN])

    # Aumenta la carga objetivo para reflejar mayor demanda real
    drifted[TARGET_COLUMN] += 25
    return drifted


def evaluate_dataset(
    model: LinearRegression,
    df: pd.DataFrame,
    feature_columns: Tuple[str, ...],
    target_column: str,
) -> Tuple[Dict[str, float], pd.Series, np.ndarray]:
    """Ejecuta inferencia y devuelve métricas con series para graficar."""
    df_clean = df.dropna(subset=list(feature_columns) + [target_column])
    X = df_clean[list(feature_columns)]
    y_true = df_clean[target_column]
    y_pred = model.predict(X)

    rmse = sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mask = y_true != 0
    mape = (np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]).mean() * 100) if mask.any() else float(
        "nan"
    )

    metrics = {"RMSE": float(rmse), "MAE": float(mae), "MAPE": float(mape)}
    return metrics, y_true, y_pred


def detect_drift(
    baseline_metrics: Dict[str, float],
    drift_metrics: Dict[str, float],
    relative_threshold: float = 0.15,
) -> Dict[str, Dict[str, float]]:
    """Compara métricas y marca alertas cuando se supera el umbral relativo."""
    alerts: Dict[str, Dict[str, float]] = {}
    for metric_name, baseline in baseline_metrics.items():
        drift_value = drift_metrics.get(metric_name, 0.0)
        if baseline <= 0:
            change = drift_value
        else:
            change = (drift_value - baseline) / baseline
        if change > relative_threshold:
            alerts[metric_name] = {
                "baseline": baseline,
                "drift": drift_value,
                "relative_change": float(change),
            }
    return alerts


def _plot_prediction_comparison(
    baseline_y: pd.Series,
    baseline_pred: np.ndarray,
    drift_y: pd.Series,
    drift_pred: np.ndarray,
    output_path: Path,
) -> None:
    """Guarda una línea comparativa de y_true y y_pred (antes/después de drift)."""
    plt.style.use("seaborn-v0_8")
    fig, ax = plt.subplots(figsize=(10, 4))
    baseline_idx = baseline_y.index[:200]
    ax.plot(baseline_idx, baseline_y.iloc[: len(baseline_idx)], label="Base actual", linewidth=1)
    ax.plot(baseline_idx, baseline_pred[: len(baseline_idx)], label="Base predicha", linestyle="--", linewidth=1)
    drift_idx = drift_y.index[:200]
    ax.plot(drift_idx, drift_y.iloc[: len(drift_idx)], label="Drift actual", linewidth=1)
    ax.plot(drift_idx, drift_pred[: len(drift_idx)], label="Drift predicha", linestyle="--", linewidth=1)
    ax.set_title("Comparación de preds antes/después del drift")
    ax.set_ylabel("Consumo (kW)")
    ax.set_xlabel("Timestamp")
    ax.legend(loc="upper left")
    fig.autofmt_xdate()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def run_drift_simulation(
    output_dir: Path, relative_threshold: float = 0.15, records: int = 720
) -> Dict[str, Dict[str, float]]:
    """Orquesta la simulación completa y persiste resultados para monitoreo."""
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline_df = build_baseline_monitoring_data(records=records)
    model = LinearRegression()
    model.fit(baseline_df[FEATURE_COLUMNS], baseline_df[TARGET_COLUMN])

    baseline_metrics, baseline_y, baseline_pred = evaluate_dataset(
        model, baseline_df, tuple(FEATURE_COLUMNS), TARGET_COLUMN
    )

    drift_df = apply_drift(baseline_df)
    drift_metrics, drift_y, drift_pred = evaluate_dataset(model, drift_df, tuple(FEATURE_COLUMNS), TARGET_COLUMN)

    alerts = detect_drift(baseline_metrics, drift_metrics, relative_threshold=relative_threshold)
    action = (
        "Revisar el pipeline de features y considerar reentrenar el modelo con datos más recientes."
        if alerts
        else "Sin alertas (el desempeño se mantiene dentro de lo esperado)."
    )

    results = {
        "baseline_metrics": baseline_metrics,
        "drift_metrics": drift_metrics,
        "alerts": alerts,
        "action": action,
    }

    output_path = output_dir / "drift_metrics.json"
    output_path.write_text(json.dumps(results, indent=2))

    _plot_prediction_comparison(
        baseline_y,
        baseline_pred,
        drift_y,
        drift_pred,
        output_dir / "drift_actual_vs_pred.png",
    )

    return results


def main() -> Dict[str, Dict[str, float]]:
    """Entry-point para el script de monitoreo de drift."""
    target_dir = Path("reports") / "drift_monitoring"
    return run_drift_simulation(target_dir)


if __name__ == "__main__":
    main()
