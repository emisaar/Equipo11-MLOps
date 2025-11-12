import math
import numpy as np
from pathlib import Path

from src.modeling.evaluate import ModelEvaluator


# Fábrica auxiliar para instanciar ModelEvaluator con rutas temporales.
def _make_evaluator(tmp_path: Path) -> ModelEvaluator:
    return ModelEvaluator(
        train_path=tmp_path / "train.parquet",
        test_path=tmp_path / "test.parquet",
        models_dir=tmp_path / "models",
        metrics_output=tmp_path / "metrics.json",
        figures_output=tmp_path / "figures",
    )


# Verifica métricas contra cálculos manuales de RMSE/MAE.
def test_compute_metrics_matches_known_values(tmp_path: Path) -> None:
    evaluator = _make_evaluator(tmp_path)
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.5, 2.5])

    metrics = evaluator._compute_metrics(y_true, y_pred)
    assert math.isclose(metrics["RMSE"], math.sqrt(((0.0) ** 2 + (0.5) ** 2 + (-0.5) ** 2) / 3))
    assert math.isclose(metrics["MAE"], (0.0 + 0.5 + 0.5) / 3)
    assert metrics["MAPE"] > 0


# Asegura que MAPE sea NaN cuando los valores reales son cero para evitar divisiones por cero.
def test_compute_metrics_handles_zero_truth(tmp_path: Path) -> None:
    evaluator = _make_evaluator(tmp_path)
    y_true = np.array([0.0, 0.0])
    y_pred = np.array([5.0, 5.0])

    metrics = evaluator._compute_metrics(y_true, y_pred)
    assert math.isnan(metrics["MAPE"])
