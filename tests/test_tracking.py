import numpy as np
import pandas as pd

from pathlib import Path

from src.tracking.mlflow_tracker import MLflowTracker


class _InnerEstimator:
    def get_params(self):
        return {"n_estimators": 5, "max_depth": 3}


class _DummyModel:
    def __init__(self):
        self.best_lag = 2
        self.rf_model = _InnerEstimator()

    def get_params(self):
        return {"learning_rate": 0.1, "complex": [1]}


# Confirms tracker aggregates simple params and ignores unsupported attributes.
def test_extract_model_params_merges_attributes() -> None:
    tracker = MLflowTracker(enabled=False)
    params = tracker._extract_model_params(_DummyModel())
    assert params["learning_rate"] == 0.1
    assert params["best_lag"] == 2
    assert params["n_estimators"] == 5
    assert "complex" not in params


# Verifies artifact export writes CSV with y_true/y_pred for monitoring.
def test_log_predictions_artifact_creates_csv(tmp_path: Path) -> None:
    tracker = MLflowTracker(enabled=False)
    dates = pd.date_range("2025-01-01 00:00", periods=3, freq="10min")
    test_df = pd.DataFrame({"zone_1_power_consumption": [1.0, 2.0, 3.0]}, index=dates)
    predictions = np.array([1.1, 2.1, 3.1])

    artifact_path = tracker._log_predictions_artifact(
        zone="zone_1_power_consumption",
        model_name="rf",
        test_df=test_df,
        predictions=predictions,
        run_dir=tmp_path,
        n_steps=3,
    )

    assert artifact_path.exists()
    assert "pred_rf_zone_1_power_consumption.csv" in artifact_path.name
    df = pd.read_csv(artifact_path)
    assert "y_true" in df.columns
    assert np.allclose(df["y_pred"], predictions[:3])
