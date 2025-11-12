from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from src.data.loaders import LoadData
from src.features.engineering import create_ml_features
from src.modeling.evaluate import ModelEvaluator
from src.preprocessing.cleaner import DatasetCleaner


def _build_sample_dataframe(records: int = 150) -> pd.DataFrame:
    dates = pd.date_range("2025-01-01 00:00", periods=records, freq="10min")
    return pd.DataFrame(
        {
            "DateTime": dates.strftime("%m/%d/%Y %H:%M"),
            "Mixed Type Col": [str(i % 2) for i in range(records)],
            "Temperature": np.linspace(20.0, 25.0, records),
            "Humidity": np.linspace(40.0, 60.0, records),
            "Wind Speed": np.linspace(3.0, 8.0, records),
            "General Diffuse Flows": np.linspace(100.0, 200.0, records),
            "Diffuse Flows": np.linspace(50.0, 150.0, records),
            "Zone 1 Power Consumption": np.linspace(1000.0, 2000.0, records),
        }
    )


# Esta integración de extremo a extremo valida limpieza, entrenamiento y métricas.
def test_end_to_end_pipeline_generates_predictions_and_metrics(tmp_path: Path) -> None:
    raw_csv = tmp_path / "raw" / "power_series.csv"
    raw_csv.parent.mkdir(parents=True, exist_ok=True)
    _build_sample_dataframe().to_csv(raw_csv, index=False)

    loaded_parquet = tmp_path / "interim" / "loaded.parquet"
    loader = LoadData(input_path=raw_csv, output_path=loaded_parquet, parse_date_cols=())
    loaded_path = loader.run()
    assert loaded_path.exists()

    cleaned_parquet = tmp_path / "processed" / "cleaned.parquet"
    cleaner = DatasetCleaner(input_path=loaded_path, output_path=cleaned_parquet, datetime_column="datetime")
    cleaned_path = cleaner.run()
    assert cleaned_path.exists()

    cleaned_df = pd.read_parquet(cleaned_path).set_index("datetime")
    assert "mixed_type_col" not in cleaned_df.columns

    features, target = create_ml_features(cleaned_df, "zone_1_power_consumption")
    assert len(features) > 0

    model = LinearRegression()
    model.fit(features, target)
    predictions = model.predict(features)

    evaluator = ModelEvaluator(
        train_path=tmp_path / "dummy_train.parquet",
        test_path=tmp_path / "dummy_test.parquet",
        models_dir=tmp_path / "models",
        metrics_output=tmp_path / "metrics.json",
        figures_output=tmp_path / "figures",
    )
    metrics = evaluator._compute_metrics(target.values, predictions)

    assert metrics["RMSE"] < 1e-6
    assert metrics["MAE"] < 1e-6
    assert metrics["MAPE"] == pytest.approx(0.0, abs=1e-6)
