import joblib
import numpy as np
import pandas as pd

from pathlib import Path
from dataclasses import dataclass

from src.modeling.predict import predict_model, predict_with_features


@dataclass
class _DummyModel:
    """Minimal model that returns constant predictions."""

    value: float = 3.14

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.full(len(X), self.value)


# Verifica que predict_model use el DataFrame dado y retorne un arreglo numpy.
def test_predict_model_reads_dataframe_and_returns_array(tmp_path: Path) -> None:
    model_bundle = {"model": _DummyModel(value=2.5), "features": ["a", "b"]}
    model_path = tmp_path / "model.pkl"
    joblib.dump(model_bundle, model_path)

    input_df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    result = predict_model(model_path, input_df)

    assert isinstance(result, np.ndarray)
    assert result.tolist() == [2.5, 2.5]


# Garantiza que predict_with_features pueda regresar DataFrame con predicciones.
def test_predict_with_features_returns_dataframe(tmp_path: Path) -> None:
    model_bundle = {"model": _DummyModel(value=1.0), "features": ["a"]}
    model_path = tmp_path / "model.pkl"
    joblib.dump(model_bundle, model_path)

    input_df = pd.DataFrame({"a": [0, 1]})
    df_result = predict_with_features(model_path, input_df, return_dataframe=True)

    assert "prediction" in df_result.columns
    assert df_result["prediction"].tolist() == [1.0, 1.0]
