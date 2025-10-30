#!/usr/bin/env python
"""
Pipeline Stage 3: Preprocesamiento y división de datos
Crea features, genera lags y divide en train/test.
"""

from pathlib import Path
import yaml
from src.features.preprocessor import PreprocessData


if __name__ == "__main__":
    # Carga parámetros desde params.yaml
    params = yaml.safe_load(open("params.yaml", "r", encoding="utf-8"))

    # Ejecuta preprocesamiento
    PreprocessData(
        input_parquet=Path(params["data"]["cleaned"]),
        datetime_column=params["preprocessing"]["datetime_column"],
        target=params["preprocessing"]["target_column"],
        lags=params["preprocessing"]["lags"],
        test_size=params["split"]["test_size"],
        random_state=params["split"]["random_state"],
        out_train=Path(params["data"]["train"]),
        out_test=Path(params["data"]["test"]),
        out_cleaned=Path(params["data"]["preprocessed"]),
    ).run()
