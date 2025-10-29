#!/usr/bin/env python
"""
Pipeline Stage 2: Limpieza de datos
Ejecuta limpieza completa: outliers, imputación, transformaciones.
"""

from pathlib import Path
import yaml
from src.dataset import DatasetCleaner


if __name__ == "__main__":
    # Carga parámetros desde params.yaml
    params = yaml.safe_load(open("params.yaml", "r", encoding="utf-8"))

    # Ejecuta limpieza de datos
    DatasetCleaner(
        input_path=Path(params["data"]["interim_loaded"]),
        output_path=Path(params["data"]["cleaned"]),
        datetime_column=params["preprocessing"]["datetime_column"],
    ).run()
