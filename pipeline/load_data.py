#!/usr/bin/env python
"""
Pipeline Stage 1: Carga de datos
Carga datos desde CSV y convierte a formato Parquet.
"""

from pathlib import Path
import yaml
from src.dataset import LoadData


if __name__ == "__main__":
    # Carga parámetros desde params.yaml
    params = yaml.safe_load(open("params.yaml", "r", encoding="utf-8"))

    # Ejecuta carga de datos
    LoadData(
        source=params["data"]["source"],
        input_path=Path(params["data"]["raw_path"]),
        datetime_column=params["preprocessing"]["datetime_column"],
        output_path=Path(params["data"]["interim_loaded"]),
    ).run()
