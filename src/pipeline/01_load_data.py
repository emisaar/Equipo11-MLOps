# 01_load_data.py
# Este script carga los datos desde una fuente especificada y los guarda en un archivo Parquet.
# Autor: Equipo 11 - MLOps
# ===========================
# librerías necesarias para el script
from pathlib import Path
import yaml
from classes.LoadData import LoadData

if __name__ == "__main__":
    # Se leen los parámetros del archivo params.yaml
    params = yaml.safe_load(open("params.yaml", "r", encoding="utf-8"))
    # Ejecutar la clase LoadData con los parámetros del archivo params.yaml
    LoadData(
        source=params["data"]["source"],
        input_path=Path(params["data"]["raw_path"]),
        datetime_column=params["features"]["datetime_column"],
        output_path=Path(params["data"]["interim_loaded"]),
    ).run()