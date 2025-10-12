# 02_explore_data.py
# Este script explora los datos cargados y genera un reporte de análisis exploratorio.
# Autor: Equipo 11 - MLOps
# ===========================
# librerías necesarias para el script
from pathlib import Path
import yaml
from classes.ExploreData import ExploreData

if __name__ == "__main__":
    # Se leen los parámetros del archivo params.yaml
    params = yaml.safe_load(open("params.yaml", "r", encoding="utf-8"))
    # Ejecutar la clase ExploreData con los parámetros del archivo params.yaml
    ExploreData(
        input_parquet=Path(params["data"]["interim_loaded"]),
        report_path=Path(params["reports"]["eda_dir"]),
        sample_rows=params["eda"]["sample_rows"],
    ).run()