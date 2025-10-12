# 03_preprocess_data.py
# Este script preprocesa los datos cargados, creando características adicionales y dividiendo los datos
# en conjuntos de entrenamiento y prueba.
# Autor: Equipo 11 - MLOps
# ===========================
# librerías necesarias para el script
from pathlib import Path
import yaml
from classes.PreprocessData import PreprocessData

if __name__ == "__main__":
    # Se leen los parámetros del archivo params.yaml
    params = yaml.safe_load(open("params.yaml", "r", encoding="utf-8"))    
    # Ejecutar la clase PreprocessData con los parámetros del archivo params.yaml
    PreprocessData(
        input_parquet=Path(params["data"]["interim_loaded"]),
        datetime_column=params["features"]["datetime_column"],
        target=params["features"]["target"]["preferred"],
        lags=params["features"]["lags"],
        test_size=params["split"]["test_size"],
        random_state=params["split"]["random_state"],
        out_train=Path(params["data"]["train"]),
        out_test=Path(params["data"]["test"]),
        out_cleaned=Path(params["data"]["cleaned"]),
    ).run()