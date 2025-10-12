# 04_train_model.py
# Este script entrena un modelo de machine learning utilizando los datos preprocesados.
# Autor: Equipo 11 - MLOps
# ===========================
# librerías necesarias para el script
from pathlib import Path
import yaml
from classes.TrainModel import TrainModel

if __name__ == "__main__":
    # Se leen los parámetros del archivo params.yaml
    params = yaml.safe_load(open("params.yaml", "r", encoding="utf-8"))
    # Ejecutar la clase TrainModel con los parámetros del archivo params.yaml
    TrainModel(
        train_parquet=Path(params["data"]["train"]),
        target=params["features"]["target"]["preferred"],
        model_out=Path(params["model"]["path"]),
        model_params=params["model"]["model_params"],
    ).run()