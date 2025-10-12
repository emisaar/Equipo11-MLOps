# 05_evaluate_model.py
# Este script evalúa el modelo entrenado utilizando el conjunto de prueba y genera métricas e informes.
# Autor: Equipo 11 - MLOps
# ===========================
# librerías necesarias para el script
from pathlib import Path
import yaml
from classes.EvaluateModel import EvaluateModel

if __name__ == "__main__":
    # se leen los parámetros del archivo params.yaml
    params = yaml.safe_load(open("params.yaml", "r", encoding="utf-8"))
    # se ejecuta la clase EvaluateModel con los parámetros del archivo params.yaml
    EvaluateModel(
        test_parquet=Path(params["data"]["test"]),
        model_path=Path(params["model"]["path"]),
        target=params["features"]["target"]["preferred"],
        metrics_path=Path(params["metrics"]["path"]),
        figures_path=Path(params["reports"]["figures_dir"]),
        tracking_uri=params["mlfow"]["tracking_uri"],
        experiment_name=params["mlfow"]["experiment_name"],
        run_name=params["mlfow"]["run_name"],
    ).run()