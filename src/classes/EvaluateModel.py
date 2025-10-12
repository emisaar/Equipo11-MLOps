# EvaluateModel.py
# Módulo que define la clase EvaluateModel para evaluar un modelo entrenado.
# Evalúa el modelo en un conjunto de prueba, calcula métricas de regresión y
# genera figuras de diagnóstico.
# Autor: Equipo 11 - MLOps
# ===========================

# Librerías para anotaciones y dataclasses
from __future__ import annotations
from dataclasses import dataclass

# Librerías de utilidades
import joblib
import json
from pathlib import Path
from dotenv import load_dotenv
import os

# Librerías de manejo y visualización de datos
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Optional

# Librerías de machine learning
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Librerias de MLFlow
import mlflow

@dataclass
class EvaluateModel:
    """Evalúa el modelo en test y genera métricas y figuras de diagnóstico.

    Attributes
    ----------
    test_parquet : Path
        Ruta al Parquet de prueba (salida de PreprocessData).
    model_path : Path
        Ruta al pickle con el bundle del modelo (salida de TrainModel).
    target : str
        Nombre de la variable objetivo.
    metrics_path : Path
        Ruta donde se guardará el JSON con métricas.
    figures_path : Path
        Carpeta donde se guardarán las figuras de evaluación.
    """

    test_parquet: Path
    model_path: Path
    target: str
    metrics_path: Path
    figures_path: Path
    tracking_uri: Optional[str]  = None
    experiment_name: Optional[str]  = None
    run_name: Optional[str]  = None
        

    def run(self) -> Path:
        """Ejecuta la evaluación del modelo y persiste resultados.

        Pasos:
        1) Carga bundle del modelo (modelo + orden de features).
        2) Carga test, alinea columnas con ``features`` y predice.
        3) Calcula MAE, RMSE y R^2; guarda ``metrics.json``.
        4) Genera figuras: serie y_true vs y_pred y diagrama de dispersión.

        Returns
        -------
        Path
            Ruta al archivo JSON de métricas generado.
        """
        # Asegura la carpeta de salida de figuras (p. ej. reports/figures)
        self.figures_path.mkdir(parents=True, exist_ok=True)
        
        # inicializa MLFlow
        if self.tracking_uri == None:
            # Si el tracking URL no se proporciona, usa la variable de entorno            
            self.tracking_uri = os.getenv("MLFLOW_TRACKING_URI")        
            
        if self.experiment_name == None:
            # Si el nombre del experimineto no se proporciona, usa la variable de entorno
            self.experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME")
            
        if self.run_name == None:
            # Se agrega el nombre de la corrida
            self.run_name = os.getenv("ML_FLOW_RUN_NAME")
            
        # Se iniacializa MLFlow        
        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.experiment_name)  

        # Carga el bundle serializado con joblib
        bundle = joblib.load(self.model_path)
        model = bundle["model"]
        features = bundle["features"]
        params = bundle["params"]

        # Carga test y respeta el orden de features para predecir correctamente
        df = pd.read_parquet(self.test_parquet)
        X_test = df[features]
        y_test = df[self.target]

        # Genera predicciones
        y_pred = model.predict(X_test)
        
        # Uso basico de MLFlow
        with mlflow.start_run(run_name=self.run_name) as run:
            mlflow.log_params(params=params) 
            
            # Calcula métricas de regresión principales
            mae = float(mean_absolute_error(y_test, y_pred))
            mse = float(mean_squared_error(y_test, y_pred))
            rmse = float(np.sqrt(mse))
            r2 = float(r2_score(y_test, y_pred))
            n_test = int(len(y_test))
            
            # Log de las metricas
            mlflow.log_metric("mae", mae)
            mlflow.log_metric("mse", mse)
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("r2", r2)
            
            # Log del model
            mlflow.sklearn.log_model(
                model,
                artifact_path="model",
                registered_model_name=self.run_name) 
            
            # Metricas de con el numero de corrida de MLFlow
            results = {"run_id": run.info.run_id,
                       "metrics": {"mae": mae,
                                   "mse": mse,
                                   "rmse": rmse,
                                   "r2": r2,
                                   "n_test": n_test}
                        }            

            # Persiste las métricas en JSON (legible por DVC para comparaciones)
            self.metrics_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.metrics_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2)

            # --- Figura 1: serie temporal y_true vs y_pred ---
            plt.figure(figsize=(10, 6))
            plt.plot(y_test.values, label="y_true")
            plt.plot(y_pred, label="y_pred")
            plt.legend()
            plt.tight_layout()
            plt.savefig(self.figures_path / "y_true_vs_y_pred.png", dpi=150)
            plt.close()

            # --- Figura 2: dispersión (predicho vs real) ---
            plt.figure(figsize=(6, 6))
            plt.scatter(y_test, y_pred, s=8)
            plt.xlabel("y_true")
            plt.ylabel("y_pred")
            plt.tight_layout()
            plt.savefig(self.figures_path / "scatter_true_vs_pred.png", dpi=150)
            plt.close()

        return results