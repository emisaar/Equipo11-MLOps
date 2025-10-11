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

# Librerías de manejo y visualización de datos
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Librerías de machine learning
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

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
    metrics_json : Path
        Ruta donde se guardará el JSON con métricas.
    figures_dir : Path
        Carpeta donde se guardarán las figuras de evaluación.
    """

    test_parquet: Path
    model_path: Path
    target: str
    metrics_json: Path
    figures_dir: Path

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
        self.figures_dir.mkdir(parents=True, exist_ok=True)

        # Carga el bundle serializado con joblib
        bundle = joblib.load(self.model_path)
        model = bundle["model"]
        features = bundle["features"]

        # Carga test y respeta el orden de features para predecir correctamente
        df = pd.read_parquet(self.test_parquet)
        X_test = df[features]
        y_test = df[self.target]

        # Genera predicciones
        y_pred = model.predict(X_test)
        
        mse = float(mean_squared_error(y_test, y_pred))
        # Calcula métricas de regresión principales
        metrics = {
            "mae": float(mean_absolute_error(y_test, y_pred)),
            "mse": mse,
            "rmse": float(np.sqrt(mse)),
            "r2": float(r2_score(y_test, y_pred)),
            "n_test": int(len(y_test)),
        }

        # Persiste las métricas en JSON (legible por DVC para comparaciones)
        self.metrics_json.parent.mkdir(parents=True, exist_ok=True)
        with open(self.metrics_json, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

        # --- Figura 1: serie temporal y_true vs y_pred ---
        plt.figure(figsize=(10, 6))
        plt.plot(y_test.values, label="y_true")
        plt.plot(y_pred, label="y_pred")
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.figures_dir / "y_true_vs_y_pred.png", dpi=150)
        plt.close()

        # --- Figura 2: dispersión (predicho vs real) ---
        plt.figure(figsize=(6, 6))
        plt.scatter(y_test, y_pred, s=8)
        plt.xlabel("y_true")
        plt.ylabel("y_pred")
        plt.tight_layout()
        plt.savefig(self.figures_dir / "scatter_true_vs_pred.png", dpi=150)
        plt.close()

        return self.metrics_json