# TrainModel.py
# Módulo que define la clase TrainModel para entrenar un modelo de regresión.   
# Utiliza RandomForestRegressor con hiperparámetros configurables.
# Autor: Equipo 11 - MLOps
# ===========================
# Librerías para anotaciones y dataclasses
from __future__ import annotations
from dataclasses import dataclass

# Librerías de utilidades
from pathlib import Path
import joblib

# Librerías de manejo de datos
import pandas as pd
from typing import Dict, Optional

# Librerías de machine learning
from sklearn.ensemble import RandomForestRegressor

@dataclass
class TrainModel:
    """Entrena un RandomForestRegressor y guarda el bundle del modelo.

    Attributes
    ----------
    train_parquet : Path
        Ruta al Parquet de entrenamiento (salida de PreprocessData).
    target : str
        Nombre de la variable objetivo.
    model_out : Path
        Ruta donde se guardará el pickle con el modelo y el orden de features.
    rf_params : dict | None
        Diccionario de hiperparámetros para RandomForestRegressor; si es None
        se usan valores por defecto razonables.
    """

    train_parquet: Path
    target: str
    model_out: Path
    rf_params: Optional[Dict] = None

    def run(self) -> Path:
        """Ejecuta el entrenamiento del modelo.

        Carga el conjunto de entrenamiento, separa X/y, inicializa y ajusta
        un RandomForestRegressor con ``rf_params`` y persiste un *bundle*
        (diccionario) con: ``{"model": model, "features": lista_de_columnas}``.

        Returns
        -------
        Path
            Ruta del archivo pickle del modelo entrenado.
        """
        # Asegura carpeta de modelos (p. ej. models/)
        self.model_out.parent.mkdir(parents=True, exist_ok=True)

        # Carga los datos de entrenamiento
        df = pd.read_parquet(self.train_parquet)

        # Separa características (todas menos target) y objetivo
        X_train = df.drop(columns=[self.target])
        y_train = df[self.target]

        # Usa hiperparámetros provistos o defaults robustos
        params = self.rf_params or {
            "n_estimators": 300,
            "max_depth": None,
            "n_jobs": -1,
            "random_state": 42,
        }

        # Inicializa y ajusta el modelo de bosque aleatorio
        model = RandomForestRegressor(**params)
        model.fit(X_train, y_train)

        # Guarda modelo + orden de features para reproducir el mismo vector X
        joblib.dump({"model": model, "features": list(X_train.columns)}, self.model_out)
        return self.model_out