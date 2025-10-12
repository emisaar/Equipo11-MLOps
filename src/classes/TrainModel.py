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
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor

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
    model_params: Optional[Dict] = None    

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
              
        # Inicializa y ajusta el modelo de bosque aleatorio
        model = XGBRegressor(**self.model_params)
        pre = ColumnTransformer([("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), X_train.columns.tolist())])
        pipe = Pipeline([("prep", pre), ("model", model)])        
        pipe.fit(X_train, y_train)                                
                    
        # Guarda modelo + orden de features para reproducir el mismo vector X
        joblib.dump({"model": model, "features": list(X_train.columns), "params": self.model_params}, self.model_out)
        return self.model_out