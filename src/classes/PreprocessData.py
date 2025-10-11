# PreprocessData.py
# Módulo que define la clase PreprocessData para preprocesar datos de series temporales.
# Realiza ingeniería de rasgos, genera lags y divide en train/test. 
# Autor: Equipo 11 - MLOps
# ===========================

# Librerías para anotaciones y dataclasses
from __future__ import annotations
from dataclasses import dataclass

# Librerías de utilidades
from pathlib import Path

# Librerías de manejo de datos
import pandas as pd
import numpy as np
from typing import List, Tuple
import re

# Librerías de machine learning
from sklearn.model_selection import train_test_split

@dataclass
class PreprocessData:
    """Realiza ingeniería de rasgos, genera lags y divide en train/test.

    Attributes
    ----------
    input_parquet : Path
        Ruta al Parquet "loaded" (salida de LoadData).
    datetime_column : str
        Nombre de la columna datetime para rasgos y orden temporal.
    target : str
        Nombre de la variable objetivo a modelar.
    lags : list[int]
        Desplazamientos para crear lags del target.
    test_size : float
        Proporción del conjunto de test (0 < test_size < 1).
    random_state : int
        Semilla para reproducibilidad en el split.
    out_train : Path
        Ruta donde se guardará el parquet de entrenamiento.
    out_test : Path
        Ruta donde se guardará el parquet de prueba.
    """

    input_parquet: Path
    datetime_column: str
    target: str
    lags: List[int]
    test_size: float
    random_state: int
    out_train: Path
    out_test: Path
    
    def add_time_features(self, df: pd.DataFrame, dt_col: str) -> pd.DataFrame:
        """Agrega rasgos temporales derivados de una columna datetime.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame de entrada.
        dt_col : str
            Nombre de la columna datetime a partir de la cual se derivan rasgos.

        Returns
        -------
        pd.DataFrame
            Copia del DataFrame con columnas ``hour``, ``dayofweek`` y ``month``
            si ``dt_col`` existe; en caso contrario retorna el DataFrame original.
        """
        # Sólo procede si la columna existe para evitar KeyError
        if dt_col in df.columns:
            # Trabaja sobre una copia para no mutar el argumento original
            out = df.copy()
            # ``.dt`` expone accesores vectorizados para Series datetime64
            out["hour"] = out[dt_col].dt.hour
            out["dayofweek"] = out[dt_col].dt.dayofweek
            out["month"] = out[dt_col].dt.month
            return out
        # Si no hay columna datetime, retorna sin cambios
        return df


    def add_lags(self, df: pd.DataFrame, dt_col: str, target: str, lags: List[int]) -> pd.DataFrame:
        """Crea variables rezagadas (lags) del target para modelar dependencia temporal.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame de entrada.
        dt_col : str
            Columna de fecha/hora usada para ordenar antes de aplicar lags.
        target : str
            Nombre de la columna objetivo (la serie a predecir).
        lags : list[int]
            Lista de desplazamientos (en filas) para crear ``target_lag{L}``.

        Returns
        -------
        pd.DataFrame
            Copia del DataFrame con columnas de lags añadidas (una por valor L).
        """
        out = df.copy()

        # Ordena temporalmente si la columna datetime existe; evita fuga de info
        if dt_col in out.columns:
            out = out.sort_values(dt_col)

        # Genera cada columna lag mediante ``Series.shift(L)``
        for L in lags:
            out[f"{target}_lag{L}"] = out[target].shift(L)
        return out
    def _norm(self, s: str) -> str:
        """Normaliza un nombre para comparar de forma flexible:
        - a minúsculas
        - sin espacios
        - sólo caracteres [a-z0-9]
        """
        return re.sub(r"[^0-9a-z]+", "", str(s).strip().lower())

    def resolve_column(self, requested: str, df: pd.DataFrame, *, friendly_name: str) -> str:
        """
        Devuelve el nombre REAL de la columna en df que mejor coincide con 'requested'.
        - Primero intenta coincidencia exacta.
        - Luego por forma normalizada (_norm).
        - Finalmente sugiere columnas si no encuentra.
        """
        cols = list(df.columns)

        # 1) exacta
        if requested in df.columns:
            return requested

        # 2) normalizada
        req_n = self._norm(requested)
        matches = [c for c in cols if self._norm(c) == req_n]
        if matches:
            return matches[0]

        # 3) sugerencias
        suggestions = [c for c in cols if req_n in self._norm(self,c)]
        hint = f"Sugerencias: {suggestions[:5]}" if suggestions else f"Columnas disponibles: {cols[:10]}..."
        raise ValueError(f"No se encontró la columna '{requested}' para {friendly_name}. {hint}")
    
    def run(self) -> Tuple[Path, Path]:
        """Ejecuta el preprocesamiento y el split en train/test.

        Pasos:
        1) Carga datos intermedios (Parquet).
        2) Resuelve nombres reales de columnas (datetime y target).
        3) Elimina filas con ``target`` nulo.
        4) Agrega rasgos temporales y lags.
        5) Elimina NaN producidos por lags y quita la columna datetime original.
        6) Asegura que X sea 100% numérica y divide en train/test sin shuffle.
        7) Persiste ``train.parquet`` y ``test.parquet``.
        """
        # Asegura directorio de salida (p. ej. data/processed/)
        self.out_train.parent.mkdir(parents=True, exist_ok=True)

        # Carga
        df = pd.read_parquet(self.input_parquet)

        # Resolver columnas reales (robusto a espacios, guiones, mayúsculas)
        real_dt_col = self.resolve_column(self.datetime_column, df, friendly_name="datetime_column")
        real_target = self.resolve_column(self.target, df, friendly_name="target")

        # (Opcional) diagnóstico rápido
        # print(f"[Preprocess] datetime={real_dt_col} | target={real_target}")

        # Coaccionar el target a numérico (convierte 'invalid' -> NaN)
        df[real_target] = pd.to_numeric(df[real_target], errors="coerce")
        
        
        # Filtrar nulos del target
        df = df.dropna(subset=[real_target])

        # Rasgos temporales + lags
        df = self.add_time_features(df, real_dt_col)
        if self.lags:
            df = self.add_lags(df, real_dt_col, real_target, self.lags)

        # Limpiar NaN generados por lags y quitar la columna datetime original
        df = df.dropna().reset_index(drop=True)
        if real_dt_col in df.columns:
            df = df.drop(columns=[real_dt_col])

        # Separar X/y y asegurar que X sea numérica
        X = df.drop(columns=[real_target]).select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan).dropna()        
        # Re-alinear y con X (por si se cayeron filas al limpiar inf/NaN)
        y = df.loc[X.index, real_target]                

        # Split temporal (sin barajar para evitar fuga de información)
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.test_size,
            random_state=self.random_state,
            shuffle=False,
        )

        # Reconstruir DataFrames con target
        train_df = X_train.copy()
        train_df[real_target] = y_train
        test_df = X_test.copy()
        test_df[real_target] = y_test

        # Persistir
        train_df.to_parquet(self.out_train, index=False)
        test_df.to_parquet(self.out_test, index=False)
        return self.out_train, self.out_test