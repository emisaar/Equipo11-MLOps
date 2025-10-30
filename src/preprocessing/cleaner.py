# cleaner.py
# Clase para limpieza y preprocesamiento completo de datos
# ===========================

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import numpy as np

from src.preprocessing.normalization import normalize_column_names
from src.preprocessing.imputation import generar_media_movil, imputar_nans_con_media_movil
from src.preprocessing.outliers import outliers
from src.config import (
    WINDOW,
    CENTER,
    MIN_PERIODS,
    OUTLIER_METHOD,
    OUTLIER_LIMIT_SIDE,
    OUTLIER_REPLACE_METRIC,
    LOG_TRANSFORM_COLS,
)


@dataclass
class DatasetCleaner:
    """
    Realiza limpieza y preprocesamiento completo de datos.

    Pipeline completo:
    1. Normalizar nombres de columnas
    2. Eliminar columnas innecesarias (mixed_type_col)
    3. Parsear y limpiar datetime
    4. Convertir columnas numéricas
    5. Imputar NaNs con media móvil
    6. Manejar outliers con IQR/zscore
    7. Aplicar transformaciones logarítmicas
    8. Guardar datos limpios

    Parameters
    ----------
    input_path : Path
        Ruta al archivo Parquet cargado (salida de LoadData)
    output_path : Path
        Ruta donde guardar el Parquet limpio
    datetime_column : str, default="datetime"
        Nombre de la columna datetime
    """

    input_path: Path
    output_path: Path
    datetime_column: str = "datetime"

    def run(self) -> Path:
        """
        Ejecuta el pipeline completo de limpieza.

        Returns
        -------
        Path
            Ruta del archivo Parquet limpio generado
        """
        # Asegura directorio de salida
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        # Carga datos
        df = pd.read_parquet(self.input_path)

        print("\nIniciando limpieza de datos...")
        print(f"   Dimensiones originales: {df.shape}")

        # Pipeline de limpieza
        df = self._normalize_columns(df)
        df = self._remove_unnecessary_columns(df)
        df = self._clean_datetime(df)
        df = self._convert_numeric_columns(df)
        df = self._impute_missing_values(df)
        df = self._handle_outliers(df)
        df = self._apply_transformations(df)

        print(f"   Dimensiones finales: {df.shape}")
        print(f"Limpieza completada\n")

        # Guardar
        df.to_parquet(self.output_path, index=False)
        return self.output_path

    def _normalize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normaliza nombres de columnas."""
        print("   • Normalizando nombres de columnas...")
        return normalize_column_names(df)

    def _remove_unnecessary_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Elimina columnas innecesarias."""
        print("   • Eliminando columnas innecesarias...")
        if 'mixed_type_col' in df.columns:
            df = df.drop(columns=['mixed_type_col'])
        return df

    def _clean_datetime(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Limpia y parsea la columna datetime.

        Pasos:
        1. Elimina filas con datetime NaN
        2. Limpia espacios en blanco
        3. Parsea a datetime
        4. Ordena por datetime
        """
        print("   • Limpiando columna datetime...")

        # Elimina filas con datetime NaN
        total_filas = df.shape[0]
        df = df[df[self.datetime_column].notna()].copy()
        filas_eliminadas = total_filas - df.shape[0]
        if filas_eliminadas > 0:
            print(f"      • Eliminadas {filas_eliminadas} filas con datetime NaN")

        # Limpia espacios y parsea
        df[self.datetime_column] = df[self.datetime_column].astype(str).str.strip()
        df['datetime_parsed'] = pd.to_datetime(
            df[self.datetime_column],
            format='%m/%d/%Y %H:%M',
            errors='coerce'
        )

        # Elimina filas no convertibles
        total_filas = df.shape[0]
        df = df[df['datetime_parsed'].notna()].copy()
        filas_eliminadas = total_filas - df.shape[0]
        if filas_eliminadas > 0:
            print(f"      • Eliminadas {filas_eliminadas} filas con datetime no convertible")

        # Reemplaza columna original
        df[self.datetime_column] = df['datetime_parsed']
        df = df.drop(columns=['datetime_parsed'])

        # Ordena y reinicia índices
        df = df.sort_values(self.datetime_column).reset_index(drop=True)

        return df

    def _convert_numeric_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convierte columnas numéricas.

        Identifica columnas tipo object (excluyendo datetime) y las convierte a float.
        """
        print("   • Convirtiendo columnas numéricas...")

        object_cols = df.select_dtypes(include=['object']).columns
        numeric_cols = [col for col in object_cols if col != self.datetime_column]

        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        df = df.reset_index(drop=True)
        return df

    def _impute_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Imputa valores faltantes usando media móvil.

        Pasos:
        1. Genera media móvil para cada columna numérica
        2. Imputa NaNs con la media móvil
        3. Elimina registros con >5% de valores NaN
        """
        print("   • Imputando valores faltantes con media móvil...")

        # Identifica columnas numéricas (excluye datetime)
        object_cols = df.select_dtypes(include=['object']).columns
        numeric_cols = [col for col in df.columns
                       if col not in object_cols and col != self.datetime_column]

        # Imputa con media móvil
        for col in numeric_cols:
            media_movil = generar_media_movil(
                df[col],
                window=WINDOW,
                center=CENTER,
                min_periods=MIN_PERIODS
            )
            df[col] = imputar_nans_con_media_movil(df[col], media_movil)

        # Elimina registros con porcentaje alto de NaN (<5%)
        total_filas = len(df)
        cols_to_check = pd.DataFrame(df.isnull().mean() * 100, columns=['avg_nan']).query('avg_nan < 5').index
        df = df.dropna(subset=cols_to_check).reset_index(drop=True)

        filas_eliminadas = total_filas - len(df)
        if filas_eliminadas > 0:
            print(f"      • Eliminadas {filas_eliminadas} filas con >5% NaN")

        return df

    def _handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Maneja outliers usando IQR/zscore.

        Aplica la función outliers() a cada columna numérica,
        reemplazando outliers superiores con media móvil.
        """
        print("   • Manejando outliers...")

        # Identifica columnas numéricas (excluye datetime)
        object_cols = df.select_dtypes(include=['object']).columns
        numeric_cols = [col for col in df.columns
                       if col not in object_cols and col != self.datetime_column]

        for col in numeric_cols:
            outliers(
                df,
                str(col),
                OUTLIER_METHOD,
                limit_side=OUTLIER_LIMIT_SIDE,
                replace=True,
                replace_metric=OUTLIER_REPLACE_METRIC,
                ma_window=WINDOW,
                ma_center=CENTER,
                ma_min_periods=MIN_PERIODS
            )
            print(80 * '-')

        return df

    def _apply_transformations(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aplica transformaciones logarítmicas.

        Transforma columnas con distribución sesgada a la derecha.
        """
        print("   • Aplicando transformaciones logarítmicas...")

        for col in LOG_TRANSFORM_COLS:
            if col in df.columns:
                df[col] = np.log(df[col])
                print(f"      • Transformación log aplicada a '{col}'")

        return df
