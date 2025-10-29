# dataset.py
# Módulo que define la clase LoadData para cargar datos desde CSV o Parquet.
# Aplica limpieza básica a los nombres de las columnas.
# ===========================

# Librerías para anotaciones y dataclasses
from __future__ import annotations
from dataclasses import dataclass

# Librerías de utilidades
from src.utils import Util, PowerConsumptionDAO
from pathlib import Path

# Librerías de manejo de datos
import pandas as pd
from typing import Optional, Sequence


@dataclass
class LoadData:
    """
    Carga datos desde CSV/Parquet o desde Base de Datos (MySQL).
    Normaliza nombres de columnas y realiza conversión de tipos básica.
    
    Parámetros:
    - source: "file" (CSV/Parquet) o "db"
    - input_path: ruta a archivo (si source="file")
    - datetime_column: nombre de columna datetime
    - output_path: ruta a archivo Parquet interim
    """    
    source: str = "file"  # "file" | "db"
    input_path: Optional[Path] = None
    datetime_column: str = None
    output_path: Path = None    
    
    # Parsing/normalización
    parse_date_cols: Sequence[str] = ("DateTime", "datetime")
    coerce_mixed_col: Optional[str] = "mixed_type_col"

    def run(self, page: Optional[int] = None, size: Optional[int] = None) -> Path:
        """
        Ejecuta la etapa de carga de datos y guarda el resultado como Parquet en
        ``output_path``:
        - Si source="file": lee CSV/Parquet desde self.path.
        - Si source="db": lee desde MySQL usando self.query o self.table.
            La paginación (LIMIT/OFFSET) aplica sólo cuando source="db".
        
        Returns        
        -Path
            Ruta del archivo Parquet generado.
        """    
        # Asegura que el directorio de salida exista
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
                   
        # Carga según source
        src = (self.source or "file").lower()
        if src == "file":
            df = self._load_file()
        elif src == "db":
            df = self._load_db(page=page, size=size)
        else:
            raise ValueError("source debe ser 'file' o 'db'.")       
        
        # Guarda como Parquet
        df.to_parquet(self.output_path, index=False)
        return self.output_path
            
    def _load_file(self) -> pd.DataFrame:
        """
        Carga desde CSV o Parquet según la extensión de self.path.
        Aplica parseo de fechas y limpieza básica de nombres de columnas.
        
        Raises        
        -ValueError
            Si el formato de archivo no es soportado.
        -FileNotFoundError
            Si el archivo no existe.
        -ValueError
            Si no se proporciona self.path.
        
        """
        # validaciones básicas de path y existencia de archivo
        if not self.input_path:
            raise ValueError("Debes proporcionar 'path' para source='file'.")
        path = Path(self.input_path)

        if not path.exists():
            raise FileNotFoundError(f"No existe el archivo: {self.input_path}")
        
        # Detecta formato por extensión
        suffix = path.suffix.lower()
        if suffix == ".csv":
            # Detecta columnas datetime presentes para parsearlas
            use_parse = [c for c in self.parse_date_cols if self._csv_has_column(path, c)]
            df = pd.read_csv(path, parse_dates=use_parse or None)
        elif suffix in {".parquet", ".pq"}:
            df = pd.read_parquet(path)
        else:
            raise ValueError("Formato no soportado. Use .csv o .parquet")

        return self._postprocess(df)

    def _csv_has_column(self, path: Path, col: str) -> bool:
        try:
            headers = pd.read_csv(path, nrows=0).columns
            return col in headers
        except Exception:
            return False

    def _load_db(self, page: Optional[int], size: Optional[int]) -> pd.DataFrame:
        """
        Carga desde MySQL usando el DAO PowerConsumptionDAO.fetch_data().
        Ignora 'table' y 'query' porque el DAO ya está especializado a power_consumption.
        """
        # Valores por defecto de paginación si no se pasan
        page = page or 1
        size = size or 1000

        try:
            payload = PowerConsumptionDAO.fetch_data(page=page, size=size)
            items = payload.get("items", [])
            df = pd.DataFrame(items)
        except Exception as e:
            raise RuntimeError(f"Error al cargar datos desde PowerConsumptionDAO: {e}")

        # Si no hay datos, devuelve DF vacío coherente
        if df.empty:
            # columnas esperadas por el DAO
            cols = ["datetime", "zone1", "zone2", "zone3", "total_power", "temperature", "humidity"]
            df = pd.DataFrame(columns=cols)

        return self._postprocess(df)     
         
    def _postprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Post-procesa el DataFrame cargado:
        - Limpieza básica de nombres de columnas.
        - Conversión a numérica de coerce_mixed_col si existe.
        - Detección y parseo de columna datetime si existe.
        - Ordena por datetime si está disponible.
        """
        # Limpieza básica de nombres de columnas
        df.columns = [Util.clean_name(c) for c in df.columns]

        return df


# ===== Preprocesamiento Avanzado =====
# Clase para limpieza completa de datos

import numpy as np
from src.utils import (
    normalize_column_names,
    generar_media_movil,
    imputar_nans_con_media_movil,
    outliers
)
from src.config import (
    WINDOW,
    CENTER,
    MIN_PERIODS,
    OUTLIER_METHOD,
    OUTLIER_LIMIT_SIDE,
    OUTLIER_REPLACE_METRIC,
    LOG_TRANSFORM_COLS,
    WEATHER_COLS,
    POWER_COLS
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
    datetime_column : str
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