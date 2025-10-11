# LoadData.py
# Módulo que define la clase LoadData para cargar datos desde CSV o Parquet.
# Aplica limpieza básica a los nombres de las columnas.
# Autor: Equipo 11 - MLOps  
# ===========================

# Librerías para anotaciones y dataclasses
from __future__ import annotations
from dataclasses import dataclass

# Librerías de utilidades
from classes.Util import Util, PowerConsumptionDAO
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
    - path: ruta a archivo (si source="file")
    - table: nombre de tabla (si source="db" y no se usa query)
    - query: consulta SQL completa (opcional, prioridad sobre table)
    - parse_date_cols: posibles nombres de columna datetime a detectar/parsear
    - coerce_mixed_col: nombre de columna a forzar a numérica si existe
    
    """    
    source: str = "file"  # "file" | "db"
    path: Optional[str] = None
    
    # Parsing/normalización
    parse_date_cols: Sequence[str] = ("DateTime", "datetime")
    coerce_mixed_col: Optional[str] = "mixed_type_col"

    def run(self, page: Optional[int] = None, size: Optional[int] = None) -> pd.DataFrame:
        """
        Ejecuta la carga:
        - Si source="file": lee CSV/Parquet desde self.path.
        - Si source="db": lee desde MySQL usando self.query o self.table.
        La paginación (LIMIT/OFFSET) aplica sólo cuando source="db".
        """
        # Carga según source
        src = (self.source or "file").lower()
        if src == "file":
            return self._load_file()
        elif src == "db":
            return self._load_db(page=page, size=size)
        else:
            raise ValueError("source debe ser 'file' o 'db'.")       
            
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
        if not self.path:
            raise ValueError("Debes proporcionar 'path' para source='file'.")
        path = Path(self.path)

        if not path.exists():
            raise FileNotFoundError(f"No existe el archivo: {self.path}")
        
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
        
        # Conversión de tipos si es necesario coerce (para evitar errores)
        # Si hay una columna que debería ser numérica pero tiene strings
        if self.coerce_mixed_col and self.coerce_mixed_col in df.columns:
            df[self.coerce_mixed_col] = pd.to_numeric(df[self.coerce_mixed_col], errors="coerce")

        # Parseo robusto de datetime: intenta con parse_date_cols normalizados
        dt_col = None
        for c in self.parse_date_cols:
            cn = Util.clean_name(c)
            if cn in df.columns:
                dt_col = cn
                # Asegura tipo datetime
                df[dt_col] = pd.to_datetime(df[dt_col], errors="coerce", infer_datetime_format=True)
                break

        # Si existe 'datetime' explícito, priorízalo como columna principal
        if "datetime" in df.columns:
            if not pd.api.types.is_datetime64_any_dtype(df["datetime"]):
                df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce", infer_datetime_format=True)
            dt_col = "datetime"

        # Orden por datetime si está disponible
        if dt_col and dt_col in df.columns:
            df = df.sort_values(dt_col).reset_index(drop=True)

        return df