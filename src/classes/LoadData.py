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