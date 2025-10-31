# loaders.py
# Módulo que define la clase LoadData para cargar datos desde CSV o Parquet.
# Aplica limpieza básica a los nombres de las columnas.
# ===========================

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import pandas as pd

from src.preprocessing.normalization import clean_name


@dataclass
class LoadData:
    """
    Carga datos desde archivos CSV o Parquet.
    Normaliza nombres de columnas y realiza conversión de tipos básica.

    Parameters
    ----------
    input_path : Path
        Ruta al archivo CSV o Parquet
    output_path : Path
        Ruta al archivo Parquet de salida
    parse_date_cols : Sequence[str]
        Columnas a parsear como fechas
    """
    input_path: Path
    output_path: Path
    parse_date_cols: Sequence[str] = ("DateTime", "datetime")

    def run(self) -> Path:
        """
        Ejecuta la etapa de carga de datos y guarda el resultado como Parquet.

        Lee el archivo CSV/Parquet desde self.input_path y lo guarda como Parquet.

        Returns
        -------
        Path
            Ruta del archivo Parquet generado.
        """
        # Asegura que el directorio de salida exista
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        # Carga desde archivo
        df = self._load_file()

        # Guarda como Parquet
        df.to_parquet(self.output_path, index=False)
        return self.output_path

    def _load_file(self) -> pd.DataFrame:
        """
        Carga desde CSV o Parquet según la extensión de self.input_path.
        Aplica parseo de fechas y limpieza básica de nombres de columnas.

        Returns
        -------
        pd.DataFrame
            DataFrame cargado y procesado

        Raises
        ------
        ValueError
            Si el formato de archivo no es soportado
        FileNotFoundError
            Si el archivo no existe
        """
        # validaciones básicas de existencia de archivo
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
        """
        Verifica si un CSV contiene una columna específica.

        Parameters
        ----------
        path : Path
            Ruta al archivo CSV
        col : str
            Nombre de la columna a buscar

        Returns
        -------
        bool
            True si la columna existe, False en caso contrario
        """
        try:
            headers = pd.read_csv(path, nrows=0).columns
            return col in headers
        except Exception:
            return False

    def _postprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Post-procesa el DataFrame cargado:
        - Limpieza básica de nombres de columnas.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame a post-procesar

        Returns
        -------
        pd.DataFrame
            DataFrame post-procesado
        """
        # Limpieza básica de nombres de columnas
        df.columns = [clean_name(c) for c in df.columns]

        return df
