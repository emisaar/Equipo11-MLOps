# ExploreData.py
# Módulo que define la clase ExploreData para generar un EDA rápido.
# Produce reportes de texto y figuras básicas.  
# Autor: Equipo 11 - MLOps
# ===========================

# Librerías para anotaciones y dataclasses
from __future__ import annotations
from dataclasses import dataclass

# Librerías de utilidades
from pathlib import Path

# Librerías de manejo y visualización de datos
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



@dataclass
class ExploreData:
    """
    Genera un EDA rápido: head/describe/info y figuras básicas.

    Attributes
    ----------
    input_parquet : Path
        Ruta al Parquet de datos cargados (salida de LoadData).
    report_path : Path
        Carpeta donde se almacenarán reportes y figuras del EDA.
    sample_rows : int, default=5
        Número de filas a incluir en el ``head``.
    """

    input_parquet: Path
    report_path: Path
    sample_rows: int = 5

    def run(self) -> None:
        """Ejecuta la exploración de datos y guarda artefactos.

        Produce:
        - ``head.txt``: primeras ``sample_rows`` filas.
        - ``describe.csv``: estadísticas descriptivas.
        - ``info.txt``: shape, dtypes y conteo de nulos.
        - ``histograms.png``: histogramas de variables numéricas.
        - ``correlation_matrix.png``: mapa de calor con correlaciones.
        """
        
        # Crea la carpeta de reportes si no existe
        self.report_path.mkdir(parents=True, exist_ok=True)

        # Carga el dataset intermedio (ya validado por la etapa anterior)
        df = pd.read_parquet(self.input_parquet)

        # --- Archivos de texto: head / describe / info ---
        head_txt = self.report_path / "head.txt"
        describe_csv = self.report_path / "describe.csv"
        info_txt = self.report_path / "info.txt"

        # Guarda un vistazo rápido de las primeras filas
        head_txt.write_text(df.head(self.sample_rows).to_string())

        # Estadísticos para todas las columnas (incluye categóricas)
        df.describe(include="all").to_csv(describe_csv)

        # Construye un pequeño resumen: shape, dtypes y nulos
        buf = []
        buf.append(f"shape: {df.shape}")
        buf.append(f"dtypes:\n{df.dtypes}")
        buf.append(f"nulls:\n{df.isna().sum()}")
        info_txt.write_text("\n\n".join(buf))

        # --- Figuras: histogramas para numéricas ---
        # Selecciona solo columnas numéricas para evitar errores de plotting
        num_df = df.select_dtypes(include=[np.number])
        if num_df.shape[1] > 0:
            fig = plt.figure(figsize=(16, 10))
            num_df.hist(bins=20, figsize=(16, 10))
            plt.tight_layout()
            plt.savefig(self.report_path / "histograms.png", dpi=150)
            plt.close(fig)

        # --- Matriz de correlación si hay suficientes numéricas ---
        if num_df.shape[1] > 1:
            plt.figure(figsize=(12, 8))
            sns.heatmap(num_df.corr(), annot=False)
            plt.tight_layout()
            plt.savefig(self.report_path / "correlation_matrix.png", dpi=150)
            plt.close()