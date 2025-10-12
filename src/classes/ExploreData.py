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
import io
import math

# Librerías de manejo y visualización de datos
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Iterable, Tuple, Any


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
        
        # Configuración de visualización
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
        pd.set_option('display.max_columns', None)
        pd.set_option('display.max_rows', 100)

        # Carga el dataset intermedio (ya validado por la etapa anterior)
        df = pd.read_parquet(self.input_parquet)

        # --- Archivos de texto: head / describe / info ---
        head_txt = self.report_path / "01_head.txt"
        tail_txt = self.report_path / "02_tail.txt"
        info_txt = self.report_path / "03_info.txt"
        describe_txt = self.report_path / "04_describe.txt"
        summary_txt = self.report_path / "05_summary.txt"
        missing_txt = self.report_path / "06_missing.txt"
        not_number_txt = self.report_path / "07_not_number.txt"
        mixed_type_txt = self.report_path / "08_mixed_type_txt.txt"
        datetime_txt = self.report_path / "09_datetime_txt.txt"
        outliers_txt = self.report_path / "10_outliers_txt.txt"      
               
        # Guarda un vistazo rápido de las primeras filas
        head_txt.write_text(df.head(self.sample_rows).to_string())
                
        # Guarda un vistazo rápido de las ultimas filas
        tail_txt.write_text(df.tail(self.sample_rows).to_string())       
        
        # Info del DataFrame
        buf = io.StringIO()
        df.info(buf=buf)
        info_txt.write_text(buf.getvalue(), encoding="utf-8")
        
        # Estadísticos para todas las columnas (incluye categóricas)
        describe_txt.write_text(df.describe(include="all").to_string())

        # Construye un pequeño resumen: shape, dtypes y nulos
        buf = []
        buf.append(f"shape: {df.shape}")            # Dimensiones
        buf.append(f"dtypes:\n{df.dtypes}")         # Tipos de datos
        buf.append(f"nulls:\n{df.isna().sum().sort_values(ascending=False)}")    # Conteo de nulos
        buf.append(f"Filas completamente duplicadas: { df.duplicated().sum()}") # Detección de duplicados
        
        # Verificar duplicados temporales
        columnas_fecha = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
        if columnas_fecha:
            for col in columnas_fecha:
                duplicados_fecha = df[col].duplicated().sum()
                buf.append(f"Timestamps duplicados en '{col}': {duplicados_fecha}")
        
        summary_txt.write_text("\n\n".join(buf))
                
        # Obtiene el % de valores faltantes en el dataframe (Por columna) y loa ordena de manera descendente
        # Conteo de valores faltantes
        missing_data = pd.DataFrame({
            'Columna': df.columns,
            'Missing_Count': df.isnull().sum(),
            'Missing_Percent': (df.isnull().sum() / len(df)) * 100})
        
        missing_data = missing_data[missing_data['Missing_Count'] > 0].sort_values('Missing_Count', ascending=False)        
        missing_txt.write_text(missing_data.to_string(), encoding="utf-8")
        
        # Identificación de valores no numéricos en columnas numéricas
        buf = []                     
        columnas_numericas = [col for col in df.columns 
                            if col not in ['datetime', 'mixed_type_col']]

        valores_invalidos_encontrados = {}
        for col in columnas_numericas:
            # Identificar valores no númericos en columnas numéricas
            temp = pd.to_numeric(df[col], errors='coerce')
            valores_problema = df[col][temp.isna() & df[col].notna()]
            
            if len(valores_problema) > 0:
                valores_unicos = valores_problema.unique()
                valores_invalidos_encontrados[col] = len(valores_problema)
                buf.append(f"\n• {col}:")
                buf.append(f"   Total de valores no numéricos: {len(valores_problema)}")
                buf.append(f"   Valores únicos encontrados: {valores_unicos[:15]}")
                not_number_txt.write_text("\n".join(buf), encoding="utf-8")
                
        # Análisis de la columna 'mixed_type_col' (columna 'extra')
        buf = []
        buf.append(f"\nValores únicos en mixed_type_col: {df['mixed_type_col'].nunique()}")
        buf.append(f"Valores faltantes: {df['mixed_type_col'].isna().sum()}")
        buf.append(f"\nPrimeros 10 valores más frecuentes:")
        buf.append(df['mixed_type_col'].value_counts().head(10).to_string())
        mixed_type_txt.write_text("\n".join(buf), encoding="utf-8") 
                
        # Timestamps duplicados
        buf = []
        duplicados_dt = df[df.duplicated(subset=['datetime'], keep=False)]
        buf.append(f"\nTotal de filas con timestamps duplicados: {len(duplicados_dt)}")

        if len(duplicados_dt) > 0:
            buf.append("\nEjemplo de timestamps duplicados (primeras 6 filas):")
            buf.append(duplicados_dt.sort_values('datetime').head(6)[['datetime', 'temperature', 'humidity']].to_string())

        # Timestamps faltantes
        filas_sin_datetime = df['datetime'].isna().sum()
        buf.append(f"\nFilas sin datetime: {filas_sin_datetime}")

        # Verificar continuidad temporal
        df_temp = df[df['datetime'].notna()].copy()
        df_temp['datetime_parsed'] = pd.to_datetime(df_temp['datetime'], format='%m/%d/%Y %H:%M', errors='coerce')

        filas_fecha_invalida = df_temp['datetime_parsed'].isna().sum()
        buf.append(f"Filas con formato de fecha inválido: {filas_fecha_invalida}")

        buf.append(f"   - {filas_sin_datetime} filas sin timestamp (inútiles para series temporales)")
        buf.append(f"   - {filas_fecha_invalida} filas con fechas imparsables")
        buf.append(f"   - {len(duplicados_dt)} filas con timestamps duplicados") 
        datetime_txt.write_text("\n".join(buf), encoding="utf-8")
        
        buf = self.detectar_outliers_extremos(df, columnas_numericas)
        outliers_txt.write_text("\n".join(buf), encoding="utf-8")        
        #-----------------------------------------------------------------------------------------------------------------------
        # Figuras: histogramas y matriz de correlación             
                
        # Visualización de valores faltantes                
        if missing_data.shape[1] > 0:
            fig = plt.figure(figsize=(10, 6))
            plt.barh(missing_data['Columna'], missing_data['Missing_Percent'])
            plt.xlabel('Porcentaje de Valores Faltantes (%)')
            plt.title('Valores Faltantes por Columna')
            plt.tight_layout()
            plt.savefig(self.report_path / "fig01_missing.png", dpi=150)
            plt.close(fig)
                
        # Selecciona columnas candidatas (excluye timestamps comunes)
        excluir = {"datetime", "DateTime"}
        cols_candidatas = [c for c in df.columns if c not in excluir]

        # Convierte posibles strings numéricos a float (maneja coma decimal)
        def _to_numeric_series(s: pd.Series) -> pd.Series:
            if s.dtype == "object":
                # reemplaza coma decimal por punto
                s = s.astype(str).str.replace(",", ".", regex=False)
            return pd.to_numeric(s, errors="coerce")

        num_df = df[cols_candidatas].apply(_to_numeric_series)

        # Conserva únicamente numéricas y elimina columnas completamente vacías
        num_df = num_df.select_dtypes(include="number").dropna(axis=1, how="all")

        # Genera y guarda el grid de histogramas
        out_path = Path(self.report_path) / "distribucion_variables.png"
        if num_df.shape[1] > 0:
            cols = num_df.columns.tolist()
            k = len(cols)
            ncols = 3
            nrows = math.ceil(k / ncols)

            fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(18, 12))
            axes = np.array(axes).reshape(-1)  # aplanar por simplicidad

            for i, col in enumerate(cols):
                ax = axes[i]
                s = num_df[col].dropna()

                if s.empty:
                    ax.set_visible(False)
                    continue

                # Recorta colas para evitar que el hist se aplaste (1–99%)
                q1, q99 = s.quantile([0.01, 0.99])
                s_clip = s[(s >= q1) & (s <= q99)]

                # Bins por columna con regla Freedman–Diaconis (robusta a colas)
                # Fallback a 'auto' si queda un único bin
                edges = np.histogram_bin_edges(s_clip, bins="fd")
                if len(edges) < 3:
                    edges = np.histogram_bin_edges(s_clip, bins="auto")

                ax.hist(s_clip, bins=edges)
                ax.set_title(col, fontsize=10)
                ax.grid(False)

            # Oculta ejes sobrantes (si la grilla es mayor que k)
            for j in range(i + 1, nrows * ncols):
                axes[j].set_visible(False)

            fig.suptitle("Distribución de variables — power_tetouan_city_modified", y=1.02)
            plt.tight_layout()
            out_path = Path(self.report_path) / "fig02_distribucion_variables_fd.png"
            plt.savefig(out_path, dpi=150, bbox_inches="tight")
            plt.close(fig)            

        # --- Matriz de correlación si hay suficientes numéricas ---
        if num_df.shape[1] > 1:
            plt.figure(figsize=(12, 8))
            sns.heatmap(num_df.corr(), annot=True, fmt=".2f", cmap="coolwarm", cbar_kws={"shrink": .8})
            plt.tight_layout()
            plt.savefig(self.report_path / "fig03_correlation_matrix.png", dpi=150)
            plt.close()
    
    def detectar_outliers_extremos(self, df: pd.DataFrame, columns: list) -> list:
        """
        Detecta outliers extremos por columna usando rangos [min_val, max_val] pre-calculados.
        """
        df_mod = df.copy()
        rangos_calculados = {}
        buf = ["Detección de outliers extremos por columna numérica:"]
        buf.append("-"*80)
        
        for col in columns:
            # Convertir a numérico y calcular estadísticas del original
            valores_orig = pd.to_numeric(df_mod[col], errors='coerce')
            
            q1 = valores_orig.quantile(0.25)
            q3 = valores_orig.quantile(0.75)
            iqr = q3 - q1
            
            # Límites usando 3*IQR (muy permisivo, solo outliers extremos)
            lower_fence = q1 - 3 * iqr
            upper_fence = q3 + 3 * iqr
            
            # No permitir valores negativos donde no tiene sentido
            if lower_fence < 0:
                lower_fence = 0
            
            rangos_calculados[col] = (lower_fence, upper_fence)
            
            buf.append(f"• {col}:")
            buf.append(f"  Q1={q1:.2f}, Q3={q3:.2f}, IQR={iqr:.2f}")
            buf.append(f"  → Rango válido calculado: [{lower_fence:.2f}, {upper_fence:.2f}]")           
          
        # Identificación de outliers en el DataFrame
        buf.append("\nRevisión de outliers extremos en el DataFrame")
        buf.append("-"*80)
        
        for col in columns:
            if col in df_mod.columns:
                df_mod[col] = pd.to_numeric(df_mod[col], errors='coerce')

        outliers_detectados = {}
        outliers_ejemplos = {}

        buf.append("\nOutliers extremos detectados con rangos calculados:")        

        for col in columns:
            if col in rangos_calculados and col in df_mod.columns:
                min_val, max_val = rangos_calculados[col]
                outliers = ((df_mod[col] < min_val) | (df_mod[col] > max_val)) & df_mod[col].notna()
                num_outliers = outliers.sum()
                outliers_detectados[col] = num_outliers
                
                if num_outliers > 0:
                    ejemplos = df_mod.loc[outliers, col].dropna().head(3).values
                    outliers_ejemplos[col] = ejemplos
                    buf.append(f"  • {col}: {num_outliers} outliers")
                    buf.append(f"    Ejemplos de valores extremos: {ejemplos}")
                    buf.append("\n")

        total_outliers = sum(outliers_detectados.values())
        buf.append(f"\nTOTAL DE OUTLIERS EXTREMOS: {total_outliers}")

        return buf