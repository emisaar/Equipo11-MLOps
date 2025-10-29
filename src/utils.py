import requests
import os
import json
import ast
from dotenv import load_dotenv
from datetime import datetime, timedelta

# Añadimos SQLAlchemy para MySQL
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

class Util:
    # Motor de base de datos compartido
    _db_engine = None          
       
    @staticmethod
    def get_db_engine():
        """
        Inicializa y retorna un engine SQLAlchemy para MySQL local.
        Usa valores por defecto para conexión local si no existen en .env.
        """
        load_dotenv()
        if Util._db_engine:
            return Util._db_engine
        # Carga variables con valores por defecto local
        user = os.getenv("DB_USER")
        pwd  = os.getenv("DB_PASSWORD")
        host = os.getenv("DB_HOST")
        port = os.getenv("DB_PORT")
        name = os.getenv("DB_NAME")
        url = f"mysql+mysqlconnector://{user}:{pwd}@{host}:{port}/{name}"
        try:
            Util._db_engine = create_engine(url, pool_pre_ping=True)
        except SQLAlchemyError as e:
            raise EnvironmentError(f"No se pudo conectar a MySQL: {e}")
        return Util._db_engine
    
    @staticmethod
    def clean_name(c: str) -> str:
        """
        Normaliza nombres de columnas: minúsculas, espacios→guion_bajo.
        """
        return (c.strip().lower().replace("  ", " ").replace(" ", "_"))
    
class PowerConsumptionDAO:
    @staticmethod
    def fetch_data(page: int = 1, size: int = 10):
        """
        Recupera datos paginados de la tabla power_consumption.
        Retorna un dict con keys: items, total, page, size.
        """
        engine = Util.get_db_engine()
        offset = (page - 1) * size
        try:
            with engine.connect() as conn:
                total = conn.execute(text("SELECT COUNT(*) FROM power_consumption")).scalar_one()
                rows = conn.execute(
                    text("""
                        SELECT datetime, zone1, zone2, zone3, total_power, temperature, humidity
                        FROM power_consumption
                        ORDER BY datetime ASC
                        LIMIT :limit OFFSET :offset
                    """),
                    {"limit": size, "offset": offset}
                ).fetchall()
        except SQLAlchemyError as e:
            raise RuntimeError(f"Error al consultar la tabla power_consumption: {e}")
        
        items = [
            {
                "datetime": str(r[0]),
                "zone1": r[1],
                "zone2": r[2],
                "zone3": r[3],
                "total_power": r[4],
                "temperature": r[5],
                "humidity": r[6],
            } for r in rows
        ]
        
        return {"items": items, "total": total, "page": page, "size": size}


# ===== Funciones de Preprocesamiento Avanzado =====
# Manejo de series temporales

import pandas as pd
import numpy as np
import scipy.stats as stats
from typing import Optional


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normaliza los nombres de columnas del DataFrame:
    - Quita espacios al inicio y final
    - Convierte a minúsculas
    - Reemplaza uno o más espacios por guion bajo

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame con columnas a normalizar

    Returns
    -------
    pd.DataFrame
        DataFrame con columnas normalizadas (copia)

    Examples
    --------
    >>> df_clean = normalize_column_names(df)
    """
    df_clean = df.copy()
    df_clean.columns = (
        df_clean.columns
        .str.strip()                            # Quita espacios al inicio y al final
        .str.lower()                            # Convierte a minúsculas
        .str.replace(r'\s+', '_', regex=True)   # Reemplaza uno o más espacios por _
    )
    return df_clean


def generar_media_movil(
    s: pd.Series,
    window: int = 3,
    center: bool = True,
    min_periods: int = 1
) -> pd.Series:
    """
    Devuelve la serie de media móvil (rolling mean) de `s`.

    Parameters
    ----------
    s : pd.Series
        Serie de datos numéricos
    window : int, default=3
        Tamaño de ventana en número de filas (p.ej., 3 => anterior, actual y siguiente si center=True)
    center : bool, default=True
        True para ventana centrada; False para usar sólo pasado (útil en escenarios causales)
    min_periods : int, default=1
        Mínimo de observaciones requeridas dentro de la ventana

    Returns
    -------
    pd.Series
        Serie con la media móvil calculada

    Examples
    --------
    >>> ma = generar_media_movil(df['temperature'], window=144, center=True)
    """
    return s.rolling(window=window, center=center, min_periods=min_periods).mean()


def imputar_nans_con_media_movil(s: pd.Series, media_movil: pd.Series) -> pd.Series:
    """
    Reemplaza los NaN de `s` con los valores correspondientes de `media_movil`.
    Mantiene alineación por índice.

    Parameters
    ----------
    s : pd.Series
        Serie original con valores NaN
    media_movil : pd.Series
        Serie con la media móvil calculada

    Returns
    -------
    pd.Series
        Serie con NaN imputados

    Examples
    --------
    >>> ma = generar_media_movil(df['temperature'], window=144)
    >>> df['temperature'] = imputar_nans_con_media_movil(df['temperature'], ma)
    """
    return s.fillna(media_movil)


def outliers(
    df: pd.DataFrame,
    str_att_name: str,
    method: str,
    replace: bool = True,
    replace_metric: Optional[str] = None,
    ma_window: int = 3,
    ma_center: bool = True,
    ma_min_periods: int = 1,
    use_ewm: bool = False,
    ewm_span: int = 3,
    limit_side: str = "both"
) -> None:
    """
    Detecta y maneja outliers en una columna del DataFrame usando IQR o Z-score.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame a procesar (se modifica in-place)
    str_att_name : str
        Nombre de la columna a analizar
    method : str
        Método de detección: 'IQR' o 'zscore'
    replace : bool, default=True
        Si True, reemplaza outliers; si False, los elimina
    replace_metric : str, optional
        Métrica para reemplazo: 'mean', 'median', 'mode', 'movavg'
    ma_window : int, default=3
        Ventana para media móvil (si replace_metric='movavg')
    ma_center : bool, default=True
        Centrar ventana de media móvil
    ma_min_periods : int, default=1
        Periodos mínimos para media móvil
    use_ewm : bool, default=False
        Usar media móvil exponencial en lugar de simple
    ewm_span : int, default=3
        Span para EWM
    limit_side : str, default="both"
        Límites a aplicar: "lower", "upper", "both"

    Returns
    -------
    None
        Modifica el DataFrame in-place

    Examples
    --------
    >>> outliers(df, 'temperature', 'IQR', replace=True, replace_metric='movavg',
    ...          ma_window=144, limit_side='upper')
    """
    # Verifica si la columna existe en el DataFrame
    if str_att_name not in df.columns:
        raise ValueError(f"La columna '{str_att_name}' no existe en el DataFrame.")

    # Validar limit_side
    if limit_side not in ("lower", "upper", "both"):
        raise ValueError("limit_side inválido. Usa 'lower', 'upper' o 'both'.")

    # --- Detección de outliers ---
    if method == 'IQR':
        # Se calcula el valor inter-cuartil
        percentile_Q1 = df[str_att_name].quantile(0.25)
        percentile_Q3 = df[str_att_name].quantile(0.75)
        iqr = percentile_Q3 - percentile_Q1
        # Se establece limite superior e inferior
        upper_limit = percentile_Q3 + 1.5 * iqr
        lower_limit = percentile_Q1 - 1.5 * iqr
        # Máscaras por lado
        mask_lower = df[str_att_name] < lower_limit
        mask_upper = df[str_att_name] > upper_limit
        if limit_side == "lower":
            mask = mask_lower
        elif limit_side == "upper":
            mask = mask_upper
        else:  # "both"
            mask = mask_lower | mask_upper
        # Se determinan los valores atípicos en un nuevo dataframe
        outliers_df = df[mask]

    elif method == 'zscore':
        # Calcula el z-score (tolerante a NaN) y alinea con el índice
        z = stats.zscore(df[str_att_name], nan_policy='omit')
        z = pd.Series(z, index=df.index)
        threshold = 3
        # Máscaras por lado (nota: z puede ser NaN en posiciones con NaN en la serie original)
        if limit_side == "lower":
            mask = (z < -threshold)
        elif limit_side == "upper":
            mask = (z > threshold)
        else:  # "both"
            mask = (z.abs() > threshold)
        mask = mask.fillna(False)
        outliers_df = df[mask]
    else:
        print('Metodo no valido :  {} %'.format(method))
        return  # Importante: salir para evitar errores posteriores

    if method == 'IQR' or method == 'zscore':
        print('Total de registros del atributo {}:  {:.3f}'.format(
            str_att_name, (df[str_att_name].shape[0])))
        print('Total de outliers del atributo {}:  {:.3f}'.format(
            str_att_name, (outliers_df[str_att_name].shape[0])))
        print('Porcentaje del atributo {}:  {:.3f} %'.format(
            str_att_name, (outliers_df[str_att_name].shape[0]*100) / df[str_att_name].shape[0]))

        if outliers_df.empty:
            print('No se encontraron outliers para los límites seleccionados. No se realizaron cambios.')
            return

        if replace == True and replace_metric is not None:
            # Reemplaza los outliers por un valor especifico
            if replace_metric == 'mean':
                replace_value = df[str_att_name].mean()

            elif replace_metric == 'median':
                replace_value = df[str_att_name].median()

            elif replace_metric == 'mode':
                moda = df[str_att_name].mode(dropna=True)
                replace_value = moda.iloc[0] if len(moda) > 0 else df[str_att_name].median()

            elif replace_metric == 'movavg':
                # --- media móvil local por fila ---
                if use_ewm:
                    ma = df[str_att_name].ewm(span=ewm_span, adjust=False, min_periods=1).mean()
                    ma_label = f"ewm(span={ewm_span})"
                else:
                    ma = df[str_att_name].rolling(
                        window=ma_window, center=ma_center, min_periods=ma_min_periods).mean()
                    ma_label = f"rolling(window={ma_window}, center={ma_center})"
                # Fallback para posiciones donde la MA sea NaN (bordes/gaps)
                fallback = df[str_att_name].median()
                # Valores de MA solo en los índices outliers; rellenar NaN con fallback
                replace_value = ma.reindex(df.index).loc[outliers_df.index].fillna(fallback)
                print(f"Outliers del atributo {str_att_name} reemplazados por media móvil {ma_label}. (limit_side='{limit_side}')")

            else:
                print('replace_metric no válido. Usa: mean | median | mode | movavg')
                return

            # Se reemplazan los outliers con la métrica seleccionada
            df.loc[outliers_df.index, str_att_name] = replace_value

            # Evitar imprimir un objeto enorme si es movavg (Series)
            if replace_metric == 'movavg':
                print('Outliers del atributo {} reemplazados usando media móvil ({}). Total reemplazados: {}'.format(
                    str_att_name, ma_label, len(outliers_df.index)))
            else:
                print('Outliers del atributo {} reemplazados por:  {}'.format(str_att_name, replace_value))

        else:
            # Elimina outliers del dataframe
            df.drop(outliers_df.index.to_list(), inplace=True)
            print('Outliers del atributo {} eliminados'.format(str_att_name))