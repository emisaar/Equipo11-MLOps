# engineering.py
# Funciones para crear features avanzadas de ML para series temporales
# ===========================

import pandas as pd
from typing import Tuple

from src.config import PASOS_POR_HORA, PASOS_POR_DIA


def create_ml_features(df: pd.DataFrame, variable_objetivo: str) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Crea features de series de tiempo para modelos de ML.

    Esta función genera características temporales de alta frecuencia y lags específicos
    para predicción de series temporales con intervalos de 10 minutos.

    Features generadas:
    - hora: Hora del día (0-23)
    - minuto: Minuto dentro de la hora (0, 10, 20, ..., 50)
    - dia_de_semana: Día de la semana (0=Lunes, 6=Domingo)
    - dia_del_ano: Día del año (1-365/366)
    - lag_1_hora: Valor de la variable objetivo 1 hora atrás
    - lag_24_horas: Valor de la variable objetivo 24 horas atrás
    - rolling_mean_1_hora: Media móvil de 1 hora
    - rolling_mean_24_horas: Media móvil de 24 horas

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame con índice de tipo DatetimeIndex
    variable_objetivo : str
        Nombre de la columna objetivo para crear lags

    Returns
    -------
    Tuple[pd.DataFrame, pd.Series]
        (X, y) donde:
        - X: DataFrame con todas las features (sin la variable objetivo)
        - y: Serie con la variable objetivo

    Examples
    --------
    >>> df_with_index = df.set_index('datetime')
    >>> X, y = create_ml_features(df_with_index, 'zone_1_power_consumption')

    Notes
    -----
    - El DataFrame debe tener un DatetimeIndex
    - Se eliminan filas con NaN generadas por lags y rolling
    - Las constantes PASOS_POR_HORA y PASOS_POR_DIA se importan de config.py
    """
    df_copy = df.copy()

    # Características de Fecha/Hora (alta frecuencia)
    df_copy['hora'] = df_copy.index.hour
    df_copy['minuto'] = df_copy.index.minute
    df_copy['dia_de_semana'] = df_copy.index.dayofweek
    df_copy['dia_del_ano'] = df_copy.index.dayofyear

    # Rezagos y Ventanas Móviles (solo de la variable objetivo)
    # Lag de 1 hora atrás (6 pasos * 10 min = 60 min)
    df_copy[f'lag_{variable_objetivo}_1_hora'] = df_copy[variable_objetivo].shift(PASOS_POR_HORA)

    # Lag de 24 horas atrás (144 pasos * 10 min = 1440 min = 24 horas)
    df_copy[f'lag_{variable_objetivo}_24_horas'] = df_copy[variable_objetivo].shift(PASOS_POR_DIA)

    # Media móvil de 1 hora (ventana de 6 pasos, desplazada 1 paso para evitar data leakage)
    df_copy[f'rolling_mean_{variable_objetivo}_1_hora'] = (
        df_copy[variable_objetivo].shift(1).rolling(window=PASOS_POR_HORA).mean()
    )

    # Media móvil de 24 horas (ventana de 144 pasos, desplazada 1 paso)
    df_copy[f'rolling_mean_{variable_objetivo}_24_horas'] = (
        df_copy[variable_objetivo].shift(1).rolling(window=PASOS_POR_DIA).mean()
    )

    # Elimina NaN generados por lags y rolling
    df_copy = df_copy.dropna()

    # Separa X (features) e y (target)
    X = df_copy.drop(columns=[variable_objetivo])
    y = df_copy[variable_objetivo]

    return X, y
