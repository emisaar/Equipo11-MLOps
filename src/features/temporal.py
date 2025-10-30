# temporal.py
# Funciones auxiliares para agregar features temporales
# ===========================

import pandas as pd


def add_high_frequency_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega características temporales de alta frecuencia.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame con índice de tipo DatetimeIndex

    Returns
    -------
    pd.DataFrame
        DataFrame con columnas adicionales: hora, minuto, dia_de_semana, dia_del_ano

    Examples
    --------
    >>> df_with_features = add_high_frequency_features(df)
    """
    df_copy = df.copy()
    df_copy['hora'] = df_copy.index.hour
    df_copy['minuto'] = df_copy.index.minute
    df_copy['dia_de_semana'] = df_copy.index.dayofweek
    df_copy['dia_del_ano'] = df_copy.index.dayofyear
    return df_copy


def add_rolling_features(
    df: pd.DataFrame,
    target: str,
    pasos_por_hora: int = 6,
    pasos_por_dia: int = 144
) -> pd.DataFrame:
    """
    Agrega features de lags y rolling means para una variable objetivo.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame con índice de tipo DatetimeIndex
    target : str
        Nombre de la columna objetivo
    pasos_por_hora : int, default=6
        Número de pasos en una hora (para intervalos de 10 min = 6)
    pasos_por_dia : int, default=144
        Número de pasos en un día (24 * 6 = 144)

    Returns
    -------
    pd.DataFrame
        DataFrame con columnas de lags y rolling means

    Examples
    --------
    >>> df_with_lags = add_rolling_features(df, 'zone_1_power_consumption')
    """
    df_copy = df.copy()

    # Lags
    df_copy[f'lag_{target}_1_hora'] = df_copy[target].shift(pasos_por_hora)
    df_copy[f'lag_{target}_24_horas'] = df_copy[target].shift(pasos_por_dia)

    # Rolling means
    df_copy[f'rolling_mean_{target}_1_hora'] = (
        df_copy[target].shift(1).rolling(window=pasos_por_hora).mean()
    )
    df_copy[f'rolling_mean_{target}_24_horas'] = (
        df_copy[target].shift(1).rolling(window=pasos_por_dia).mean()
    )

    return df_copy
