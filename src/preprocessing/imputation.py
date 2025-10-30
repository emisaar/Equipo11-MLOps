# imputation.py
# Funciones para imputación de valores faltantes usando media móvil
# ===========================

import pandas as pd


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
        Tamaño de ventana en número de filas
        (p.ej., 3 => anterior, actual y siguiente si center=True)
    center : bool, default=True
        True para ventana centrada; False para usar sólo pasado
        (útil en escenarios causales)
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
