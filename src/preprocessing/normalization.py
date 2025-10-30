# normalization.py
# Funciones para normalización de nombres de columnas
# ===========================

import pandas as pd


def clean_name(c: str) -> str:
    """
    Normaliza nombres de columnas: minúsculas, espacios→guion_bajo.

    Parameters
    ----------
    c : str
        Nombre de columna original

    Returns
    -------
    str
        Nombre de columna normalizado

    Examples
    --------
    >>> clean_name("  Temperature Value  ")
    'temperature_value'
    """
    return (c.strip().lower().replace("  ", " ").replace(" ", "_"))


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
