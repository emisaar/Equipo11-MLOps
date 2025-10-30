# outliers.py
# Funciones para detección y manejo de outliers
# ===========================

import pandas as pd
import numpy as np
import scipy.stats as stats
from typing import Optional


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
