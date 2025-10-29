# predict.py
# Módulo para ejecutar inferencia con modelos entrenados.
# ===========================

from pathlib import Path
from typing import Union, Dict, Any
import joblib
import pandas as pd
import numpy as np


def load_model(model_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Carga un bundle de modelo entrenado desde disco.

    Parameters
    ----------
    model_path : str or Path
        Ruta al archivo pickle del modelo.

    Returns
    -------
    dict
        Diccionario conteniendo 'model', 'features', y 'params'.
    """
    return joblib.load(model_path)


def predict_model(
    model_path: Union[str, Path],
    input_data: Union[pd.DataFrame, str, Path]
) -> np.ndarray:
    """
    Ejecuta inferencia del modelo sobre datos de entrada.

    Parameters
    ----------
    model_path : str or Path
        Ruta al archivo pickle del modelo entrenado.
    input_data : DataFrame, str, or Path
        Características de entrada como DataFrame o ruta a archivo parquet.

    Returns
    -------
    np.ndarray
        Predicciones del modelo.

    Examples
    --------
    >>> predictions = predict_model(
    ...     "models/power_model.pkl",
    ...     "data/processed/test.parquet"
    ... )
    """
    # Carga el bundle del modelo
    bundle = load_model(model_path)
    model = bundle["model"]
    features = bundle["features"]

    # Carga datos de entrada si es una ruta
    if isinstance(input_data, (str, Path)):
        df = pd.read_parquet(input_data)
    else:
        df = input_data.copy()

    # Selecciona y ordena características como las espera el modelo
    X = df[features]

    # Genera predicciones
    predictions = model.predict(X)

    return predictions


def predict_with_features(
    model_path: Union[str, Path],
    input_data: pd.DataFrame,
    return_dataframe: bool = False
) -> Union[np.ndarray, pd.DataFrame]:
    """
    Ejecuta inferencia del modelo y opcionalmente retorna DataFrame con predicciones.

    Parameters
    ----------
    model_path : str or Path
        Ruta al archivo pickle del modelo entrenado.
    input_data : DataFrame
        DataFrame con características de entrada.
    return_dataframe : bool, default=False
        Si es True, retorna DataFrame con características y predicciones.

    Returns
    -------
    np.ndarray or DataFrame
        Array de predicciones o DataFrame con características y predicciones.
    """
    predictions = predict_model(model_path, input_data)

    if return_dataframe:
        result_df = input_data.copy()
        result_df['prediction'] = predictions
        return result_df

    return predictions


# ===== Predicción Recursiva para Series Temporales =====

def predict_recursive_ml(
    model,
    df_historico: pd.DataFrame,
    n_pasos: int,
    variable_objetivo: str,
    warmup_steps: int = 200
) -> pd.Series:
    """
    Realiza predicción recursiva para modelos ML en series temporales.

    La predicción recursiva utiliza las predicciones anteriores como entrada
    para generar predicciones futuras, permitiendo forecasting de múltiples
    pasos hacia adelante.

    Parameters
    ----------
    model : sklearn estimator
        Modelo entrenado compatible con sklearn (debe tener método predict)
    df_historico : pd.DataFrame
        DataFrame histórico con DatetimeIndex.
        Debe contener la variable objetivo y todas las features exógenas.
    n_pasos : int
        Número de pasos futuros a predecir
    variable_objetivo : str
        Nombre de la columna objetivo
    warmup_steps : int, default=200
        Número de pasos históricos a usar para crear features.
        Debe ser suficiente para calcular lags y rolling features.

    Returns
    -------
    pd.Series
        Serie temporal con las predicciones.
        El índice son las fechas futuras con frecuencia de 10 minutos.

    Notes
    -----
    - El DataFrame debe tener un DatetimeIndex
    - La función crea features temporales automáticamente usando create_ml_features
    - Cada predicción se agrega al histórico para generar la siguiente

    Examples
    --------
    >>> from sklearn.ensemble import RandomForestRegressor
    >>> model = RandomForestRegressor()
    >>> # ... entrenar modelo ...
    >>> predictions = predict_recursive_ml(
    ...     model, df_train, n_pasos=10,
    ...     variable_objetivo='zone_1_power_consumption'
    ... )
    """
    from src.features import create_ml_features

    df_recursivo = df_historico.copy()
    predicciones = []

    for i in range(n_pasos):
        # Toma ventana reciente para crear features
        datos_para_features = df_recursivo.tail(warmup_steps)

        # Crea features ML (lags, rolling, temporales)
        X_nuevas, _ = create_ml_features(datos_para_features, variable_objetivo)
        X_a_predecir = X_nuevas.tail(1)

        # Genera predicción
        prediccion_actual = model.predict(X_a_predecir)[0]
        predicciones.append(prediccion_actual)

        # Calcula la siguiente fecha
        siguiente_fecha = df_recursivo.index[-1] + pd.Timedelta(minutes=10)

        # Crea nueva fila con la predicción
        # Mantiene valores exógenos constantes (asume última observación)
        nueva_fila = {
            col: df_recursivo[col].iloc[-1]
            for col in df_recursivo.columns
            if col != variable_objetivo
        }
        nueva_fila[variable_objetivo] = prediccion_actual

        # Agrega la nueva fila al histórico
        df_nueva_fila = pd.DataFrame([nueva_fila], index=[siguiente_fecha])
        df_recursivo = pd.concat([df_recursivo, df_nueva_fila])

    # Crea serie con las predicciones
    fechas_prediccion = pd.date_range(
        start=df_historico.index[-1] + pd.Timedelta(minutes=10),
        periods=n_pasos,
        freq='10T'
    )

    return pd.Series(
        predicciones,
        index=fechas_prediccion,
        name='predicciones_futuras'
    )
