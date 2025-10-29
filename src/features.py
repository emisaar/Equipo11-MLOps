# features.py
# Módulo que define la clase PreprocessData para preprocesar datos de series temporales.
# Realiza ingeniería de rasgos, genera lags y divide en train/test.
# ===========================

# Librerías para anotaciones y dataclasses
from __future__ import annotations
from dataclasses import dataclass

# Librerías de utilidades
from pathlib import Path

# Librerías de manejo de datos
import pandas as pd
import numpy as np
from typing import List, Iterable, Dict, Tuple, Optional
import re


# Librerías de machine learning
from sklearn.model_selection import train_test_split

@dataclass
class PreprocessData:
    """Realiza ingeniería de rasgos, genera lags y divide en train/test.

    Attributes
    ----------
    input_parquet : Path
        Ruta al Parquet "loaded" (salida de LoadData).
    datetime_column : str
        Nombre de la columna datetime para rasgos y orden temporal.
    target : str
        Nombre de la variable objetivo a modelar.
    lags : list[int]
        Desplazamientos para crear lags del target.
    test_size : float
        Proporción del conjunto de test (0 < test_size < 1).
    random_state : int
        Semilla para reproducibilidad en el split.
    out_train : Path
        Ruta donde se guardará el parquet de entrenamiento.
    out_test : Path
        Ruta donde se guardará el parquet de prueba.
    out_cleaned : Path
        Ruta donde se guardará el parquet de limpio.
    """

    input_parquet: Path
    datetime_column: str
    target: str
    lags: List[int]
    test_size: float
    random_state: int
    out_train: Path
    out_test: Path
    out_cleaned: Path       
        
    def run(self) -> Tuple[Path, Path]:
        """Ejecuta el preprocesamiento y el split en train/test.

        Pasos:
        1) Carga datos intermedios (Parquet).
        2) Resuelve nombres reales de columnas (datetime y target).
        3) Elimina filas con ``target`` nulo.
        4) Agrega rasgos temporales y lags.
        5) Elimina NaN producidos por lags y quita la columna datetime original.
        6) Asegura que X sea 100% numérica y divide en train/test sin shuffle.
        7) Persiste ``train.parquet`` y ``test.parquet``.
        """
        # Asegura directorio de salida (p. ej. data/processed/)
        self.out_train.parent.mkdir(parents=True, exist_ok=True)

        # Carga los datos intermedios
        df = pd.read_parquet(self.input_parquet)                
            
        # Parseo robusto de datetime: intenta con parse_date_cols normalizados        
        if self.datetime_column != "":
            self.parse_date_cols = self.datetime_column                      
                
        # Columnas numéricas (excluye datetime y mixed_type_col si existen)
        columnas_numericas = [col for col in df.columns if col not in [self.datetime_column, 'mixed_type_col']]
        
        # Resolver columnas reales (robusto a espacios, guiones, mayúsculas)
        real_dt_col = self.resolve_column(self.datetime_column, df, friendly_name=self.datetime_column)
        real_target = self.resolve_column(self.target, df, friendly_name="target")
        
        # Se elimina la columna mixed_col ya que no es útil para el modelo
        if 'mixed_type_col' in df.columns:
            df = df.drop(columns=['mixed_type_col'])
        
        # Limpieza de valores no numéricos en columnas numéricas
        #valores_invalidos = ['error', 'invalid', '?', 'NAN', 'n/a', 'null', 'INVALID', 
        #                     'ERROR', ' NAN ', ' ? ', ' ERROR ', ' INVALID ', ' n/a ', ' null ']
        
        # Reemplaza valores inválidos por NaN y convierte a numérico
        #for col in columnas_numericas:
        #    df[col] = df[col].replace(valores_invalidos, np.nan)
        #    df[col] = pd.to_numeric(df[col], errors='coerce')

        # Coaccionar el target a numérico (convierte 'invalid' -> NaN)
        df[real_target] = pd.to_numeric(df[real_target], errors="coerce")
                
        # Filtrar nulos del target
        df = df.dropna(subset=[real_target])
        
        # Elimina registros con un porcentaje menor a 5% mínimo de valores que sean NaN        
        df.dropna(subset=pd.DataFrame(df.isnull().mean()*100, columns=['avg_nan'])
                                        .query('avg_nan < 5').index, inplace=True)
        
        # Se eliminan outliers y se reemplazan por la mediana
        df, reemplazos = self.sustituir_outliers_superior_mediana_df(df,
                                                         factor_iqr=2.0,        # mismo multiplicador que tu función original
                                                         coerce_numeric=True,   # convierte strings numéricos a float
                                                         inplace=False          # devuelve copia; pon True si quieres modificar en sitio
                                                         )               
        # Orden por datetime si está disponible
        if real_dt_col and real_dt_col in df.columns:
            df = df.sort_values(real_dt_col).reset_index(drop=True)
                 
        # Rasgos temporales + lags
        df = self.add_time_features(df, real_dt_col)
        if self.lags:
            df = self.add_lags(df, real_dt_col, real_target, self.lags)

        # Limpiar NaN generados por lags
        df = df.dropna().reset_index(drop=True)

        # IMPORTANTE: Establecer datetime como índice (NO eliminarlo)
        # Esto es necesario para create_ml_features que usa df.index.hour, etc.
        if real_dt_col in df.columns:
            df = df.set_index(real_dt_col)
        
        # Guardar una copia del DataFrame limpio para inspección
        # index=True preserva el DatetimeIndex
        df.to_parquet(self.out_cleaned, index=True)

        # Limpiar infinitos antes de separar X e y para mantener alineación
        df = df.replace([np.inf, -np.inf], np.nan).dropna()

        # Separar X/y (ya alineados porque limpiamos df completo)
        X = df.drop(columns=[real_target]).select_dtypes(include=[np.number])
        y = df[real_target]                

        # Split temporal (sin barajar para evitar fuga de información)
        X_train, X_test, y_train, y_test = train_test_split(X,
                                                            y,
                                                            test_size=self.test_size,
                                                            random_state=self.random_state,
                                                            shuffle=False)

        # Reconstruir DataFrames con target
        train_df = X_train.copy()
        train_df[real_target] = y_train
        test_df = X_test.copy()
        test_df[real_target] = y_test

        # Guardar resultados en archivos Parquet
        # IMPORTANTE: Guardar con index=True para preservar el DatetimeIndex
        train_df.to_parquet(self.out_train, index=True)
        test_df.to_parquet(self.out_test, index=True)
        
        return self.out_train, self.out_test
    
    def add_time_features(self, df: pd.DataFrame, dt_col: str) -> pd.DataFrame:
        """Agrega rasgos temporales derivados de una columna datetime.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame de entrada.
        dt_col : str
            Nombre de la columna datetime a partir de la cual se derivan rasgos.

        Returns
        -------
        pd.DataFrame
            Copia del DataFrame con columnas ``hour``, ``dayofweek`` y ``month``
            si ``dt_col`` existe; en caso contrario retorna el DataFrame original.
        """      
            
        # Sólo procede si la columna existe para evitar KeyError
        if dt_col in df.columns:
            # Si existe 'datetime' explícito, priorízalo como columna principal
            if not pd.api.types.is_datetime64_any_dtype(df[dt_col]):
                df[dt_col] = pd.to_datetime(df[dt_col], errors="coerce", infer_datetime_format=True)
                
            # Trabaja sobre una copia para no mutar el argumento original
            out = df.copy()
            # ``.dt`` expone accesores vectorizados para Series datetime64
            out["hour"] = out[dt_col].dt.hour
            out["dayofweek"] = out[dt_col].dt.dayofweek
            out["month"] = out[dt_col].dt.month                        
            
            return out
        # Si no hay columna datetime, retorna sin cambios
        return df


    def add_lags(self, df: pd.DataFrame, dt_col: str, target: str, lags: List[int]) -> pd.DataFrame:
        """Crea variables rezagadas (lags) del target para modelar dependencia temporal.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame de entrada.
        dt_col : str
            Columna de fecha/hora usada para ordenar antes de aplicar lags.
        target : str
            Nombre de la columna objetivo (la serie a predecir).
        lags : list[int]
            Lista de desplazamientos (en filas) para crear ``target_lag{L}``.

        Returns
        -------
        pd.DataFrame
            Copia del DataFrame con columnas de lags añadidas (una por valor L).
        """
        out = df.copy()

        # Ordena temporalmente si la columna datetime existe; evita fuga de info
        if dt_col in out.columns:
            out = out.sort_values(dt_col)

        # Genera cada columna lag mediante ``Series.shift(L)``
        for L in lags:
            out[f"{target}_lag{L}"] = out[target].shift(L)
        return out
    def _norm(self, s: str) -> str:
        """Normaliza un nombre para comparar de forma flexible:
        - a minúsculas
        - sin espacios
        - sólo caracteres [a-z0-9]
        """
        return re.sub(r"[^0-9a-z]+", "", str(s).strip().lower())

    def resolve_column(self, requested: str, df: pd.DataFrame, *, friendly_name: str) -> str:
        """
        Devuelve el nombre REAL de la columna en df que mejor coincide con 'requested'.
        - Primero intenta coincidencia exacta.
        - Luego por forma normalizada (_norm).
        - Finalmente sugiere columnas si no encuentra.
        """
        cols = list(df.columns)

        # exacta
        if requested in df.columns:
            return requested

        # normalizada
        req_n = self._norm(requested)
        matches = [c for c in cols if self._norm(c) == req_n]
        if matches:
            return matches[0]

        # sugerencias
        suggestions = [c for c in cols if req_n in self._norm(self,c)]
        hint = f"Sugerencias: {suggestions[:5]}" if suggestions else f"Columnas disponibles: {cols[:10]}..."
        raise ValueError(f"No se encontró la columna '{requested}' para {friendly_name}. {hint}")

    
    def sustituir_outliers_superior_mediana_df(self,
                                               df: pd.DataFrame,
                                               columnas: Optional[Iterable[str]] = None,
                                               *,
                                               factor_iqr: float = 2.0,
                                               coerce_numeric: bool = True,
                                               inplace: bool = False,
                                            ) -> Tuple[pd.DataFrame, Dict[str, int]]:
        """
        Reemplaza outliers superiores (> Q3 + factor*IQR) por la mediana de cada columna.
        Aplica a todas las columnas numéricas (o a 'columnas' si se proporcionan).

        Parámetros        
        -df : DataFrame original.
        -columnas : columnas a procesar (opcional). Si None, usa columnas numéricas.
        -factor_iqr : multiplicador de IQR para el límite superior (por defecto 2.0).
        -coerce_numeric : si True, intenta convertir a numérico con errors='coerce'.
        -inplace : si True, modifica df en sitio; si False, retorna una copia.

        Retorna        
        -(df_mod, reemplazos_por_col)
            df_mod : DataFrame modificado.
            reemplazos_por_col : dict con conteo de valores reemplazados por columna.
        """
        df_mod = df if inplace else df.copy()
        reemplazos: Dict[str, int] = {}

        # Determina columnas a procesar
        if columnas is None:
            cols = df_mod.select_dtypes(include="number").columns.tolist()
        else:
            cols = [c for c in columnas if c in df_mod.columns]

        for col in cols:
            s = df_mod[col]

            # Coerción a numérico si se solicita
            if coerce_numeric and not pd.api.types.is_numeric_dtype(s):
                s = pd.to_numeric(s, errors="coerce")

            # Si todo es NaN o constante, salta
            if s.dropna().empty:
                reemplazos[col] = 0
                continue

            q1 = s.quantile(0.25)
            q3 = s.quantile(0.75)
            iqr = q3 - q1

            # Si IQR es 0 (columna constante o casi), no hay outliers superiores
            if pd.isna(iqr) or iqr == 0:
                reemplazos[col] = 0
                continue

            limite_superior = q3 + factor_iqr * iqr
            mediana = s.median()

            mask = s.notna() & (s > limite_superior)
            n = int(mask.sum())
            if n > 0:
                s = s.where(~mask, mediana)

            df_mod[col] = s
            reemplazos[col] = n

        return df_mod, reemplazos


# ===== Feature Engineering para Modelos ML =====
# Funciones para crear features avanzadas

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