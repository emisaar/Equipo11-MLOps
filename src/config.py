# config.py
# Configuración central del proyecto MLOps - Tetouan City Power Consumption
# ===========================

from pathlib import Path
from typing import List, Tuple, Dict, Any

# ===== Rutas del proyecto =====
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
EDA_DIR = REPORTS_DIR / "eda"

# ===== Configuración de datos =====
DEFAULT_DATETIME_COLUMN = "DateTime"
DEFAULT_TARGET_COLUMN = "zone_1_power_consumption"

# Columnas del dataset
POWER_COLS: List[str] = [
    'zone_1_power_consumption',
    'zone_2_power_consumption',
    'zone_3_power_consumption'
]

WEATHER_COLS: List[str] = [
    'temperature',
    'humidity',
    'wind_speed',
    'general_diffuse_flows',
    'diffuse_flows'
]

ALL_NUMERIC_COLS = WEATHER_COLS + POWER_COLS

# Columnas para parseo de fechas
PARSE_DATE_COLS: List[str] = ["DateTime", "datetime"]

# Valores inválidos a manejar en preprocesamiento
INVALID_VALUES = [
    'error', 'invalid', '?', 'NAN', 'n/a', 'null',
    'INVALID', 'ERROR', ' NAN ', ' ? ', ' ERROR ',
    ' INVALID ', ' n/a ', ' null '
]

# ===== Preprocesamiento Avanzado =====
# Parámetros de media móvil (60 días de datos con intervalos de 10 min)
# Cada hora: 6 registros, Cada día: 144 registros, 60 días: 8640 registros
WINDOW = 8640
CENTER = True
MIN_PERIODS = 1

# Manejo de outliers
OUTLIER_METHOD = 'IQR'  # 'IQR' o 'zscore'
OUTLIER_FACTOR_IQR = 2.0
OUTLIER_LIMIT_SIDE = 'upper'  # 'lower', 'upper', 'both'
OUTLIER_REPLACE_METRIC = 'movavg'  # 'mean', 'median', 'mode', 'movavg'

# Transformaciones logarítmicas (columnas a transformar)
LOG_TRANSFORM_COLS: List[str] = [
    'general_diffuse_flows',
    'diffuse_flows',
    'wind_speed'
]

# ===== Feature Engineering =====
# Parámetros de alta frecuencia (intervalos de 10 minutos)
PASOS_POR_HORA = 6  # 60 min / 10 min
PASOS_POR_DIA = 24 * PASOS_POR_HORA  # 144 registros por día

# Lags básicos
DEFAULT_LAGS: Tuple[int, ...] = (1, 2, 24)

# ===== Configuración del split =====
DEFAULT_TEST_SIZE = 0.05
DEFAULT_RANDOM_STATE = 42

# ===== Configuración de Modelos =====

# VAR (Vector AutoRegression)
VAR_CONFIG: Dict[str, Any] = {
    "max_lags": 10,
    "n_splits": 5
}

# Random Forest
RF_CONFIG: Dict[str, Any] = {
    "n_iter": 10,
    "n_splits": 5,
    "random_state": DEFAULT_RANDOM_STATE
}

# XGBoost
XGB_CONFIG: Dict[str, Any] = {
    "n_iter": 10,
    "n_splits": 5,
    "random_state": DEFAULT_RANDOM_STATE
}

# LSTM
LSTM_CONFIG: Dict[str, Any] = {
    "n_steps_in": 144,  # 24 horas de lookback
    "epochs": 30,
    "batch_size": 32,
    "random_state": DEFAULT_RANDOM_STATE
}

# ===== Evaluación =====
N_STEPS_PREDICT = 10  # Número de pasos a predecir
WARMUP_STEPS = 200    # Pasos de calentamiento para predicción recursiva

# ===== Configuración de MLFlow =====
EXPERIMENT_NAME = "TetouanCityPower"
MLFLOW_TRACKING_URI = None  # Usa por defecto ./mlruns

# ===== Visualización =====
# Parámetros para descomposición estacional
SEASONAL_PERIOD = 13  # Periodo para descomposición STL/seasonal
