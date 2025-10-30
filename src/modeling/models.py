# models.py
# Clases de modelos para predicción de series temporales de consumo de energía
# ===========================

from abc import ABC, abstractmethod
from typing import Dict, Any
import pandas as pd
import numpy as np
import warnings

# Modelos estadísticos
from statsmodels.tsa.api import VAR

# Modelos de ML
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
import xgboost as xgb

# Deep Learning - Lazy import to avoid TensorFlow blocking on startup
# TensorFlow will only be imported when LSTMModel is actually used
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# Utilidades
from scipy.stats import randint, uniform

warnings.filterwarnings("ignore")


class BaseModel(ABC):
    """
    Interfaz base para todos los modelos de predicción.

    Define la estructura común que todos los modelos deben implementar
    siguiendo el patrón Template Method.
    """

    def __init__(self):
        """Inicializa el modelo base."""
        self.model = None
        self.is_trained = False

    @abstractmethod
    def train(self, train_df: pd.DataFrame, **kwargs) -> None:
        """
        Entrena el modelo con los datos proporcionados.

        Parameters
        ----------
        train_df : pd.DataFrame
            DataFrame con datos de entrenamiento
        **kwargs : dict
            Parámetros adicionales específicos del modelo
        """
        pass

    @abstractmethod
    def predict(self, train_df: pd.DataFrame, n_steps: int) -> pd.Series:
        """
        Genera predicciones para n_steps hacia adelante.

        Parameters
        ----------
        train_df : pd.DataFrame
            DataFrame histórico para contexto
        n_steps : int
            Número de pasos a predecir

        Returns
        -------
        pd.Series or pd.DataFrame
            Predicciones generadas
        """
        pass

    def get_params(self) -> Dict[str, Any]:
        """
        Retorna parámetros del modelo.

        Returns
        -------
        Dict[str, Any]
            Diccionario con parámetros del modelo
        """
        return {}


class VARModel(BaseModel):
    """
    Vector AutoRegression (VAR) para series temporales multivariadas.

    El modelo VAR captura las relaciones dinámicas entre múltiples series
    temporales, permitiendo que cada variable dependa de sus propios rezagos
    y de los rezagos de las demás variables.

    Attributes
    ----------
    model_fit : VARResultsWrapper
        Modelo VAR ajustado
    best_lag : int
        Orden de rezago óptimo encontrado mediante validación cruzada
    """

    def __init__(self):
        """Inicializa el modelo VAR."""
        super().__init__()
        self.model_fit = None
        self.best_lag = 0
        self.columns = None  # Guardar columnas con las que se entrenó
        print("• Instancia de VARModel creada")

    def train(
        self,
        train_df: pd.DataFrame,
        max_lags: int = 10,
        n_splits: int = 5
    ) -> None:
        """
        Entrena el modelo VAR con búsqueda automática del mejor rezago.

        Utiliza validación cruzada temporal para encontrar el orden de rezago
        que minimiza el error de predicción.

        Parameters
        ----------
        train_df : pd.DataFrame
            DataFrame con todas las series temporales a modelar.
            Cada columna representa una variable del sistema.
        max_lags : int, default=10
            Máximo número de rezagos a evaluar
        n_splits : int, default=5
            Número de divisiones para validación cruzada temporal
        """
        print(f"  • Entrenando modelo VAR (max_lags={max_lags})...")

        # Guardar las columnas con las que se entrena
        self.columns = train_df.columns.tolist()

        best_score = float('inf')
        best_lag_found = 0

        # Búsqueda del mejor rezago usando validación cruzada
        for p in range(1, max_lags + 1):
            scores = []
            tscv = TimeSeriesSplit(n_splits=n_splits)

            try:
                for train_index, test_index in tscv.split(train_df):
                    train_data = train_df.iloc[train_index]
                    test_data = train_df.iloc[test_index]

                    if len(train_data) < p:
                        continue

                    # Entrena y evalúa
                    model = VAR(train_data)
                    model_fit_cv = model.fit(maxlags=p)
                    lag_order = model_fit_cv.k_ar

                    forecast_input = train_data.values[-lag_order:]
                    predictions = model_fit_cv.forecast(y=forecast_input, steps=len(test_data))

                    error = mean_squared_error(test_data.values, predictions)
                    scores.append(error)

                if scores:
                    mean_score = np.mean(scores)
                    if mean_score < best_score:
                        best_score = mean_score
                        best_lag_found = p
            except Exception:
                continue

        print(f"    • Mejor rezago encontrado: {best_lag_found}")

        # Entrena modelo final con el mejor rezago
        final_model = VAR(train_df)
        self.model_fit = final_model.fit(maxlags=best_lag_found)
        self.best_lag = self.model_fit.k_ar
        self.is_trained = True

        print(f"    • Modelo VAR entrenado (lag={self.best_lag})")

    def predict(self, train_df: pd.DataFrame, n_steps: int) -> pd.DataFrame:
        """
        Genera predicciones multivariadas para todos los horizontes temporales.

        Parameters
        ----------
        train_df : pd.DataFrame
            DataFrame histórico con todas las variables del sistema
        n_steps : int
            Número de pasos futuros a predecir

        Returns
        -------
        pd.DataFrame
            DataFrame con predicciones para todas las variables.
            Las filas representan los pasos futuros y las columnas
            corresponden a cada variable del sistema.

        Raises
        ------
        Exception
            Si el modelo no ha sido entrenado previamente
        """
        if not self.is_trained or self.model_fit is None:
            raise Exception("El modelo VAR no ha sido entrenado. Llama a .train() primero.")

        # Filtrar solo las columnas con las que se entrenó el modelo
        if hasattr(self, 'columns') and self.columns is not None:
            train_df_filtered = train_df[self.columns]
        else:
            train_df_filtered = train_df

        forecast_input = train_df_filtered.values[-self.best_lag:]
        prediccion = self.model_fit.forecast(y=forecast_input, steps=n_steps)

        # Crea DataFrame con predicciones usando las columnas del modelo
        df_prediccion = pd.DataFrame(
            prediccion,
            index=pd.date_range(
                start=train_df.index[-1] + pd.Timedelta(minutes=10),
                periods=n_steps,
                freq='10min'
            ),
            columns=self.columns if self.columns is not None else train_df.columns
        )

        return df_prediccion

    def get_params(self) -> Dict[str, Any]:
        """Retorna parámetros del modelo VAR."""
        return {
            'model_type': 'VAR',
            'best_lag': self.best_lag,
            'is_trained': self.is_trained
        }


class RandomForestModel(BaseModel):
    """
    Random Forest para predicción univariada de series temporales.

    Utiliza un ensemble de árboles de decisión con características temporales
    avanzadas (lags, rolling means, features cíclicas) para capturar patrones
    complejos en los datos.

    Attributes
    ----------
    rf_model : RandomForestRegressor
        Modelo Random Forest entrenado
    target_col : str
        Columna objetivo a predecir
    weather_cols : list
        Columnas de variables exógenas (meteorológicas)
    """

    def __init__(self):
        """Inicializa el modelo Random Forest."""
        super().__init__()
        self.rf_model = None
        self.target_col = None
        self.weather_cols = None
        print("• Instancia de RandomForestModel creada")

    def train(
        self,
        train_df: pd.DataFrame,
        target_col: str,
        weather_cols: list,
        n_iter: int = 20,
        n_splits: int = 5,
        random_state: int = 42,
        param_distributions: dict = None
    ) -> None:
        """
        Entrena Random Forest con optimización de hiperparámetros.

        Utiliza RandomizedSearchCV con validación cruzada temporal para
        encontrar la mejor combinación de hiperparámetros.

        Parameters
        ----------
        train_df : pd.DataFrame
            DataFrame con datos de entrenamiento (debe tener DatetimeIndex)
        target_col : str
            Nombre de la columna objetivo a predecir
        weather_cols : list
            Lista de columnas con variables exógenas
        n_iter : int, default=20
            Número de iteraciones de búsqueda aleatoria
        n_splits : int, default=5
            Número de divisiones para validación cruzada temporal
        random_state : int, default=42
            Semilla aleatoria para reproducibilidad
        param_distributions : dict, optional
            Distribuciones de parámetros para RandomizedSearchCV.
            Si no se proporciona, usa valores por defecto.
        """
        from src.features.engineering import create_ml_features

        print(f"  • Entrenando Random Forest para: {target_col}...")

        self.target_col = target_col
        self.weather_cols = weather_cols

        # Modelo base
        rf_model_base = RandomForestRegressor(
            n_jobs=-1,
            random_state=random_state
        )

        # Espacio de búsqueda de hiperparámetros
        if param_distributions is not None:
            # Usa distribuciones provistas desde params.yaml
            param_distribuciones_rf = param_distributions
        else:
            # Usa valores por defecto
            param_distribuciones_rf = {
                'n_estimators': randint(50, 500),
                'max_depth': randint(3, 15),
                'max_features': uniform(0.3, 0.7),
                'min_samples_leaf': randint(2, 10)
            }

        # Prepara datos con features temporales avanzadas
        input_df = train_df[self.weather_cols + [self.target_col]]
        X_train_feat, y_train_feat = create_ml_features(input_df, self.target_col)

        # Búsqueda aleatoria con validación cruzada temporal
        tscv = TimeSeriesSplit(n_splits=n_splits)
        random_search = RandomizedSearchCV(
            estimator=rf_model_base,
            param_distributions=param_distribuciones_rf,
            n_iter=n_iter,
            cv=tscv,
            scoring='neg_root_mean_squared_error',
            verbose=0,
            random_state=random_state
        )

        random_search.fit(X_train_feat, y_train_feat)

        # Guarda el mejor modelo
        self.rf_model = random_search.best_estimator_
        self.is_trained = True

        print(f"    • RF entrenado. Mejor RMSE (CV): {-random_search.best_score_:.4f}")

    def predict(self, train_df: pd.DataFrame, n_steps: int) -> pd.Series:
        """
        Genera predicciones mediante estrategia recursiva.

        Utiliza las predicciones anteriores como entrada para generar
        predicciones futuras, simulando un escenario real donde solo
        se conoce el pasado.

        Parameters
        ----------
        train_df : pd.DataFrame
            DataFrame histórico con variables exógenas y objetivo
        n_steps : int
            Número de pasos futuros a predecir

        Returns
        -------
        pd.Series
            Serie temporal con las predicciones

        Raises
        ------
        Exception
            Si el modelo no ha sido entrenado previamente
        """
        if not self.is_trained or self.rf_model is None:
            raise Exception("El modelo Random Forest no ha sido entrenado. Llama a .train() primero.")

        from src.modeling.predict import predict_recursive_ml

        input_df_hist = train_df[self.weather_cols + [self.target_col]].copy()
        input_df_hist.index = train_df.index

        return predict_recursive_ml(
            model=self.rf_model,
            df_historico=input_df_hist,
            n_pasos=n_steps,
            variable_objetivo=self.target_col
        )

    def get_params(self) -> Dict[str, Any]:
        """Retorna parámetros del modelo Random Forest."""
        params = {
            'model_type': 'RandomForest',
            'target_col': self.target_col,
            'is_trained': self.is_trained
        }
        if self.rf_model is not None:
            params.update(self.rf_model.get_params())
        return params


class XGBoostModel(BaseModel):
    """
    XGBoost para predicción univariada de series temporales.

    Implementa gradient boosting con optimización de segunda orden,
    proporcionando alta precisión y velocidad de entrenamiento.

    Attributes
    ----------
    xgb_model : XGBRegressor
        Modelo XGBoost entrenado
    target_col : str
        Columna objetivo a predecir
    weather_cols : list
        Columnas de variables exógenas (meteorológicas)
    """

    def __init__(self):
        """Inicializa el modelo XGBoost."""
        super().__init__()
        self.xgb_model = None
        self.target_col = None
        self.weather_cols = None
        print("• Instancia de XGBoostModel creada")

    def train(
        self,
        train_df: pd.DataFrame,
        target_col: str,
        weather_cols: list,
        n_iter: int = 20,
        n_splits: int = 5,
        random_state: int = 42,
        param_distributions: dict = None
    ) -> None:
        """
        Entrena XGBoost con optimización de hiperparámetros.

        Utiliza RandomizedSearchCV con validación cruzada temporal para
        encontrar la mejor combinación de hiperparámetros que minimice
        el error de predicción.

        Parameters
        ----------
        train_df : pd.DataFrame
            DataFrame con datos de entrenamiento (debe tener DatetimeIndex)
        target_col : str
            Nombre de la columna objetivo a predecir
        weather_cols : list
            Lista de columnas con variables exógenas
        n_iter : int, default=20
            Número de iteraciones de búsqueda aleatoria
        n_splits : int, default=5
            Número de divisiones para validación cruzada temporal
        random_state : int, default=42
            Semilla aleatoria para reproducibilidad
        param_distributions : dict, optional
            Distribuciones de parámetros para RandomizedSearchCV.
            Si no se proporciona, usa valores por defecto.
        """
        from src.features.engineering import create_ml_features

        print(f"  • Entrenando XGBoost para: {target_col}...")

        self.target_col = target_col
        self.weather_cols = weather_cols

        # Modelo base
        xgb_model_base = xgb.XGBRegressor(
            objective='reg:squarederror',
            eval_metric='rmse',
            n_jobs=-1,
            random_state=random_state
        )

        # Espacio de búsqueda de hiperparámetros
        if param_distributions is not None:
            # Usa distribuciones provistas desde params.yaml
            param_distribuciones_xgb = param_distributions
        else:
            # Usa valores por defecto
            param_distribuciones_xgb = {
                'n_estimators': randint(100, 1000),
                'max_depth': randint(3, 10),
                'learning_rate': uniform(0.01, 0.2),
                'subsample': uniform(0.6, 0.4),
                'colsample_bytree': uniform(0.6, 0.4)
            }

        # Prepara datos con features temporales avanzadas
        input_df = train_df[self.weather_cols + [self.target_col]]
        X_train_feat, y_train_feat = create_ml_features(input_df, self.target_col)

        # Búsqueda aleatoria con validación cruzada temporal
        tscv = TimeSeriesSplit(n_splits=n_splits)
        random_search = RandomizedSearchCV(
            estimator=xgb_model_base,
            param_distributions=param_distribuciones_xgb,
            n_iter=n_iter,
            cv=tscv,
            scoring='neg_root_mean_squared_error',
            verbose=0,
            random_state=random_state
        )

        random_search.fit(X_train_feat, y_train_feat)

        # Guarda el mejor modelo
        self.xgb_model = random_search.best_estimator_
        self.is_trained = True

        print(f"    • XGBoost entrenado. Mejor RMSE (CV): {-random_search.best_score_:.4f}")

    def predict(self, train_df: pd.DataFrame, n_steps: int) -> pd.Series:
        """
        Genera predicciones mediante estrategia recursiva.

        Parameters
        ----------
        train_df : pd.DataFrame
            DataFrame histórico con variables exógenas y objetivo
        n_steps : int
            Número de pasos futuros a predecir

        Returns
        -------
        pd.Series
            Serie temporal con las predicciones

        Raises
        ------
        Exception
            Si el modelo no ha sido entrenado previamente
        """
        if not self.is_trained or self.xgb_model is None:
            raise Exception("El modelo XGBoost no ha sido entrenado. Llama a .train() primero.")

        from src.modeling.predict import predict_recursive_ml

        input_df_hist = train_df[self.weather_cols + [self.target_col]].copy()
        input_df_hist.index = train_df.index

        return predict_recursive_ml(
            model=self.xgb_model,
            df_historico=input_df_hist,
            n_pasos=n_steps,
            variable_objetivo=self.target_col
        )

    def get_params(self) -> Dict[str, Any]:
        """Retorna parámetros del modelo XGBoost."""
        params = {
            'model_type': 'XGBoost',
            'target_col': self.target_col,
            'is_trained': self.is_trained
        }
        if self.xgb_model is not None:
            params.update(self.xgb_model.get_params())
        return params


class LSTMModel(BaseModel):
    """
    Long Short-Term Memory (LSTM) para predicción de series temporales.

    Arquitectura de red neuronal recurrente que captura dependencias
    temporales de largo plazo mediante celdas de memoria especializadas.

    Attributes
    ----------
    model : Sequential
        Modelo LSTM de Keras
    scaler_x : MinMaxScaler
        Escalador para features de entrada
    scaler_y : MinMaxScaler
        Escalador para variable objetivo
    n_steps_in : int
        Ventana temporal de entrada (lookback)
    n_features : int
        Número total de features
    target_col : str
        Columna objetivo
    weather_cols : list
        Columnas de variables exógenas
    """

    def __init__(self):
        """Inicializa el modelo LSTM."""
        # Lazy import of TensorFlow to avoid blocking on startup
        import os
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
        os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense

        # Store references for use in other methods
        self.tf = tf
        self.Sequential = Sequential
        self.LSTM = LSTM
        self.Dense = Dense

        super().__init__()
        self.model = None
        self.scaler_x = MinMaxScaler(feature_range=(0, 1))
        self.scaler_y = MinMaxScaler(feature_range=(0, 1))

        self.n_steps_in = 144
        self.n_features = 0
        self.target_col = None
        self.feature_names = None
        self.weather_cols = None

        # Semillas para reproducibilidad
        tf.random.set_seed(42)
        np.random.seed(42)

        print("• Instancia de LSTMModel creada")

    def _create_sequences(self, data: np.ndarray):
        """
        Transforma datos en secuencias para LSTM.

        Crea ventanas deslizantes donde cada secuencia de entrada
        contiene n_steps_in pasos temporales y la salida es el
        siguiente valor del objetivo.

        Parameters
        ----------
        data : np.ndarray
            Datos escalados (features + target concatenados)

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            (X, y) donde X tiene forma (samples, n_steps_in, n_features)
            e y tiene forma (samples,)
        """
        X, y = [], []
        target_idx = -1  # El target es la última columna

        for i in range(len(data)):
            end_ix = i + self.n_steps_in
            if end_ix > len(data) - 1:
                break

            seq_x = data[i:end_ix, :]
            seq_y = data[end_ix, target_idx]

            X.append(seq_x)
            y.append(seq_y)

        return np.array(X), np.array(y)

    def train(
        self,
        train_df: pd.DataFrame,
        target_col: str,
        weather_cols: list,
        n_steps_in: int = 144,
        epochs: int = 20,
        batch_size: int = 32
    ) -> None:
        """
        Entrena el modelo LSTM con los datos proporcionados.

        El entrenamiento incluye normalización de datos, creación de
        secuencias temporales y optimización mediante backpropagation.

        Parameters
        ----------
        train_df : pd.DataFrame
            DataFrame con datos de entrenamiento (debe tener DatetimeIndex)
        target_col : str
            Nombre de la columna objetivo a predecir
        weather_cols : list
            Lista de columnas con variables exógenas
        n_steps_in : int, default=144
            Ventana temporal de entrada (24 horas con intervalos de 10 min)
        epochs : int, default=20
            Número de épocas de entrenamiento
        batch_size : int, default=32
            Tamaño de batch para entrenamiento
        """
        print(f"  • Entrenando LSTM para: {target_col}...")

        self.target_col = target_col
        self.weather_cols = weather_cols
        self.n_steps_in = n_steps_in
        self.feature_names = self.weather_cols + [self.target_col]
        self.n_features = len(self.feature_names)

        # Separa y escala X e y
        df_train_subset = train_df[self.feature_names]
        data_x = df_train_subset[self.weather_cols]
        data_y = df_train_subset[[self.target_col]]

        scaled_x = self.scaler_x.fit_transform(data_x)
        scaled_y = self.scaler_y.fit_transform(data_y)

        # Combina datos escalados
        scaled_data = np.hstack((scaled_x, scaled_y))

        # Crea secuencias
        X_train, y_train = self._create_sequences(scaled_data)

        if X_train.shape[0] == 0:
            raise ValueError("No se pudieron crear secuencias.")

        # Define arquitectura (optimizada para GPU con cuDNN)
        self.model = self.Sequential()
        self.model.add(self.LSTM(
            50,
            input_shape=(self.n_steps_in, self.n_features)
        ))
        self.model.add(self.Dense(1))
        self.model.compile(optimizer='adam', loss='mean_squared_error')

        # Entrena
        print(f"    • Iniciando entrenamiento LSTM (Epochs: {epochs})...")
        self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
        self.is_trained = True

        print(f"    • LSTM entrenado para {target_col}")

    def predict(self, train_df: pd.DataFrame, n_steps: int) -> pd.Series:
        """
        Genera predicciones recursivas con LSTM.

        Utiliza las predicciones previas como entrada para generar
        predicciones futuras, manteniendo la coherencia temporal.

        Parameters
        ----------
        train_df : pd.DataFrame
            DataFrame histórico con variables exógenas y objetivo
        n_steps : int
            Número de pasos futuros a predecir

        Returns
        -------
        pd.Series
            Serie temporal con las predicciones en escala original

        Raises
        ------
        Exception
            Si el modelo no ha sido entrenado previamente
        """
        if not self.is_trained or self.model is None:
            raise Exception("El modelo LSTM no ha sido entrenado. Llama a .train() primero.")

        # Prepara historial
        history_df = train_df[self.feature_names].tail(self.n_steps_in)

        # Escala el historial
        history_x = history_df[self.weather_cols]
        history_y = history_df[[self.target_col]]

        scaled_history_x = self.scaler_x.transform(history_x)
        scaled_history_y = self.scaler_y.transform(history_y)

        current_batch = np.hstack((scaled_history_x, scaled_history_y))

        predictions_scaled = []

        # Bucle de predicción recursiva
        for _ in range(n_steps):
            input_window = current_batch.reshape((1, self.n_steps_in, self.n_features))

            # Predice (escalado)
            pred_scaled_y = self.model.predict(input_window, verbose=0)[0][0]
            predictions_scaled.append(pred_scaled_y)

            # Crea nueva fila para el siguiente paso
            new_row_scaled_x = current_batch[-1, :self.n_features-1]
            new_row_scaled = np.append(new_row_scaled_x, pred_scaled_y)
            new_row_scaled = new_row_scaled.reshape(1, self.n_features)

            # Desliza la ventana
            current_batch = np.append(current_batch, new_row_scaled, axis=0)
            current_batch = current_batch[-self.n_steps_in:, :]

        # Invierte el escalado
        predictions_scaled_array = np.array(predictions_scaled).reshape(-1, 1)
        final_predictions = self.scaler_y.inverse_transform(predictions_scaled_array)

        # Crea serie con fechas
        dates = pd.date_range(
            start=train_df.index[-1] + pd.Timedelta(minutes=10),
            periods=n_steps,
            freq='10T'
        )

        return pd.Series(
            final_predictions.flatten(),
            index=dates,
            name='predicciones_futuras_lstm'
        )

    def get_params(self) -> Dict[str, Any]:
        """Retorna parámetros del modelo LSTM."""
        return {
            'model_type': 'LSTM',
            'target_col': self.target_col,
            'n_steps_in': self.n_steps_in,
            'n_features': self.n_features,
            'is_trained': self.is_trained
        }
