# predictor.py
# Lógica de carga y predicción de modelos desde MLFlow Registry
# ================================================================

import os
import logging
from typing import Dict, Any
from pathlib import Path

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import pandas as pd

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelNotFoundError(Exception):
    """Excepción cuando no se encuentra el modelo en MLFlow Registry."""
    pass


class InvalidFeaturesError(Exception):
    """Excepción cuando las features proporcionadas son inválidas."""
    pass


class ModelPredictor:
    """
    Gestor de carga y predicción de modelos desde MLFlow Registry.

    Esta clase descarga modelos champion desde MLFlow Registry (almacenados en S3),
    los cachea en memoria para mejorar el rendimiento, y ejecuta predicciones.
    Los modelos se cargan dinámicamente según la zona solicitada.

    Attributes
    ----------
    mlflow_uri : str
        URI del servidor MLFlow
    mlflow_client : MlflowClient
        Cliente de MLFlow para interactuar con el Registry
    models_cache : Dict[str, Any]
        Cache de modelos cargados en memoria (por zona)
    models_dir : Path
        Directorio donde se buscan modelos locales
    """

    def __init__(self, models_dir: str = "models", mlflow_uri: str = None):
        """
        Inicializa el predictor de modelos.

        Parameters
        ----------
        models_dir : str, optional
            Directorio donde se encuentran los modelos champion guardados físicamente.
        mlflow_uri : str, optional
            URI del servidor MLFlow. Si no se provee, se obtiene de la variable
            de entorno MLFLOW_TRACKING_URI (default: http://mlflow:5000)
        """
        # Configurar MLFlow
        self.mlflow_uri = mlflow_uri or os.getenv(
            "MLFLOW_TRACKING_URI",
            "http://mlflow:5000"
        )
        mlflow.set_tracking_uri(self.mlflow_uri)
        self.mlflow_client = MlflowClient()
        self.models_cache: Dict[str, Any] = {}
        self.models_dir = Path(models_dir)

        # Cache para modelos disponibles y flag para evitar reintentos de MLflow
        self._available_models_cache: Dict[str, Any] = {}
        self._mlflow_attempted = False

        logger.info(f"ModelPredictor inicializado")
        logger.info(f"MLFlow Server: {self.mlflow_uri}")
        logger.info(f"Directorio de modelos: {self.models_dir}")
        logger.info(
            "Predictor inicializado. Los modelos champion se cargarán "
            "dinámicamente desde MLflow/S3 según la zona solicitada."
        )

    def _get_registry_name(self, zone: int, model_type: str) -> str:
        """
        Construye el nombre del modelo en MLFlow Registry.

        Parameters
        ----------
        zone : int
            Zona de consumo (1, 2 o 3)
        model_type : str
            Tipo de modelo ('VAR', 'RandomForest', 'XGBoost')

        Returns
        -------
        str
            Nombre del modelo en Registry (ej: powerTetouan_RF_zone_1_power_consumption)

        Raises
        ------
        ValueError
            Si el tipo de modelo o zona son inválidos
        """
        # Normalizar el tipo de modelo
        model_type_map = {
            'VAR': 'VAR',
            'RandomForest': 'RF',
            'RF': 'RF',
            'XGBoost': 'XGB',
            'XGB': 'XGB'
        }

        model_prefix = model_type_map.get(model_type)
        if model_prefix is None:
            raise ValueError(
                f"Tipo de modelo inválido: {model_type}. "
                f"Use: VAR, RandomForest, o XGBoost"
            )

        # VAR es un modelo multivariado (no específico por zona)
        if model_prefix == 'VAR':
            return "powerTetouan_VAR"

        # Otros modelos son específicos por zona
        if zone not in [1, 2, 3]:
            raise ValueError(f"Zona debe ser 1, 2 o 3. Recibido: {zone}")

        return f"powerTetouan_{model_prefix}_zone_{zone}_power_consumption"

    def _load_zone_champion(self, zone: int) -> Any:
        """
        Carga el modelo champion específico de una zona.

        Intenta cargar en este orden:
        1. Desde cache en memoria
        2. Desde archivos locales (*_zone_X_*_champion.pkl)
        3. Desde MLflow Registry con alias 'champion'

        Parameters
        ----------
        zone : int
            Zona de consumo (1, 2 o 3)

        Returns
        -------
        Any
            Modelo sklearn cargado o None si no se encuentra
        """
        # Generar clave de cache
        cache_key = f"champion_zone_{zone}"

        # 1. Verificar cache en memoria
        if cache_key in self.models_cache:
            logger.info(f"• Modelo champion zona {zone} cargado desde cache")
            return self.models_cache[cache_key]

        # 2. Buscar en archivos locales
        if self.models_dir.exists():
            # Buscar archivos del patrón: *_zone_X_*_champion.pkl
            champion_files = list(self.models_dir.glob(f"*_zone_{zone}_*_champion.pkl"))

            if champion_files:
                try:
                    import joblib
                    champion_path = champion_files[0]  # Usar el primero encontrado
                    model = joblib.load(champion_path)

                    # Guardar en cache
                    self.models_cache[cache_key] = model

                    logger.info(f"• Modelo champion zona {zone} cargado desde: {champion_path}")
                    return model
                except Exception as e:
                    logger.warning(f"Error al cargar modelo champion local zona {zone}: {e}")

        # 3. Intentar cargar desde MLflow Registry
        try:
            # Buscar modelos registrados para esta zona con alias 'champion'
            registered_models = self.mlflow_client.search_registered_models(
                filter_string=f"name LIKE '%zone_{zone}%'"
            )

            for rm in registered_models:
                try:
                    # Verificar que existe versión con alias 'champion'
                    self.mlflow_client.get_model_version_by_alias(
                        name=rm.name,
                        alias="champion"
                    )

                    model_uri = f"models:/{rm.name}@champion"
                    model = mlflow.sklearn.load_model(model_uri)

                    # Guardar en cache
                    self.models_cache[cache_key] = model

                    logger.info(f"• Modelo champion zona {zone} cargado desde MLFlow: {rm.name}")
                    return model

                except Exception:
                    # Este modelo no tiene alias champion, continuar
                    continue

            logger.warning(f"No se encontró modelo champion para zona {zone} en MLFlow Registry")

        except Exception as e:
            logger.error(f"Error al buscar modelo champion zona {zone} en MLFlow: {e}")

        return None

    def _transform_features_for_zone(
        self,
        features: Dict[str, float],
        zone: int
    ) -> Dict[str, float]:
        """
        Transforma features genéricas agregando el prefijo de zona.

        Las features que requieren transformación son:
        - lag_power_consumption_* → lag_zone_X_power_consumption_*
        - rolling_mean_power_consumption_* → rolling_mean_zone_X_power_consumption_*

        Features estáticas (temperature, humidity, etc.) se mantienen sin cambios.

        Parameters
        ----------
        features : Dict[str, float]
            Diccionario con features genéricas
        zone : int
            Zona de consumo (1, 2 o 3)

        Returns
        -------
        Dict[str, float]
            Diccionario con features transformadas
        """
        transformed = {}

        for feature_name, value in features.items():
            # Identificar features que necesitan prefijo de zona
            if feature_name.startswith("lag_") and "power_consumption" in feature_name:
                # lag_power_consumption_X → lag_zone_1_power_consumption_X
                new_name = feature_name.replace(
                    "lag_power_consumption",
                    f"lag_zone_{zone}_power_consumption"
                )
                transformed[new_name] = value

            elif feature_name.startswith("rolling_mean_") and "power_consumption" in feature_name:
                # rolling_mean_power_consumption_X → rolling_mean_zone_1_power_consumption_X
                new_name = feature_name.replace(
                    "rolling_mean_power_consumption",
                    f"rolling_mean_zone_{zone}_power_consumption"
                )
                transformed[new_name] = value

            else:
                # Features estáticas: mantener sin cambios
                transformed[feature_name] = value

        logger.debug(
            f"Features transformadas para zona {zone}: "
            f"{len(features)} → {len(transformed)} features"
        )

        return transformed

    def load_model(self, zone: int, model_type: str) -> Any:
        """
        Carga modelo champion desde MLFlow Registry.

        El modelo se descarga desde S3 a través de MLFlow y se cachea en memoria
        para evitar descargas repetidas.

        Parameters
        ----------
        zone : int
            Zona de consumo (1, 2 o 3)
        model_type : str
            Tipo de modelo ('VAR', 'RandomForest', 'XGBoost')

        Returns
        -------
        Any
            Modelo sklearn listo para predicción

        Raises
        ------
        ModelNotFoundError
            Si el modelo no existe en MLFlow Registry
        """
        # Generar clave única para el cache
        cache_key = f"{model_type}_zone_{zone}"

        # Si ya está en cache, retornarlo
        if cache_key in self.models_cache:
            logger.info(f"• Modelo cargado desde cache: {cache_key}")
            return self.models_cache[cache_key]

        # Construir nombre del modelo en Registry
        model_name = self._get_registry_name(zone, model_type)

        # URI con alias 'champion' para obtener el mejor modelo
        model_uri = f"models:/{model_name}@champion"

        try:
            logger.info(f"Descargando desde MLFlow Registry: {model_uri}")
            sklearn_model = mlflow.sklearn.load_model(model_uri)

            # Validar que es un modelo válido
            if not hasattr(sklearn_model, 'predict'):
                raise ValueError(
                    f"El objeto descargado no tiene método 'predict'. "
                    f"Tipo: {type(sklearn_model)}"
                )

            # Guardar en cache
            self.models_cache[cache_key] = sklearn_model
            logger.info(f"Modelo cargado exitosamente: {cache_key}")

            return sklearn_model

        except mlflow.exceptions.MlflowException as e:
            error_msg = (
                f"No se encontró el modelo en MLFlow Registry: {model_name}@champion. "
                f"Asegúrate de que el modelo esté registrado y tenga el alias 'champion'."
            )
            logger.error(f"{error_msg}\nDetalle: {str(e)}")
            raise ModelNotFoundError(error_msg) from e

        except Exception as e:
            logger.error(f"Error al cargar modelo desde MLFlow: {str(e)}")
            raise ModelNotFoundError(
                f"Error al cargar el modelo: {str(e)}"
            ) from e

    def predict(
        self,
        zone: int,
        model_type: str,
        features: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Ejecuta predicción con el modelo especificado.

        Parameters
        ----------
        zone : int
            Zona de consumo (1, 2 o 3)
        model_type : str
            Tipo de modelo ('VAR', 'RandomForest', 'XGBoost')
        features : Dict[str, float]
            Diccionario con las features de entrada (incluidos lags)

        Returns
        -------
        Dict[str, Any]
            Diccionario con la predicción y metadatos:
            - prediction: valor predicho
            - model_name: nombre del modelo usado
            - features_used: lista de features utilizadas

        Raises
        ------
        ModelNotFoundError
            Si el modelo no existe
        InvalidFeaturesError
            Si las features proporcionadas son inválidas
        """
        # Cargar el modelo
        sklearn_model = self.load_model(zone, model_type)

        # Preparar DataFrame con las features
        try:
            df = pd.DataFrame([features])

            # Verificar que no hay valores nulos
            if df.isnull().any().any():
                null_features = df.columns[df.isnull().any()].tolist()
                raise InvalidFeaturesError(
                    f"Features con valores nulos: {null_features}"
                )

            # Ejecutar predicción
            prediction = sklearn_model.predict(df)[0]

            # Preparar respuesta
            model_name = self._get_registry_name(zone, model_type)
            available_features = list(features.keys())

            result = {
                'prediction': float(prediction),
                'model_name': model_name,
                'features_used': available_features
            }

            logger.info(
                f"Predicción exitosa - Zona: {zone}, Modelo: {model_type}, "
                f"Predicción: {prediction:.2f}"
            )

            return result

        except KeyError as e:
            raise InvalidFeaturesError(
                f"Error al acceder a features: {str(e)}"
            ) from e
        except Exception as e:
            logger.error(f"Error durante la predicción: {str(e)}")
            raise

    def get_available_models(self) -> Dict[str, Any]:
        """
        Obtiene lista de modelos disponibles en MLFlow Registry (con caché).

        Usa caché para evitar reintentos continuos de conexión a MLflow.
        Solo intenta conectar una vez al inicio.

        Returns
        -------
        Dict[str, Any]
            Diccionario con modelos disponibles y sus versiones
        """
        # Si ya tenemos el caché, retornarlo
        if self._available_models_cache:
            return self._available_models_cache

        # Si ya intentamos y falló, retornar caché vacío
        if self._mlflow_attempted:
            return {}

        # Marcar que ya intentamos conectar a MLflow
        self._mlflow_attempted = True

        try:
            # Buscar modelos registrados que empiecen con 'powerTetouan_'
            registered_models = self.mlflow_client.search_registered_models(
                filter_string="name LIKE 'powerTetouan_%'"
            )

            available = {}
            for rm in registered_models:
                # Obtener alias 'champion' si existe
                try:
                    champion_version = self.mlflow_client.get_model_version_by_alias(
                        name=rm.name,
                        alias="champion"
                    )
                    available[rm.name] = {
                        'champion_version': champion_version.version,
                        'latest_version': rm.latest_versions[0].version if rm.latest_versions else None,
                        'source': 'mlflow'
                    }
                except Exception:
                    # No tiene alias champion
                    available[rm.name] = {
                        'champion_version': None,
                        'latest_version': rm.latest_versions[0].version if rm.latest_versions else None,
                        'source': 'mlflow'
                    }

            # Guardar en caché
            self._available_models_cache = available
            return available

        except Exception as e:
            logger.error(f"Error al obtener modelos disponibles: {str(e)}")
            return {}

    def predict_with_champion(
        self,
        zone: int,
        features: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Ejecuta predicción con el modelo champion de la zona especificada.

        Parameters
        ----------
        zone : int
            Zona de consumo (1, 2 o 3)
        features : Dict[str, float]
            Diccionario con las features de entrada (sin prefijo de zona).
            Las features específicas de zona (lags, rolling means) se transformarán
            automáticamente agregando el prefijo zone_X.

        Returns
        -------
        Dict[str, Any]
            Diccionario con la predicción y metadatos:
            - prediction: valor predicho
            - model_name: nombre del modelo champion usado
            - features_used: lista de features utilizadas

        Raises
        ------
        ModelNotFoundError
            Si el modelo champion no está cargado
        InvalidFeaturesError
            Si las features proporcionadas son inválidas
        ValueError
            Si la zona es inválida
        """
        # Validar zona
        if zone not in [1, 2, 3]:
            raise ValueError(f"Zona debe ser 1, 2 o 3. Recibido: {zone}")

        # Cargar modelo champion específico de la zona
        champion_model = self._load_zone_champion(zone)

        if champion_model is None:
            raise ModelNotFoundError(
                f"Modelo champion de zona {zone} no está disponible. "
                "Verifica que el modelo esté desplegado correctamente."
            )

        try:
            # Transformar features: agregar prefijo de zona a features específicas
            transformed_features = self._transform_features_for_zone(features, zone)
            df = pd.DataFrame([transformed_features])

            # Verificar que no hay valores nulos
            if df.isnull().any().any():
                null_features = df.columns[df.isnull().any()].tolist()
                raise InvalidFeaturesError(
                    f"Features con valores nulos: {null_features}"
                )

            # Ejecutar predicción
            prediction = champion_model.predict(df)[0]

            # Preparar respuesta (usar features originales para la respuesta)
            original_features = list(features.keys())
            model_name = f"champion_zone_{zone}"

            result = {
                'prediction': float(prediction),
                'model_name': model_name,
                'features_used': original_features
            }

            logger.info(
                f"Predicción exitosa con modelo champion zona {zone} - "
                f"Predicción: {prediction:.2f}"
            )

            return result

        except KeyError as e:
            raise InvalidFeaturesError(
                f"Error al acceder a features: {str(e)}"
            ) from e
        except Exception as e:
            logger.error(f"Error durante la predicción zona {zone}: {str(e)}")
            raise

    def clear_cache(self):
        """
        Limpia el cache de modelos cargados.
        """
        self.models_cache.clear()
        logger.info("Cache de modelos limpiado")