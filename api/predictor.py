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

    Attributes
    ----------
    mlflow_uri : str
        URI del servidor MLFlow
    mlflow_client : MlflowClient
        Cliente de MLFlow para interactuar con el Registry
    models_cache : Dict[str, Any]
        Cache de modelos cargados en memoria
    """

    def __init__(self, models_dir: str = None, mlflow_uri: str = None):
        """
        Inicializa el predictor de modelos.

        Parameters
        ----------
        models_dir : str, optional
            Ignorado. Se mantiene por compatibilidad con código existente.
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

        logger.info(f"ModelPredictor inicializado")
        logger.info(f"MLFlow Server: {self.mlflow_uri}")
        logger.info(f"Modelos se cargarán desde S3 vía MLFlow Registry")

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
            logger.info(f"✓ Modelo cargado desde cache: {cache_key}")
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
                f"✓ Predicción exitosa - Zona: {zone}, Modelo: {model_type}, "
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
        Obtiene lista de modelos disponibles en MLFlow Registry.

        Returns
        -------
        Dict[str, Any]
            Diccionario con modelos disponibles y sus versiones
        """
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
                        'latest_version': rm.latest_versions[0].version if rm.latest_versions else None
                    }
                except Exception:
                    # No tiene alias champion
                    available[rm.name] = {
                        'champion_version': None,
                        'latest_version': rm.latest_versions[0].version if rm.latest_versions else None
                    }

            return available

        except Exception as e:
            logger.error(f"Error al obtener modelos disponibles: {str(e)}")
            return {}

    def clear_cache(self):
        """
        Limpia el cache de modelos cargados.
        """
        self.models_cache.clear()
        logger.info("Cache de modelos limpiado")