# predictor.py
# Lógica de carga y predicción de modelos
# ==========================================

from pathlib import Path
from typing import Dict, Any, List
import joblib
import pandas as pd
import logging

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelNotFoundError(Exception):
    """Excepción cuando no se encuentra el modelo solicitado."""
    pass


class InvalidFeaturesError(Exception):
    """Excepción cuando las features proporcionadas son inválidas."""
    pass


class ModelPredictor:
    """
    Gestor de carga y predicción de modelos de consumo eléctrico.

    Esta clase maneja la carga de modelos desde disco, mantiene un cache
    de modelos cargados para mejorar el rendimiento, y ejecuta predicciones.

    Attributes
    ----------
    models_dir : Path
        Directorio donde se almacenan los modelos entrenados
    models_cache : Dict[str, Any]
        Cache de modelos cargados para evitar lecturas repetidas de disco
    """

    def __init__(self, models_dir: str = "models"):
        """
        Inicializa el predictor de modelos.

        Parameters
        ----------
        models_dir : str, default="models"
            Directorio donde se almacenan los modelos
        """
        self.models_dir = Path(models_dir)
        self.models_cache: Dict[str, Any] = {}

        # Verificar que el directorio de modelos existe
        if not self.models_dir.exists():
            raise FileNotFoundError(
                f"El directorio de modelos no existe: {self.models_dir}"
            )

        logger.info(f"ModelPredictor inicializado con directorio: {self.models_dir}")

    def _get_model_path(self, zone: int, model_type: str) -> Path:
        """
        Construye la ruta al archivo del modelo basándose en zona y tipo.

        Parameters
        ----------
        zone : int
            Zona de consumo (1, 2 o 3)
        model_type : str
            Tipo de modelo ('VAR', 'RandomForest', 'XGBoost')

        Returns
        -------
        Path
            Ruta al archivo .pkl del modelo

        Raises
        ------
        ValueError
            Si la combinación de zona y tipo de modelo es inválida
        """
        # Normalizar el tipo de modelo
        model_type_map = {
            'VAR': 'var',
            'RandomForest': 'rf',
            'RF': 'rf',
            'XGBoost': 'xgb',
            'XGB': 'xgb'
        }

        model_prefix = model_type_map.get(model_type)
        if model_prefix is None:
            raise ValueError(
                f"Tipo de modelo inválido: {model_type}. "
                f"Use: VAR, RandomForest, o XGBoost"
            )

        # VAR es un modelo multivariado (no específico por zona)
        if model_prefix == 'var':
            return self.models_dir / "var_model.pkl"

        # Otros modelos son específicos por zona
        if zone not in [1, 2, 3]:
            raise ValueError(f"Zona debe ser 1, 2 o 3. Recibido: {zone}")

        return self.models_dir / f"{model_prefix}_zone_{zone}_power_consumption.pkl"

    def load_model(self, zone: int, model_type: str) -> Any:
        """
        Carga un modelo desde disco o desde el cache.

        Parameters
        ----------
        zone : int
            Zona de consumo (1, 2 o 3)
        model_type : str
            Tipo de modelo ('VAR', 'RandomForest', 'XGBoost')

        Returns
        -------
        Any
            Instancia del modelo entrenado (VARModel, RandomForestModel, XGBoostModel)

        Raises
        ------
        ModelNotFoundError
            Si el archivo del modelo no existe
        """
        # Generar clave única para el cache
        cache_key = f"{model_type}_zone_{zone}"

        # Si ya está en cache, retornarlo
        if cache_key in self.models_cache:
            logger.info(f"Modelo cargado desde cache: {cache_key}")
            return self.models_cache[cache_key]

        # Obtener ruta del modelo
        model_path = self._get_model_path(zone, model_type)

        # Verificar que el archivo existe
        if not model_path.exists():
            raise ModelNotFoundError(
                f"No se encontró el modelo en: {model_path}. "
            )

        # Cargar el modelo
        try:
            logger.info(f"Cargando modelo desde disco: {model_path}")
            model_obj = joblib.load(model_path)

            # Validar que es un objeto de modelo válido
            if not hasattr(model_obj, 'is_trained'):
                raise ValueError(
                    f"El objeto cargado no es un modelo válido. "
                    f"Tipo: {type(model_obj)}"
                )

            if not model_obj.is_trained:
                raise ValueError(
                    f"El modelo en {model_path} no ha sido entrenado"
                )

            # Guardar en cache
            self.models_cache[cache_key] = model_obj
            logger.info(f"Modelo cargado exitosamente: {cache_key}")

            return model_obj

        except Exception as e:
            logger.error(f"Error al cargar modelo desde {model_path}: {str(e)}")
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
            Diccionario con las features de entrada (ya calculadas, incluidos lags)

        Returns
        -------
        Dict[str, Any]
            Diccionario con la predicción y metadatos:
            - prediction: valor predicho
            - model_path: ruta del modelo usado
            - features_used: lista de features utilizadas

        Raises
        ------
        ModelNotFoundError
            Si el modelo no existe
        InvalidFeaturesError
            Si las features proporcionadas son inválidas
        """
        # Cargar el modelo
        model_obj = self.load_model(zone, model_type)

        # Extraer el modelo sklearn subyacente según el tipo
        if hasattr(model_obj, 'rf_model') and model_obj.rf_model is not None:
            sklearn_model = model_obj.rf_model
            model_name = "RandomForest"
        elif hasattr(model_obj, 'xgb_model') and model_obj.xgb_model is not None:
            sklearn_model = model_obj.xgb_model
            model_name = "XGBoost"
        elif hasattr(model_obj, 'model_fit') and model_obj.model_fit is not None:
            # VAR model - requiere manejo especial
            raise NotImplementedError(
                "La predicción directa con VAR aún no está implementada en la API. "
                "Use Random Forest o XGBoost."
            )
        else:
            raise ValueError(
                f"No se pudo extraer el modelo subyacente de {type(model_obj)}"
            )

        # Preparar DataFrame con las features
        try:
            df = pd.DataFrame([features])

            # Obtener las features esperadas por el modelo sklearn
            # El modelo es un pipeline, así que usamos todas las features del dict
            available_features = list(features.keys())

            # Verificar que no hay valores nulos
            if df.isnull().any().any():
                null_features = df.columns[df.isnull().any()].tolist()
                raise InvalidFeaturesError(
                    f"Features con valores nulos: {null_features}"
                )

            # Ejecutar predicción
            prediction = sklearn_model.predict(df)[0]

            # Preparar respuesta
            model_path = self._get_model_path(zone, model_type)

            result = {
                'prediction': float(prediction),
                'model_path': str(model_path),
                'features_used': available_features
            }

            logger.info(
                f"Predicción exitosa - Zona: {zone}, Modelo: {model_name}, "
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

    def get_available_models(self) -> Dict[str, List[int]]:
        """
        Obtiene lista de modelos disponibles en el directorio.

        Returns
        -------
        Dict[str, List[int]]
            Diccionario con tipos de modelo como claves y listas de zonas
            disponibles como valores. Por ejemplo:
            {'VAR': [0], 'RandomForest': [1, 2, 3], 'XGBoost': [1, 2, 3]}
        """
        available = {
            'VAR': [],
            'RandomForest': [],
            'XGBoost': []
        }

        # Verificar VAR (modelo multivariado)
        if (self.models_dir / "var_model.pkl").exists():
            available['VAR'] = [0]  # 0 indica que es multivariado

        # Verificar modelos por zona
        for zone in [1, 2, 3]:
            # Random Forest
            rf_path = self.models_dir / f"rf_zone_{zone}_power_consumption.pkl"
            if rf_path.exists():
                available['RandomForest'].append(zone)

            # XGBoost
            xgb_path = self.models_dir / f"xgb_zone_{zone}_power_consumption.pkl"
            if xgb_path.exists():
                available['XGBoost'].append(zone)

        return available

    def clear_cache(self):
        """
        Limpia el cache de modelos cargados.
        """
        self.models_cache.clear()
        logger.info("Cache de modelos eliminados de cache")
