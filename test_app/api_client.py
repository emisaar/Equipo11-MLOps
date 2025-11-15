"""
Cliente para interactuar con la API del modelo.

Proporciona métodos para hacer predicciones, registrar valores reales,
y monitorear el estado del drift.
"""

import requests
import logging
from typing import Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class APIClient:
    """
    Cliente para la API de predicción de consumo eléctrico.
    """

    def __init__(self, base_url: str = "http://localhost:8000"):
        """
        Inicializa el cliente de la API.

        Parameters
        ----------
        base_url : str
            URL base de la API
        """
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json'
        })

        logger.info(f"APIClient inicializado: {self.base_url}")

    def health_check(self) -> Dict[str, Any]:
        """
        Verifica el estado de la API.

        Returns
        -------
        Dict[str, Any]
            Respuesta del health check

        Raises
        ------
        requests.exceptions.RequestException
            Si hay error en la conexión
        """
        try:
            response = self.session.get(f"{self.base_url}/health", timeout=5)
            response.raise_for_status()
            data = response.json()
            logger.info(f"Health check exitoso: status={data.get('status')}")
            return data
        except requests.exceptions.RequestException as e:
            logger.error(f"Error en health check: {e}")
            raise

    def predict(self, features: Dict[str, float]) -> Dict[str, Any]:
        """
        Hace una predicción usando el modelo champion.

        Parameters
        ----------
        features : Dict[str, float]
            Features para la predicción

        Returns
        -------
        Dict[str, Any]
            Respuesta con la predicción

        Raises
        ------
        requests.exceptions.RequestException
            Si hay error en la petición
        """
        payload = {
            "features": features
        }

        try:
            response = self.session.post(
                f"{self.base_url}/predict",
                json=payload,
                timeout=10
            )
            response.raise_for_status()
            data = response.json()

            logger.debug(
                f"Prediccion exitosa: model={data.get('model_name')}, "
                f"prediction={data.get('prediction'):.2f}"
            )

            return data
        except requests.exceptions.RequestException as e:
            logger.error(f"Error en prediccion: {e}")
            if hasattr(e, 'response') and e.response is not None:
                try:
                    error_detail = e.response.json()
                    logger.error(f"Detalle del error: {error_detail}")
                except:
                    pass
            raise

    def log_actual_value(self, zone: int, actual_value: float,
                        timestamp: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Registra un valor real observado.

        Parameters
        ----------
        zone : int
            Zona del consumo
        actual_value : float
            Valor real observado
        timestamp : datetime, optional
            Timestamp de la observación

        Returns
        -------
        Dict[str, Any]
            Respuesta de la API
        """
        payload = {
            "zone": zone,
            "actual_value": actual_value
        }

        if timestamp is not None:
            payload["timestamp"] = timestamp.isoformat()

        try:
            response = self.session.post(
                f"{self.base_url}/monitoring/actual",
                json=payload,
                timeout=10
            )
            response.raise_for_status()
            data = response.json()

            logger.debug(f"Valor real registrado: zone={zone}, value={actual_value:.2f}")

            return data
        except requests.exceptions.RequestException as e:
            logger.error(f"Error registrando valor real: {e}")
            raise

    def get_drift_status(self, zone: int) -> Dict[str, Any]:
        """
        Obtiene el estado del monitoreo de drift.

        Parameters
        ----------
        zone : int
            Zona a monitorear

        Returns
        -------
        Dict[str, Any]
            Estado del drift
        """
        try:
            response = self.session.get(
                f"{self.base_url}/monitoring/drift/status",
                params={"zone": zone},
                timeout=10
            )
            response.raise_for_status()
            data = response.json()

            logger.debug(
                f"Estado de drift: zone={zone}, "
                f"needs_check={data.get('needs_drift_check')}"
            )

            return data
        except requests.exceptions.RequestException as e:
            logger.error(f"Error obteniendo estado de drift: {e}")
            raise

    def trigger_drift_check(self, zone: int) -> Dict[str, Any]:
        """
        Ejecuta un chequeo manual de drift.

        Parameters
        ----------
        zone : int
            Zona a evaluar

        Returns
        -------
        Dict[str, Any]
            Resultado del chequeo de drift
        """
        try:
            response = self.session.post(
                f"{self.base_url}/monitoring/drift/check",
                params={"zone": zone},
                timeout=30
            )
            response.raise_for_status()
            data = response.json()

            logger.info(
                f"Chequeo de drift: zone={zone}, status={data.get('status')}"
            )

            return data
        except requests.exceptions.RequestException as e:
            logger.error(f"Error en chequeo de drift: {e}")
            raise
