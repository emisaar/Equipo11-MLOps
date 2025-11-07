# schemas.py
# Modelos Pydantic para validación de entrada/salida de la API
# ==============================================================

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, field_validator
from datetime import datetime


class PredictionRequest(BaseModel):
    """
    Esquema de solicitud para predicción de consumo eléctrico.

    Attributes
    ----------
    zone : int
        Zona de consumo a predecir (1, 2 o 3)
    model_type : str
        Tipo de modelo a usar ('VAR', 'RandomForest' o 'XGBoost')
    features : Dict[str, float]
        Diccionario con las features requeridas por el modelo.
        Las features varían según el modelo pero típicamente incluyen:
        - Features temporales: hour, dayofweek, month
        - Features meteorológicas: temperature, humidity, wind_speed, etc.
        - Lags: zone_X_power_consumption_lag6, zone_X_power_consumption_lag144, etc.
    """

    zone: int = Field(
        ...,
        ge=1,
        le=3,
        description="Zona de consumo a predecir (1, 2 o 3)"
    )

    model_type: str = Field(
        ...,
        description="Tipo de modelo: 'VAR', 'RandomForest' o 'XGBoost'"
    )

    features: Dict[str, float] = Field(
        ...,
        description="Diccionario con las features requeridas por el modelo"
    )

    @field_validator('model_type')
    @classmethod
    def validate_model_type(cls, v: str) -> str:
        """Valida que el tipo de modelo sea válido."""
        allowed = {'VAR', 'RandomForest', 'XGBoost', 'RF', 'XGB'}
        v_normalized = v.strip()
        if v_normalized not in allowed:
            raise ValueError(
                f"model_type debe ser uno de: VAR, RandomForest, XGBoost. "
                f"Recibido: {v}"
            )
        return v_normalized

    @field_validator('features')
    @classmethod
    def validate_features(cls, v: Dict[str, float]) -> Dict[str, float]:
        """Valida que features no esté vacío."""
        if not v:
            raise ValueError("features no puede estar vacío")
        return v

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "zone": 1,
                    "model_type": "RandomForest",
                    "features": {
                        "temperature": 23.5,
                        "humidity": 65.2,
                        "wind_speed": 5.3,
                        "general_diffuse_flows": 120.5,
                        "diffuse_flows": 80.3,
                        "hora": 14,
                        "minuto": 30,
                        "dia_de_semana": 2,
                        "dia_del_ano": 150,
                        "lag_zone_1_power_consumption_1_hora": 25000.0,
                        "lag_zone_1_power_consumption_24_horas": 26500.0,
                        "rolling_mean_zone_1_power_consumption_1_hora": 25200.0,
                        "rolling_mean_zone_1_power_consumption_24_horas": 24800.0
                    }
                }
            ]
        }
    }


class PredictionResponse(BaseModel):
    """
    Esquema de respuesta para predicción de consumo eléctrico.

    Attributes
    ----------
    zone : int
        Zona de consumo predicha
    model_type : str
        Tipo de modelo usado
    model_path : str
        Ruta del modelo usado
    prediction : float
        Valor de predicción de consumo eléctrico (kW)
    timestamp : datetime
        Timestamp de la predicción
    features_used : List[str]
        Lista de features utilizadas por el modelo
    """

    zone: int = Field(..., description="Zona de consumo predicha")
    model_type: str = Field(..., description="Tipo de modelo usado")
    model_path: str = Field(..., description="Ruta del modelo usado")
    prediction: float = Field(..., description="Predicción de consumo eléctrico (kW)")
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="Timestamp de la predicción"
    )
    features_used: List[str] = Field(
        ...,
        description="Lista de features utilizadas por el modelo"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "zone": 1,
                    "model_type": "RandomForest",
                    "model_path": "models/rf_zone_1_power_consumption.pkl",
                    "prediction": 25432.18,
                    "timestamp": "2025-01-15T14:30:00",
                    "features_used": [
                        "temperature",
                        "humidity",
                        "wind_speed",
                        "hour",
                        "dayofweek",
                        "month",
                        "zone_1_power_consumption_lag6"
                    ]
                }
            ]
        }
    }


class ErrorResponse(BaseModel):
    """
    Esquema de respuesta para errores.

    Attributes
    ----------
    error : str
        Tipo de error
    message : str
        Mensaje descriptivo del error
    detail : Optional[str]
        Detalles adicionales del error
    """

    error: str = Field(..., description="Tipo de error")
    message: str = Field(..., description="Mensaje descriptivo del error")
    detail: Optional[str] = Field(None, description="Detalles adicionales del error")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "error": "ModelNotFoundError",
                    "message": "No se encontró el modelo especificado",
                    "detail": "El archivo models/rf_zone_1_power_consumption.pkl no existe"
                }
            ]
        }
    }


class HealthResponse(BaseModel):
    """
    Esquema de respuesta para el endpoint de health check.

    Attributes
    ----------
    status : str
        Estado del servicio ('healthy' o 'unhealthy')
    models_available : Dict[str, List[int]]
        Modelos disponibles por tipo y zona
    timestamp : datetime
        Timestamp del health check
    """

    status: str = Field(..., description="Estado del servicio")
    models_available: Dict[str, List[int]] = Field(
        ...,
        description="Modelos disponibles por tipo y zona"
    )
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="Timestamp del health check"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "status": "healthy",
                    "models_available": {
                        "VAR": [0],
                        "RandomForest": [1, 2, 3],
                        "XGBoost": [1, 2, 3]
                    },
                    "timestamp": "2025-01-15T14:30:00"
                }
            ]
        }
    }
