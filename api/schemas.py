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


class DriftStatusResponse(BaseModel):
    """
    Esquema de respuesta para el estado de drift monitoring.

    Attributes
    ----------
    zone : int
        Zona monitoreada
    model_type : str
        Tipo de modelo monitoreado
    needs_drift_check : bool
        Indica si es necesario realizar un chequeo de drift
    last_check_time : Optional[str]
        Timestamp del último chequeo de drift
    next_check_in_hours : float
        Horas hasta el próximo chequeo programado
    latest_report_summary : Optional[Dict[str, Any]]
        Resumen del último reporte de drift
    """

    zone: int = Field(..., description="Zona monitoreada")
    model_type: str = Field(..., description="Tipo de modelo monitoreado")
    needs_drift_check: bool = Field(..., description="Indica si necesita chequeo de drift")
    last_check_time: Optional[str] = Field(None, description="Último chequeo de drift")
    next_check_in_hours: float = Field(..., description="Horas hasta próximo chequeo")
    latest_report_summary: Optional[Dict[str, Any]] = Field(
        None, description="Resumen del último reporte"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "zone": 1,
                    "model_type": "RandomForest",
                    "needs_drift_check": False,
                    "last_check_time": "2025-01-15T14:00:00",
                    "next_check_in_hours": 4.5,
                    "latest_report_summary": {
                        "total_alerts": 2,
                        "requires_action": True,
                        "has_critical_alerts": False
                    }
                }
            ]
        }
    }


class DriftCheckResponse(BaseModel):
    """
    Respuesta para chequeos manuales de drift.

    Attributes
    ----------
    status : str
        Estado general del chequeo (success, insufficient_data, error)
    message : str
        Mensaje descriptivo del resultado
    zone : int
        Zona evaluada
    model_type : str
        Tipo de modelo evaluado
    summary : Optional[Dict[str, Any]]
        Resumen del reporte generado
    recommendations : Optional[List[str]]
        Recomendaciones accionables
    """

    status: str = Field(..., description="Estado del chequeo manual")
    message: str = Field(..., description="Detalles del resultado")
    zone: int = Field(..., ge=1, le=3, description="Zona evaluada")
    model_type: str = Field(..., description="Modelo evaluado")
    summary: Optional[Dict[str, Any]] = Field(
        None, description="Resumen del reporte de drift"
    )
    recommendations: Optional[List[str]] = Field(
        None, description="Lista de recomendaciones"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "status": "success",
                    "message": "Chequeo de drift completado",
                    "zone": 1,
                    "model_type": "RandomForest",
                    "summary": {
                        "total_alerts": 3,
                        "has_critical_alerts": False,
                        "has_high_alerts": True,
                        "requires_action": True,
                    },
                    "recommendations": [
                        "Schedule model retraining within 24-48 hours",
                        "Review feature engineering pipeline for potential issues",
                    ],
                }
            ]
        }
    }


class ActualValueRequest(BaseModel):
    """
    Esquema de solicitud para registrar valor real observado.

    Attributes
    ----------
    zone : int
        Zona de consumo (1, 2 o 3)
    actual_value : float
        Valor real observado
    timestamp : Optional[datetime]
        Timestamp de la observación (por defecto: ahora)
    """

    zone: int = Field(..., ge=1, le=3, description="Zona de consumo (1, 2 o 3)")
    actual_value: float = Field(..., description="Valor real observado")
    timestamp: Optional[datetime] = Field(
        None, description="Timestamp de la observación (opcional)"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "zone": 1,
                    "actual_value": 25432.18,
                    "timestamp": "2025-01-15T14:30:00"
                }
            ]
        }
    }


class ActualValueResponse(BaseModel):
    """
    Respuesta al registrar valores reales observados.

    Attributes
    ----------
    status : str
        Resultado de la operaciA3n (success o error)
    message : str
        Mensaje detallando la operaciA3n
    zone : int
        Zona asociada
    actual_value : float
        Valor real registrado
    timestamp : datetime
        Momento asociado a la observaciA3n
    """

    status: str = Field(..., description="Resultado del registro")
    message: str = Field(..., description="Mensaje informativo")
    zone: int = Field(..., ge=1, le=3, description="Zona registrada")
    actual_value: float = Field(..., description="Valor real observado")
    timestamp: datetime = Field(..., description="Timestamp registrado")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "status": "success",
                    "message": "Valor real registrado exitosamente",
                    "zone": 1,
                    "actual_value": 25432.18,
                    "timestamp": "2025-01-15T14:30:00",
                }
            ]
        }
    }
