# main.py
# Aplicación FastAPI para servir modelos de predicción de consumo eléctrico
# ===========================================================================

from datetime import datetime
from typing import Optional, Dict

from fastapi import FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import logging
from api.schemas import (
    PredictionRequest,
    PredictionResponse,
    ErrorResponse,
    HealthResponse,
    DriftStatusResponse,
    DriftCheckResponse,
    ActualValueRequest,
    ActualValueResponse,
)
from api.predictor import (
    ModelPredictor,
    ModelNotFoundError,
    InvalidFeaturesError
)
from api.drift_monitor import (
    initialize_monitoring,
    get_prediction_logger,
    get_drift_monitor,
)

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Variable global para el predictor y nombre estandar del modelo champion
predictor: Optional[ModelPredictor] = None
CHAMPION_MODEL_TYPE = "Champion"


def _infer_zone_from_request(model_name: Optional[str], features: Dict[str, float]) -> int:
    """
    Determina la zona asociada a una predicción sin depender de valores hardcodeados.

    Intenta primero extraerla del nombre del modelo (powerTetouan_*_zone_X_*). Si no
    existe, infiere la zona a partir de las features enviadas (ej: lag_zone_2_*).
    """
    model_name = (model_name or "").lower()
    zone_tokens = (1, 2, 3)

    for zone in zone_tokens:
        if f"zone_{zone}" in model_name:
            return zone

    zone_votes = {zone: 0 for zone in zone_tokens}
    for feature_name in features.keys():
        feature_name = feature_name.lower()
        for zone in zone_tokens:
            if f"zone_{zone}" in feature_name:
                zone_votes[zone] += 1

    # Elegir la zona con mayor presencia en las features o default a zona 1
    zone_with_votes = max(zone_votes.items(), key=lambda item: item[1])
    return zone_with_votes[0] if zone_with_votes[1] > 0 else 1

def ensure_predictor_ready() -> ModelPredictor:
    """Garantiza que el predictor de MLflow esté inicializado antes de usarlo."""
    if predictor is None:
        logger.warning("ModelPredictor no está disponible: uso de MLflow inhabilitado.")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error": "ModelPredictorUnavailable",
                "message": (
                    "El servicio de predicción no está disponible porque no pudo "
                    "conectarse con MLflow. Intenta nuevamente más tarde."
                )
            }
        )
    return predictor


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Contexto de ciclo de vida de la aplicación.
    Inicializa recursos al inicio y los limpia al finalizar.
    """
    # Startup: Inicializar el predictor
    global predictor
    try:
        logger.info("Inicializando ModelPredictor...")
        predictor = ModelPredictor(models_dir="models")
        logger.info("ModelPredictor inicializado exitosamente")
    except Exception as e:
        predictor = None
        logger.warning(
            "ModelPredictor no pudo inicializarse. La API funcionará en modo degradado "
            "hasta que MLflow esté disponible. "
            f"Detalle: {e}"
        )

    try:
        logger.info("Inicializando sistema de monitoreo de drift...")
        initialize_monitoring(reference_data_path=None)
        logger.info("Sistema de monitoreo de drift inicializado exitosamente")
    except Exception as e:
        logger.error(f"Error al inicializar el monitoreo de drift: {e}")
        raise

    yield

    # Shutdown: Limpiar recursos
    logger.info("Limpiando recursos...")
    if predictor:
        predictor.clear_cache()
    logger.info("Recursos eliminados de cache")


# Inicializar la aplicación FastAPI
app = FastAPI(
    title="API de Predicción de Consumo Eléctrico - Tetouan City",
    description="""
    API para realizar predicciones de consumo eléctrico en las tres zonas de Tetouan City
    utilizando modelos de Machine Learning entrenados (VAR, Random Forest, XGBoost).
    """,
    version="1.0.0",
    lifespan=lifespan
)

# Configurar CORS para permitir llamadas desde cualquier origen
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get(
    "/",
    summary="Información de la API",
    description="Retorna información básica",
    tags=["Información"]
)
async def root():
    """
    Endpoint raíz que proporciona información básica sobre la API.

    Returns
    -------
    dict
        Información sobre la API y enlaces útiles
    """
    return {
        "message": "API de Predicción de Consumo Eléctrico - Tetouan City",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "endpoints": {
            "predict": {
                "method": "POST",
                "path": "/predict",
                "description": "Realizar predicción de consumo eléctrico para una zona específica"
            },
            "health": {
                "method": "GET",
                "path": "/health",
                "description": "Verificar estado del servicio"
            }
        }
    }


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Health Check",
    description="Verifica el estado del servicio y lista los modelos disponibles",
    tags=["Health"]
)
async def health_check():
    """
    Verifica el estado del servicio y retorna información sobre modelos disponibles.

    Returns
    -------
    HealthResponse
        Estado del servicio y modelos disponibles

    Raises
    ------
    HTTPException
        Si hay un error al verificar el estado
    """
    try:
        # Intentar obtener modelos disponibles
        try:
            models_dict = predictor.get_available_models()
            # Convertir dict plano a ModelAvailability objects para validación Pydantic
            from api.schemas import ModelAvailability
            models_available = {
                model_name: ModelAvailability(**model_info)
                for model_name, model_info in models_dict.items()
            }
        except Exception as e:
            logger.warning(f"No se pudo listar modelos: {e}")
            models_available = {}

        service_status = "healthy" if predictor else "unhealthy"
        return HealthResponse(
            status=service_status,
            models_available=models_available
        )

    except Exception as e:
        logger.error(f"Error en health check: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Servicio no disponible: {str(e)}"
        )


@app.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Realizar Predicción",
    description="Realiza una predicción de consumo eléctrico usando el modelo champion de la zona especificada",
    tags=["Predicción"],
    responses={
        200: {
            "description": "Predicción exitosa",
            "model": PredictionResponse
        },
        400: {
            "description": "Solicitud inválida (features incorrectas)",
            "model": ErrorResponse
        },
        404: {
            "description": "Modelo champion no encontrado",
            "model": ErrorResponse
        },
        500: {
            "description": "Error interno del servidor",
            "model": ErrorResponse
        }
    }
)
async def predict(request: PredictionRequest):
    """
    Realiza una predicción de consumo eléctrico usando el modelo champion de la zona especificada.

    Este endpoint recibe la zona y las features de entrada, y retorna la predicción
    del consumo eléctrico usando el mejor modelo desplegado (champion) para esa zona.

    Parameters
    ----------
    request : PredictionRequest
        Solicitud con zona y features

    Returns
    -------
    PredictionResponse
        Predicción y metadatos del modelo champion usado

    Raises
    ------
    HTTPException
        - 400: Si las features son inválidas
        - 404: Si el modelo champion no está disponible
        - 500: Si hay un error interno durante la predicción

    Examples
    --------
    ```python
    import requests

    response = requests.post(
        "http://localhost:8000/predict",
        json={
            "zone": 1,
            "features": {
                "temperature": 23.5,
                "humidity": 65.2,
                "hour": 14,
                "dayofweek": 2,
                "month": 6,
                "lag_power_consumption_1_hora": 25000.0
            }
        }
    )
    print(response.json())
    ```
    """
    try:
        # Ejecutar predicción con modelo champion
        active_predictor = ensure_predictor_ready()
        result = active_predictor.predict_with_champion(
            features=request.features
        )

        prediction_timestamp = datetime.now()

        # Construir respuesta
        response = PredictionResponse(
            model_name=result.get('model_name', 'champion'),
            prediction=result['prediction'],
            features_used=result['features_used'],
            timestamp=prediction_timestamp
        )

        # Intentar registrar en el sistema de monitoreo
        try:
            prediction_logger = get_prediction_logger()

            prediction_logger.log_prediction(
                zone=request.zone,
                model_type=CHAMPION_MODEL_TYPE,
                features=request.features,
                prediction=result['prediction'],
                timestamp=prediction_timestamp
            )
        except RuntimeError:
            logger.warning(
                "Sistema de monitoreo no inicializado: la predicción no se registró."
            )
        except Exception as log_error:
            logger.error(f"No se pudo registrar la predicción para monitoreo: {log_error}")

        logger.info(
            f"Predicción exitosa con modelo champion zona {request.zone} - "
            f"Modelo: {result.get('model_name', 'champion')}, "
            f"Predicción: {result['prediction']:.2f}"
        )

        return response

    except ModelNotFoundError as e:
        logger.error(f"Modelo champion no encontrado: {e}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error": "ModelNotFoundError",
                "message": "No se encontró el modelo champion desplegado",
                "detail": str(e)
            }
        )

    except InvalidFeaturesError as e:
        logger.error(f"Features inválidas: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": "InvalidFeaturesError",
                "message": "Las features proporcionadas son inválidas",
                "detail": str(e)
            }
        )

    except ValueError as e:
        logger.error(f"Error de validación: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": "ValidationError",
                "message": "Error de validación en los parámetros",
                "detail": str(e)
            }
        )


@app.post(
    "/monitoring/actual",
    response_model=ActualValueResponse,
    summary="Registrar valor real observado",
    description="Guarda el valor real asociado a una predicciA3n para habilitar monitoreo de performance.",
    tags=["Monitoreo"]
)
async def log_actual_value(request: ActualValueRequest):
    """
    Registra valores reales observados para compararlos con las predicciones.

    Parameters
    ----------
    request : ActualValueRequest
        Zona, valor real y timestamp opcional
    """
    timestamp = request.timestamp or datetime.now()

    try:
        prediction_logger = get_prediction_logger()
    except RuntimeError:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error": "MonitoringUnavailable",
                "message": "El sistema de monitoreo no estA� inicializado. Intenta mA�s tarde."
            }
        )

    try:
        prediction_logger.log_actual_value(
            timestamp=timestamp,
            zone=request.zone,
            actual_value=request.actual_value
        )
    except Exception as exc:
        logger.error(f"Error al registrar valor real: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "ActualValueLoggingError",
                "message": "No se pudo registrar el valor real para monitoreo",
                "detail": str(exc)
            }
        )

    return ActualValueResponse(
        status="success",
        message="Valor real registrado exitosamente",
        zone=request.zone,
        actual_value=request.actual_value,
        timestamp=timestamp
    )


@app.get(
    "/monitoring/drift/status",
    response_model=DriftStatusResponse,
    summary="Consultar estado del monitoreo de drift",
    description="Retorna la informacion disponible del monitoreo de drift para el modelo champion.",
    tags=["Monitoreo"]
)
async def drift_status(
    zone: int = Query(..., ge=1, le=3, description="Zona a monitorear")
):
    """Obtiene el estado del monitoreo de drift para una zona usando el modelo champion."""

    try:
        drift_monitor = get_drift_monitor()
    except RuntimeError:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error": "MonitoringUnavailable",
                "message": "El sistema de monitoreo no esta inicializado. Intenta mas tarde."
            }
        )

    try:
        status_payload = drift_monitor.get_drift_status(
            zone=zone,
            model_type=CHAMPION_MODEL_TYPE
        )
    except Exception as exc:
        logger.error(f"No se pudo obtener el estado de drift: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "DriftStatusError",
                "message": "No se pudo obtener el estado de monitoreo de drift",
                "detail": str(exc)
            }
        )

    next_check = status_payload.get("next_check_in_hours") or 0
    status_payload["next_check_in_hours"] = max(0.0, float(next_check))
    status_payload["model_type"] = CHAMPION_MODEL_TYPE

    return DriftStatusResponse(**status_payload)


@app.post(
    "/monitoring/drift/check",
    response_model=DriftCheckResponse,
    summary="Ejecutar chequeo manual de drift",
    description="Fuerza la ejecucion del pipeline de monitoreo de drift para el modelo champion.",
    tags=["Monitoreo"]
)
async def manual_drift_check(
    zone: int = Query(..., ge=1, le=3, description="Zona a evaluar")
):
    """Ejecuta un chequeo manual de drift usando el modelo champion."""

    try:
        drift_monitor = get_drift_monitor()
    except RuntimeError:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error": "MonitoringUnavailable",
                "message": "El sistema de monitoreo no esta inicializado. Intenta mas tarde."
            }
        )

    try:
        report = drift_monitor.check_drift(zone=zone, model_type=CHAMPION_MODEL_TYPE)
    except Exception as exc:
        logger.error(f"Error ejecutando drift check: {exc}")
        return DriftCheckResponse(
            status="error",
            message=f"Error: {exc}",
            zone=zone,
            model_type=CHAMPION_MODEL_TYPE
        )

    if report is None:
        return DriftCheckResponse(
            status="insufficient_data",
            message="Datos insuficientes para ejecutar el monitoreo",
            zone=zone,
            model_type=CHAMPION_MODEL_TYPE
        )

    summary = report.get_summary()
    recommendations = report.get_recommendations()

    return DriftCheckResponse(
        status="success",
        message="Chequeo de drift completado",
        zone=zone,
        model_type=CHAMPION_MODEL_TYPE,
        summary=summary,
        recommendations=recommendations
    )


# Manejador global de excepciones
@app.exception_handler(Exception)
async def global_exception_handler(_request, exc):
    """
    Manejador global para excepciones no capturadas.

    Parameters
    ----------
    request : Request
        Objeto de solicitud FastAPI
    exc : Exception
        Excepción lanzada

    Returns
    -------
    JSONResponse
        Respuesta JSON con información del error
    """
    logger.error(f"Excepción no manejada: {exc}", exc_info=True)

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "InternalServerError",
            "message": "Ocurrió un error inesperado",
            "detail": str(exc)
        }
    )


if __name__ == "__main__":
    import uvicorn

    # Ejecutar el servidor
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Recarga automática en desarrollo
        log_level="info"
    )

