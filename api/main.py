# main.py
# Aplicación FastAPI para servir modelos de predicción de consumo eléctrico
# ===========================================================================

from datetime import datetime
from typing import Optional

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

# Variable global para el predictor
predictor: Optional[ModelPredictor] = None

_MODEL_TYPE_ALIASES = {
    "VAR": "VAR",
    "RANDOMFOREST": "RandomForest",
    "RF": "RandomForest",
    "XGBOOST": "XGBoost",
    "XGB": "XGBoost",
}


def normalize_model_type_name(model_type: str) -> str:
    """Normaliza el nombre del modelo y valida opciones soportadas."""
    if not model_type:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": "InvalidModelType",
                "message": "model_type es obligatorio"
            }
        )

    normalized_key = model_type.replace(" ", "").upper()
    normalized_value = _MODEL_TYPE_ALIASES.get(normalized_key)

    if normalized_value is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": "InvalidModelType",
                "message": "Tipo de modelo invA�lido. Usa VAR, RandomForest o XGBoost."
            }
        )

    return normalized_value


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
                "description": "Realizar predicción de consumo eléctrico"
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
            models_available = predictor.get_available_models()
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
    description="Realiza una predicción de consumo eléctrico para la zona y modelo especificados",
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
            "description": "Modelo no encontrado",
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
    Realiza una predicción de consumo eléctrico.

    Este endpoint recibe las features de entrada y retorna la predicción
    del consumo eléctrico usando el modelo especificado.

    Parameters
    ----------
    request : PredictionRequest
        Solicitud con zona, tipo de modelo y features

    Returns
    -------
    PredictionResponse
        Predicción y metadatos del modelo usado

    Raises
    ------
    HTTPException
        - 400: Si las features son inválidas
        - 404: Si el modelo no existe
        - 500: Si hay un error interno durante la predicción

    Examples
    --------
    ```python
    import requests

    response = requests.post(
        "http://localhost:8000/predict",
        json={
            "zone": 1,
            "model_type": "RandomForest",
            "features": {
                "temperature": 23.5,
                "humidity": 65.2,
                "hour": 14,
                "dayofweek": 2,
                "month": 6,
                "zone_1_power_consumption_lag6": 25000.0
            }
        }
    )
    print(response.json())
    ```
    """
    try:
        # Ejecutar predicción
        active_predictor = ensure_predictor_ready()
        normalized_model_type = normalize_model_type_name(request.model_type)
        result = active_predictor.predict(
            zone=request.zone,
            model_type=normalized_model_type,
            features=request.features
        )

        prediction_timestamp = datetime.now()

        # Construir respuesta
        response = PredictionResponse(
            zone=request.zone,
            model_type=normalized_model_type,
            model_path=result.get('model_name', 'unknown'),  # Usar model_name de MLFlow
            prediction=result['prediction'],
            features_used=result['features_used'],
            timestamp=prediction_timestamp
        )

        try:
            prediction_logger = get_prediction_logger()
            prediction_logger.log_prediction(
                zone=request.zone,
                model_type=normalized_model_type,
                features=request.features,
                prediction=result['prediction'],
                timestamp=prediction_timestamp
            )
        except RuntimeError:
            logger.warning(
                "Sistema de monitoreo no inicializado: la predicciA3n no se registrA3."
            )
        except Exception as log_error:
            logger.error(f"No se pudo registrar la predicciA3n para monitoreo: {log_error}")

        logger.info(
            f"Predicción exitosa - Zona: {request.zone}, "
            f"Modelo: {normalized_model_type}, Predicción: {result['prediction']:.2f}"
        )

        return response

    except ModelNotFoundError as e:
        logger.error(f"Modelo no encontrado: {e}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error": "ModelNotFoundError",
                "message": "No se encontró el modelo especificado",
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
    description="Retorna la A�ltima informaciA3n disponible del monitoreo de drift para un modelo.",
    tags=["Monitoreo"]
)
async def drift_status(
    zone: int = Query(..., ge=1, le=3, description="Zona a monitorear"),
    model_type: str = Query(..., description="Tipo de modelo (VAR, RandomForest, XGBoost)")
):
    """Obtiene el estado del monitoreo de drift para una zona y modelo."""
    normalized_model = normalize_model_type_name(model_type)

    try:
        drift_monitor = get_drift_monitor()
    except RuntimeError:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error": "MonitoringUnavailable",
                "message": "El sistema de monitoreo no estA� inicializado. Intenta mA�s tarde."
            }
        )

    try:
        status_payload = drift_monitor.get_drift_status(
            zone=zone,
            model_type=normalized_model
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

    return DriftStatusResponse(**status_payload)


@app.post(
    "/monitoring/drift/check",
    response_model=DriftCheckResponse,
    summary="Ejecutar chequeo manual de drift",
    description="Fuerza la ejecuciA3n del pipeline de monitoreo de drift independientemente del intervalo automA�tico.",
    tags=["Monitoreo"]
)
async def manual_drift_check(
    zone: int = Query(..., ge=1, le=3, description="Zona a evaluar"),
    model_type: str = Query(..., description="Tipo de modelo (VAR, RandomForest, XGBoost)")
):
    """Ejecuta un chequeo manual de drift."""
    normalized_model = normalize_model_type_name(model_type)

    try:
        drift_monitor = get_drift_monitor()
    except RuntimeError:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error": "MonitoringUnavailable",
                "message": "El sistema de monitoreo no estA� inicializado. Intenta mA�s tarde."
            }
        )

    try:
        report = drift_monitor.check_drift(zone=zone, model_type=normalized_model)
    except Exception as exc:
        logger.error(f"Error ejecutando drift check: {exc}")
        return DriftCheckResponse(
            status="error",
            message=f"Error: {exc}",
            zone=zone,
            model_type=normalized_model
        )

    if report is None:
        return DriftCheckResponse(
            status="insufficient_data",
            message="Datos insuficientes para ejecutar el monitoreo",
            zone=zone,
            model_type=normalized_model
        )

    summary = report.get_summary()
    recommendations = report.get_recommendations()

    return DriftCheckResponse(
        status="success",
        message="Chequeo de drift completado",
        zone=zone,
        model_type=normalized_model,
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
