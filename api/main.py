# main.py
# Aplicación FastAPI para servir modelos de predicción de consumo eléctrico
# ===========================================================================

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import logging
from api.schemas import (
    PredictionRequest,
    PredictionResponse,
    ErrorResponse,
    HealthResponse
)
from api.predictor import (
    ModelPredictor,
    ModelNotFoundError,
    InvalidFeaturesError
)

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Variable global para el predictor
predictor: ModelPredictor = None


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
        logger.error(f"Error al inicializar ModelPredictor: {e}")
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

    ## Características

    * **Predicción multi-zona**: Soporte para zonas 1, 2 y 3
    * **Múltiples modelos**: VAR (multivariado), Random Forest y XGBoost
    * **Validación robusta**: Validación de entrada con Pydantic
    * **Manejo de errores**: Mensajes de error descriptivos y códigos HTTP apropiados
    * **Cache de modelos**: Los modelos se cargan una vez y se mantienen en memoria
    * **Health checks**: Endpoint para verificar el estado del servicio

    ## Modelos Disponibles

    Los modelos se encuentran en el directorio `models/`:

    * **VAR**: `models/var_model.pkl` (modelo multivariado)
    * **Random Forest**: `models/rf_zone_{zone}_power_consumption.pkl`
    * **XGBoost**: `models/xgb_zone_{zone}_power_consumption.pkl`

    Donde `{zone}` es 1, 2 o 3.
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
        models_available = predictor.get_available_models()

        return HealthResponse(
            status="healthy",
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
        result = predictor.predict(
            zone=request.zone,
            model_type=request.model_type,
            features=request.features
        )

        # Construir respuesta
        response = PredictionResponse(
            zone=request.zone,
            model_type=request.model_type,
            model_path=result['model_path'],
            prediction=result['prediction'],
            features_used=result['features_used']
        )

        logger.info(
            f"Predicción exitosa - Zona: {request.zone}, "
            f"Modelo: {request.model_type}, Predicción: {result['prediction']:.2f}"
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
