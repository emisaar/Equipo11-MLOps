# Equipo11-MLOps

## Tetouan Power Consumption — Predicción Multivariada con MLOps

Proyecto de predicción de consumo eléctrico en Tetouan City utilizando múltiples modelos de machine learning (VAR, Random Forest, XGBoost, LSTM) con pipeline completo de MLOps.

### Estructura del Proyecto
```
.
├─ data/                     # Datos del proyecto
│  ├─ raw/                   # CSV originales (versionados con DVC)
│  ├─ interim/               # Datos cargados en formato Parquet
│  ├─ processed/             # Datos limpios, preprocesados, train/test splits
│  └─ external/              # Datos externos de terceros
│
├─ src/                      # Lógica de negocio (importable, testeable)
│  ├─ __init__.py
│  ├─ config.py              # Configuración centralizada y constantes
│  │
│  ├─ data/                  # Gestión de datos
│  │  ├─ __init__.py
│  │  ├─ database.py         # Conexiones y operaciones de base de datos
│  │  └─ loaders.py          # Cargadores de datos (LoadData)
│  │
│  ├─ preprocessing/         # Preprocesamiento de datos
│  │  ├─ __init__.py
│  │  ├─ cleaner.py          # Limpieza de datos (DatasetCleaner)
│  │  ├─ outliers.py         # Detección y tratamiento de outliers
│  │  ├─ imputation.py       # Imputación de valores faltantes
│  │  └─ normalization.py    # Normalización y escalado
│  │
│  ├─ features/              # Ingeniería de características
│  │  ├─ __init__.py
│  │  ├─ engineering.py      # Feature engineering general
│  │  ├─ preprocessor.py     # PreprocessData principal
│  │  └─ temporal.py         # Features temporales y lags
│  │
│  ├─ modeling/              # Modelos de Machine Learning
│  │  ├─ __init__.py
│  │  ├─ models.py           # Implementación de VAR, RF, XGBoost, LSTM
│  │  ├─ train.py            # Orquestación de entrenamiento (ModelTrainer)
│  │  ├─ evaluate.py         # Evaluación y comparación (ModelEvaluator)
│  │  └─ predict.py          # Predicción recursiva para series temporales
│  │
│  ├─ monitoring/            # Sistema de monitoreo de drift
│  │  ├─ __init__.py
│  │  ├─ drift_detectors.py  # Detectores de drift (Statistical, TimeSeries, Performance)
│  │  ├─ alert_system.py     # Sistema de alertas multi-canal
│  │  └─ drift_pipeline.py   # Pipeline de orquestación
│  │
│  └─ visualization/         # Visualizaciones
│     ├─ __init__.py
│     ├─ eda.py              # Análisis exploratorio de datos
│     └─ plots.py            # Gráficos y visualizaciones
│
├─ pipeline/                 # Scripts de orquestación DVC
│  ├─ load_data.py           # Stage 1: Carga de datos
│  ├─ clean_data.py          # Stage 2: Limpieza
│  ├─ preprocess.py          # Stage 3: Preprocesamiento y splits
│  ├─ train.py               # Stage 4: Entrenamiento de modelos
│  ├─ evaluate.py            # Stage 5: Evaluación y comparación
│  └─ deploy.py              # Stage 6: Copia del champion versionado, sync a S3 y rebuild de imagen Docker
│
├─ models/                   # Modelos entrenados (.pkl)
├─ metrics/                  # Métricas de evaluación (JSON)
├─ reports/                  # Reportes y visualizaciones
│  └─ figures/               # Gráficas comparativas por zona
│
├─ notebooks/                # Análisis y experimentación
│  ├─ Fase1/                 # Notebooks de Fase 1
│  ├─ Fase2/                 # Notebooks de Fase 2
│  └─ IndividualAnalysis/    # Análisis individuales del equipo
│
├─ api/                      # API REST con FastAPI
│  ├─ main.py                # Aplicación principal con endpoints
│  ├─ schemas.py             # Modelos Pydantic para validación
│  ├─ predictor.py           # Lógica de predicción
│  └─ drift_monitor.py       # Monitoreo de drift en tiempo real
│
├─ docs/                     # Documentación adicional
│  ├─ DOCKER_DEPLOYMENT.md   # Guía completa de despliegue Docker
│  ├─ DOCKER_QUICKSTART.md   # Referencia rápida de Docker
│  ├─ DRIFT_MONITORING.md    # Sistema de monitoreo de drift
│  ├─ Drift_Monitoring_Implementation.md  # Paso a paso técnico
│  ├─ CUMPLIMIENTO_RUBRICA_MLOPS.md       # Evidencia de rúbrica
│  └─ postman/               # Colecciones y ambientes (parametrizados con champion_zone)
│
├─ scripts/                  # Scripts de automatización y despliegue
│  ├─ docker-build.sh        # Build de imagen Docker con tags semánticos/git
│  ├─ docker-run.sh          # Ejecución de contenedor local (volúmenes y puertos)
│  ├─ docker-push.sh         # Publicación a DockerHub
│  ├─ setup-docker-hub.sh    # Pipeline completo: copia champion, rebuild y push
│  └─ verify-deployment.sh   # Checklist automatizado post-despliegue (champion, DVC, endpoints)
│
├─ tests/                    # Tests automatizados
│  ├─ test_monitoring.py     # Tests del sistema de drift (20 tests)
│  ├─ test_preprocessing.py  # Tests de preprocesamiento
│  ├─ test_integration_pipeline.py  # Tests de integración
│  └─ test_api.py            # Smoke tests de la API FastAPI
│
├─ test_app/                 # Cliente CLI para pruebas end-to-end
│  ├─ main.py                # Orquesta predicciones, drift y reportes
│  ├─ data_generator.py      # Genera cargas sintéticas (con drift gradual)
│  └─ visualizer.py          # Reportes /plots de predicciones y errores
│
├─ examples/                 # Ejemplos de uso
│  └─ drift_monitoring_demo.py  # Demo del sistema de drift
│
├─ Dockerfile                # Configuración Docker multi-stage
├─ docker-compose.yml        # Orquestación de servicios (API + MLFlow)
├─ .dockerignore             # Exclusiones para build de Docker
├─ params.yaml               # Configuración de pipeline y hiperparámetros
├─ dvc.yaml                  # Pipeline DVC (5 stages)
├─ dvc.lock                  # Lock file de DVC
├─ requirements.txt          # Dependencias del proyecto
├─ environment.yml           # Entorno Conda (opcional)
├─ pyproject.toml            # Configuración del proyecto
├─ setup.cfg                 # Configuración de setup
├─ Makefile                  # Comandos automatizados
├─ LICENSE                   # Licencia del proyecto
├─ README.md                 # Documentación principal
└─ .gitignore
```

### Modelos Implementados

Este proyecto entrena y compara 3 tipos de modelos:

1. **VAR (Vector AutoRegression)**: Modelo estadístico multivariado para capturar interdependencias entre zonas
2. **Random Forest**: Modelo de ensemble con features temporales y meteorológicas
3. **XGBoost**: Gradient boosting optimizado con búsqueda de hiperparámetros

Cada modelo (excepto VAR) se entrena individualmente para cada una de las 3 zonas de consumo.

## Requisitos

El proyecto tiene dos archivos de dependencias:
- **`requirements.txt`**: Dependencias completas para desarrollo (incluye notebooks, DVC, testing)
- **`requirements-api.txt`**: Dependencias optimizadas para producción (solo API y modelos ML)

**Nota**: El `Dockerfile` utiliza `requirements-api.txt` para generar imágenes más livianas optimizadas para producción.

## Pipeline DVC

El proyecto utiliza DVC para orquestar un pipeline de 5 etapas:

```
load_data → clean_data → preprocess_data → train_models → evaluate_models
```

### Descripción de Etapas

1. **load_data**: Carga datos CSV y convierte a Parquet
   - Entrada: `data/raw/power_tetouan_city_modified.csv`
   - Salida: `data/interim/loaded.parquet`

2. **clean_data**: Limpieza, detección de outliers, imputación
   - Entrada: `data/interim/loaded.parquet`
   - Salida: `data/processed/cleaned.parquet`

3. **preprocess_data**: Feature engineering, creación de lags, train/test split
   - Entrada: `data/processed/cleaned.parquet`
   - Salida: `data/processed/train.parquet`, `data/processed/test.parquet`

4. **train_models**: Entrenamiento de VAR, RF, XGBoost, LSTM
   - Entrada: `data/processed/train.parquet`
   - Salida: `models/*.pkl` (10 modelos: 1 VAR + 3x3 modelos por zona)

5. **evaluate_models**: Evaluación, comparación y visualización
   - Entrada: `data/processed/test.parquet`, `models/`
   - Salida: `metrics/metrics.json`, `reports/figures/`

## Primeros Pasos

### 1. Configuración Inicial
```bash
# Crear entorno virtual
python -m venv .venv

# Activar entorno
# Linux/Mac:
source .venv/bin/activate
# Windows:
.venv\Scripts\activate

# Actualizar pip
python -m pip install --upgrade pip

# Instalar dependencias
pip install -r requirements.txt

# En caso de reinstalar librerias
#pip install --upgrade --force-reinstall -r requirements.txt
```

### 2. Inicializar DVC
```bash
# Inicializar Git y DVC (si no está hecho)
git init
dvc init

# Versionar datos crudos
dvc add data/raw/power_tetouan_city_modified.csv
git add data/raw/*.dvc .gitignore
git commit -m "Track Tetouan raw data with DVC"
```

### 3. Ejecutar Pipeline Completo
```bash
# Reproducir todo el pipeline
dvc repro

# O ejecutar etapas individuales
dvc repro load_data
dvc repro clean_data
dvc repro preprocess_data
dvc repro train_models
dvc repro evaluate_models
```

### 4. Visualizar Resultados
```bash
# Ver métricas de evaluación
dvc metrics show

# Ver el grafo del pipeline
dvc dag

# Inspeccionar métricas detalladas
cat metrics/metrics.json
```

## Configuración (params.yaml)

Todos los hiperparámetros y configuraciones se gestionan en `params.yaml`:

- **data**: Rutas de entrada/salida de datos
- **preprocessing**: Configuración de limpieza y outliers
- **split**: Proporción train/test y random state
- **models.to_train**: Lista de modelos a entrenar (usar snake_case: `random_forest`, `xgboost`, `var`)
- **models.var/random_forest/xgboost**: Hiperparámetros específicos por modelo
- **evaluation**: Configuración de evaluación y métricas

Para modificar hiperparámetros, edita `params.yaml` y ejecuta `dvc repro` para regenerar resultados.

## Almacenamiento Local DVC

```bash
# Configurar almacenamiento local
mkdir -p local-dvc-storage
dvc remote add -d localstorage local-dvc-storage

git add .dvc/config
git commit -m "Configure DVC remote storage"

# Subir datos y modelos al remoto
dvc push -r localstorage

# Descargar desde remoto (en otra máquina)
dvc pull -r localstorage
```

## Almacenamiento Remoto DVC

```bash
# Configurar almacenamiento Remoto S3
pip install awscli

aws configure --profile equipo11

# Configurar los siguientes parámetros
AWS Access Key ID [None]: [AWS_ SECRET_ID]
AWS Secret Access Key [None]: [AWS_SECRET_KE]
Default region name [None]: us-east-2
Default output format [None]: json

# dvc remote add -d s3_storage s3://{bucket_name}/{optional_folder}
dvc remote add -d team_remote s3://itesm-mna/202502-equipo11
dvc remote modify team_remote region us-east-2
dvc remote modify team_remote profile equipo11
cat .dvc/config
git add .
git commit -m "feat: Initializing DVC and setting up the remote storage in S3"
dvc push
```

## Uso Individual de Scripts

También puedes ejecutar scripts individuales sin DVC:

```bash
# IMPORTANTE: Instalar el proyecto en modo editable primero
pip install -e .

# Ejecutar manualmente cada etapa del pipeline
python pipeline/load_data.py
python pipeline/clean_data.py
python pipeline/preprocess.py
python pipeline/train.py
python pipeline/evaluate.py
```

> **Nota**: El comando `pip install -e .` es necesario porque los scripts en `pipeline/` importan módulos desde `src/` (ej: `from src.data.loaders import LoadData`). La instalación en modo editable registra el paquete en tu entorno Python, permitiendo que estas importaciones absolutas funcionen correctamente. El flag `-e` significa que los cambios en el código se reflejan inmediatamente sin reinstalar.

## Estructura de Métricas

El archivo `metrics/metrics.json` contiene métricas para cada zona y modelo:

```json
{
  "zone_1_power_consumption": {
    "VAR": {"RMSE": X.XX, "MAE": X.XX, "MAPE": X.XX},
    "RF": {"RMSE": X.XX, "MAE": X.XX, "MAPE": X.XX},
    "XGB": {"RMSE": X.XX, "MAE": X.XX, "MAPE": X.XX},
    "LSTM": {"RMSE": X.XX, "MAE": X.XX, "MAPE": X.XX}
  },
  ...
}
```

## Visualizaciones

Las gráficas comparativas se guardan en `reports/figures/`:
- `comparison_zone_1_power_consumption.png`
- `comparison_zone_2_power_consumption.png`
- `comparison_zone_3_power_consumption.png`

Cada gráfica muestra las predicciones de los 4 modelos vs. valores reales.

## API REST y Endpoints

El proyecto incluye una API REST desarrollada con FastAPI para exponer los modelos entrenados y monitorear drift en tiempo real.

### Características de la API

- **Predicción**: Endpoint POST `/predict` usando modelos champion versionados
- **Validación robusta**: Pydantic schemas con manejo de errores detallado
- **Carga inteligente**: Modelos desde MLFlow Registry (S3) o directorio local con cache en memoria
- **Documentación automática**: Swagger UI en `/docs` y OpenAPI schema en `/openapi.json`
- **Health checks**: Endpoint `/health` con información de versiones y estado de modelos
- **Monitoreo de drift**: Sistema en tiempo real con detección estadística, temporal y de performance
- **Alertas configurables**: Por severidad con recomendaciones automáticas

### Iniciar el Servidor API

```bash
# Instalar dependencias de la API
pip install -r requirements.txt

# Iniciar servidor en modo desarrollo
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

# Verificar estado del servicio
curl http://localhost:8000/health
```

### Modelos Disponibles en la API

Los modelos se cargan desde MLflow Registry y están versionados tanto en DVC como en MLflow:

| Tipo de Modelo | Nombre en MLflow Registry | Ruta MLflow Champion | Ruta MLflow Latest |
|----------------|---------------------------|---------------------|-------------------|
| Random Forest Zona 1 | `powerTetouan_RF_zone_1_power_consumption` | `models:/powerTetouan_RF_zone_1_power_consumption/1` | `models:/powerTetouan_RF_zone_1_power_consumption/2` |
| Random Forest Zona 2 | `powerTetouan_RF_zone_2_power_consumption` | `models:/powerTetouan_RF_zone_2_power_consumption/1` | `models:/powerTetouan_RF_zone_2_power_consumption/2` |
| Random Forest Zona 3 | `powerTetouan_RF_zone_3_power_consumption` | `models:/powerTetouan_RF_zone_3_power_consumption/1` | `models:/powerTetouan_RF_zone_3_power_consumption/2` |

**Versionado de Modelos:**
- **Champion Version**: Versión en producción (actualmente v1 para todas las zonas)
- **Latest Version**: Última versión entrenada (actualmente v2 para todas las zonas)
- **Ruta MLflow**: Formato `models:/<nombre_modelo>/<version>` para acceso programático desde MLflow **Registry**

**Nota**: Las versiones de DVC (hashes MD5) se encuentran en el archivo `dvc.lock`. Los modelos se sincronizan entre MLflow Registry (S3) y el directorio local `models/`

### Lista de Endpoints

**API Principal:**
- **GET** `/` - Información básica de la API
- **GET** `/health` - Health check
- **POST** `/predict` - Realizar predicción de consumo eléctrico usando modelo champion

**Monitoreo de Drift:**
- **POST** `/monitoring/actual` - Registrar valor real observado para comparación
- **GET** `/monitoring/drift/status` - Consultar estado del monitoreo de drift (query param: `zone`)
- **POST** `/monitoring/drift/check` - Ejecutar chequeo manual de drift (query param: `zone`)

### Endpoint POST `/predict` - Predicción

Realiza una predicción de consumo eléctrico usando el modelo champion desplegado para la zona especificada.

**Request (`PredictionRequest`):**
```json
{
  "zone": 1,
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
    "lag_power_consumption_1_hora": 25000.0,
    "lag_power_consumption_24_horas": 26500.0,
    "rolling_mean_power_consumption_1_hora": 25200.0,
    "rolling_mean_power_consumption_24_horas": 24800.0
  }
}
```

**Parámetros:**
- **zone** (`int`, requerido): Zona de consumo eléctrico (1, 2 o 3)
- **features** (`object`, requerido): Diccionario con todas las variables del modelo:
  - **Temporales**: `hora`, `minuto`, `dia_de_semana`, `dia_del_ano`
  - **Meteorológicas**: `temperature`, `humidity`, `wind_speed`, `general_diffuse_flows`, `diffuse_flows`
  - **Lags**: `lag_power_consumption_1_hora`, `lag_power_consumption_24_horas`
  - **Rolling means**: `rolling_mean_power_consumption_1_hora`, `rolling_mean_power_consumption_24_horas`

**Response (`PredictionResponse`):**
```json
{
  "model_name": "powerTetouan_RF_zone_1_version_02_champion",
  "prediction": 28668.46,
  "timestamp": "2025-01-15T14:30:00",
  "features_used": ["temperature", "humidity", "hora", "minuto", "lag_power_consumption_1_hora", ...]
}
```

**Campos de respuesta:**
- **model_name**: Nombre del modelo champion cargado (formato `*_version_XX_champion.pkl`)
- **prediction**: Consumo estimado en kW
- **timestamp**: Instante de ejecución de la inferencia (ISO 8601)
- **features_used**: Lista exacta de features aplicadas por el pipeline

### Endpoint GET `/health` - Health Check

Verifica el estado del servicio y lista los modelos disponibles con sus versiones.

**Response:**
```json
{
  "status": "healthy",
  "models_available": {
    "powerTetouan_RF_zone_1_power_consumption": {
      "champion_version": "1",
      "latest_version": "2",
      "source": "mlflow"
    },
    "powerTetouan_RF_zone_2_power_consumption": {
      "champion_version": "1",
      "latest_version": "2",
      "source": "mlflow"
    },
    "powerTetouan_RF_zone_3_power_consumption": {
      "champion_version": "1",
      "latest_version": "2",
      "source": "mlflow"
    }
  },
  "timestamp": "2025-01-15T14:30:00"
}
```

### Endpoints de Monitoreo de Drift

#### POST `/monitoring/actual` - Registrar Valor Real

Registra el valor real de consumo eléctrico para compararlo con la predicción y calcular métricas de performance.

**Request (`ActualValueRequest`):**
```json
{
  "zone": 1,
  "actual_value": 28500.0,
  "timestamp": "2025-01-15T14:30:00"
}
```

**Response (`ActualValueResponse`):**
```json
{
  "status": "success",
  "message": "Valor real registrado exitosamente",
  "zone": 1,
  "actual_value": 28500.0,
  "timestamp": "2025-01-15T14:30:00"
}
```

#### GET `/monitoring/drift/status` - Estado del Monitoreo

Consulta el estado actual del sistema de monitoreo de drift para el modelo champion de una zona.

**Query Parameters:**
- `zone` (int, required): Zona a consultar (1, 2 o 3)

**Response (`DriftStatusResponse`):**
```json
{
  "zone": 1,
  "model_type": "Champion",
  "needs_drift_check": false,
  "last_check_time": "2025-01-15T12:00:00",
  "next_check_in_hours": 4.5,
  "latest_report_summary": {
    "total_alerts": 2,
    "critical_alerts": 0,
    "high_alerts": 1,
    "drift_detected": true
  }
}
```

#### POST `/monitoring/drift/check` - Ejecutar Chequeo Manual

Fuerza la ejecución del pipeline completo de detección de drift para el modelo champion.

**Query Parameters:**
- `zone` (int, required): Zona a evaluar (1, 2 o 3)

**Response (`DriftCheckResponse`):**
```json
{
  "status": "success",
  "message": "Chequeo de drift completado",
  "zone": 1,
  "model_type": "Champion",
  "summary": {
    "total_alerts": 3,
    "has_critical_alerts": false,
    "has_high_alerts": true,
    "requires_action": true
  },
  "recommendations": [
    "Schedule model retraining within 24-48 hours",
    "Review feature engineering pipeline for potential issues"
  ]
}
```

### Esquemas Pydantic (OpenAPI / Postman)

Los modelos Pydantic están definidos en `api/schemas.py` y se publican automáticamente en:
- **Swagger UI**: http://localhost:8000/docs
- **OpenAPI JSON**: http://localhost:8000/openapi.json (importable en Postman con *Import → Raw text*)

**Esquemas disponibles:**
- `PredictionRequest` / `PredictionResponse`: Para el endpoint `/predict`
- `ActualValueRequest` / `ActualValueResponse`: Para `/monitoring/actual`
- `DriftStatusResponse`: Para `/monitoring/drift/status`
- `DriftCheckResponse`: Para `/monitoring/drift/check`
- `ErrorResponse`: Respuestas de error con código, mensaje y detalles opcionales

### MLFlow Integration

**MLFlow UI**: http://localhost:5001

MLFlow proporciona tracking de experimentos, versionado de modelos y registro centralizado.

**Configuración:**
- **Backend Store**: SQLite local (`/mlflow/mlflow.db`) para metadatos de experimentos
- **Artifact Store**: S3 (`s3://itesm-mna/202502-equipo11/mlflow-artifacts`) para modelos y artefactos grandes
- **Tracking URI**: `http://mlflow:5000` (interno en Docker network) o `http://localhost:5001` (externo)

**Flujo de trabajo:**
1. Los modelos se entrenan y registran en MLFlow Registry con `mlflow.log_model()`
2. Se asignan alias (`champion`, `staging`, `production`) para gestionar versiones
3. La API carga automáticamente el modelo champion

**Comandos útiles:**
```bash
# Ver experimentos registrados
mlflow experiments list

# Ver runs de un experimento
mlflow runs list --experiment-id 1

# Registrar modelo con alias champion
mlflow models set-tag -n "powerTetouan_RF_zone_1_power_consumption" -v 5 champion
```

## Despliegue con Docker

El proyecto incluye soporte para Docker, facilitando el despliegue y la portabilidad del servicio.

#### Opción 1: Docker Build y Run (Solo API)

```bash
# Construir la imagen Docker
docker build -t ml-service:latest .

# Ejecutar el contenedor
docker run -p 8000:8000 ml-service:latest

# Verificar que el servicio está funcionando
curl http://localhost:8000/health
```

#### Opción 2: Docker Compose (API + MLFlow)

Esta opción levanta tanto la API como el servidor MLFlow para tracking de experimentos y registro de modelos.

```bash
# Configurar variables de entorno (crear archivo .env)
cat > .env << EOF
# AWS Credentials para acceso a S3 (modelos y artefactos)
AWS_ACCESS_KEY_ID=<<access_key>>
AWS_SECRET_ACCESS_KEY=<<secret_key>>
AWS_DEFAULT_REGION=<<region>>
AWS_S3_BUCKET=<<bucket_name>>

# Configuración de puertos
API_PORT=8000
MLFLOW_PORT=5001

# Configuración de logging
LOG_LEVEL=info

# Configuración de drift monitoring
DRIFT_CHECK_INTERVAL_HOURS=6
DRIFT_MONITORING_WINDOW_HOURS=24
DRIFT_PERFORMANCE_THRESHOLD=0.15
EOF

# Levantar todos los servicios
docker compose up -d

# Ver logs de todos los servicios
docker compose logs -f

# Ver logs solo de la API
docker compose logs -f api

# Ver logs solo de MLFlow
docker compose logs -f mlflow

# Detener servicios
docker compose down

# Detener y eliminar volúmenes
docker compose down -v
```

**Servicios disponibles:**
- **API FastAPI**: http://localhost:8000
- **MLFlow UI**: http://localhost:5001
- **Documentación API (Swagger)**: http://localhost:8000/docs
- **OpenAPI Schema**: http://localhost:8000/openapi.json

### Tags Versionados (DockerHub)

La imagen se publica con 3 tags para trazabilidad:

1. **Versión específica**: `equipo11/power-tetouan-api:2.0.0` (producción)
2. **Latest**: `equipo11/power-tetouan-api:latest` (última estable)
3. **Git commit**: `equipo11/power-tetouan-api:<hash>` (trazabilidad completa)

## Pruebas automatizadas

El proyecto incluye 54 tests automatizados que aseguran la calidad del codigo en todos los componentes.

**Tests disponibles:**
- `tests/test_api.py` (20 tests) - Endpoints de FastAPI (health, predict, monitoring)
- `tests/test_monitoring.py` (18 tests) - Sistema de drift monitoring
- `tests/test_preprocessing.py` (6 tests) - Normalizacion, medias moviles, outliers
- `tests/test_data.py` (3 tests) - Carga de datos y deteccion de headers
- `tests/test_modeling_evaluate.py` (2 tests) - Calculo de metricas
- `tests/test_modeling_predict.py` (2 tests) - Prediccion de modelos
- `tests/test_tracking.py` (2 tests) - MLflow tracking
- `tests/test_integration_pipeline.py` (1 test) - Pipeline extremo a extremo
- `tests/test_features.py` (1 test) - Feature engineering

Ejecuta todos los tests:

```bash
pytest -q
# Resultado esperado: 54 passed in 4.45s
```

O tests especificos:

```bash
# Tests de API FastAPI
pytest tests/test_api.py -v

# Tests de monitoreo de drift
pytest tests/test_monitoring.py -v

# Tests de preprocesamiento
pytest tests/test_preprocessing.py -v
```

## Sistema de Monitoreo de Drift

El proyecto incluye un sistema completo de detección de drift diseñado específicamente para series temporales.

### Características del Sistema

**Detección Multi-Nivel:**
- **Statistical Drift**: KS test, PSI (Population Stability Index), JS Divergence
- **Temporal Drift**: ACF (Autocorrelation Function), ADF test, Seasonal Decomposition
- **Performance Drift**: Sliding windows para monitorear RMSE, MAE, MAPE

**Arquitectura OOP:**
- Clases especializadas: `StatisticalDriftDetector`, `TimeSeriesDriftDetector`, `ModelPerformanceMonitor`
- Sistema de alertas multi-canal: consola, archivo, email (configurable)
- Pipeline de orquestación para ejecución automatizada

**Configuración para Tetouan:**
- Seasonal period: 144 (datos en intervalos de 10 minutos, 144 por día)
- Monitoring window: 24 horas
- Check interval: 6 horas (configurable)

### Uso del Sistema de Drift

#### Monitoreo Batch (Offline)

```bash
# Ejecutar análisis de drift entre datasets
python examples/drift_monitoring_demo.py
```

El script genera:
- Reportes JSON en `reports/drift_monitoring/`
- Alertas en consola con niveles de severidad
- Recomendaciones automatizadas

#### Monitoreo en Tiempo Real (API)

```bash
# Obtener estado del monitoreo
curl "http://localhost:8000/monitoring/drift/status?zone=1"

# Registrar valor real observado
curl -X POST http://localhost:8000/monitoring/actual \
  -H "Content-Type: application/json" \
  -d '{
    "zone": 1,
    "actual_value": 28500.0,
    "timestamp": "2025-01-15T14:30:00"
  }'

# Ejecutar chequeo manual de drift
curl -X POST "http://localhost:8000/monitoring/drift/check?zone=1"
```

**Response Example:**
```json
{
  "zone": 1,
  "model_type": "Champion",
  "needs_drift_check": false,
  "predictions_logged": 150,
  "actuals_logged": 145,
  "last_check_time": "2025-01-15T12:00:00",
  "next_check_in_hours": 4.5,
  "latest_report_summary": {
    "total_alerts": 2,
    "critical_alerts": 0,
    "high_alerts": 1,
    "drift_detected": true
  }
}
```

### Tipos de Drift Detectados

1. **Feature Drift**: Cambios en distribuciones de variables de entrada
2. **Label Drift**: Cambios en distribución del target
3. **Concept Drift**: Cambios en la relación X → Y
4. **Performance Drift**: Degradación de métricas del modelo
5. **Temporal Drift**: Cambios en patrones estacionales o autocorrelación

### Tests del Sistema

```bash
# Ejecutar tests de monitoreo
pytest tests/test_monitoring.py -v

# Resultado esperado: 20 tests passing
```