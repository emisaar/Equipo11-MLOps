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
│  └─ evaluate.py            # Stage 5: Evaluación y comparación
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
│  └─ RESUMEN_IMPLEMENTACION.md  # Resumen ejecutivo de implementación
│
├─ scripts/                  # Scripts de automatización
│  ├─ docker-build.sh        # Build de imagen Docker con versionado
│  ├─ docker-run.sh          # Ejecución de contenedor
│  └─ docker-push.sh         # Publicación a DockerHub
│
├─ tests/                    # Tests automatizados
│  ├─ test_monitoring.py     # Tests del sistema de drift (20 tests)
│  ├─ test_preprocessing.py  # Tests de preprocesamiento
│  └─ test_integration_pipeline.py  # Tests de integración
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

Este proyecto entrena y compara 4 tipos de modelos:

1. **VAR (Vector AutoRegression)**: Modelo estadístico multivariado para capturar interdependencias entre zonas
2. **Random Forest**: Modelo de ensemble con features temporales y meteorológicas
3. **XGBoost**: Gradient boosting optimizado con búsqueda de hiperparámetros

Cada modelo (excepto VAR) se entrena individualmente para cada una de las 3 zonas de consumo.

## Requisitos
```bash
python -m venv .venv 

# Linux/Mac:
# source .venv/bin/activate  

# Windows: 
.venv\Scripts\activate

# Actualizar pip
python.exe -m pip install --upgrade pip

# Instalar librerias desde requirements.txt
pip install -r requirements.txt

# En caso de reinstalar las libreria
# pip install --upgrade --force-reinstall -r requirements.txt
# (opcional) conda: conda env create -f environment.yml
```

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
- **models.types**: Lista de modelos a entrenar
- **models.var/random_forest/xgboost/lstm**: Hiperparámetros específicos por modelo
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

## API REST para Predicción

El proyecto incluye una API REST desarrollada con FastAPI para exponer los modelos entrenados.

### Características de la API

- Endpoint POST `/predict` para predicción de consumo eléctrico
- Validación de entrada con Pydantic y manejo de errores robusto
- Soporte para modelos: VAR, Random Forest, XGBoost
- Documentación automática en `/docs` (Swagger UI)
- Cache de modelos en memoria para mejor rendimiento
- Health check endpoint en `/health`
- **Sistema de monitoreo de drift en tiempo real** (NUEVO)
  - Detección automática de drift estadístico, temporal y de performance
  - Alertas configurables por severidad
  - Endpoints para consulta de estado y chequeo manual

### Iniciar el Servidor API

```bash
# Instalar dependencias de la API
pip install -r requirements.txt

# Iniciar servidor en modo desarrollo
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

# Verificar estado del servicio
curl http://localhost:8000/health
```

### Endpoints Principales

#### POST `/predict` - Realizar Predicción

**Request:**
```json
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
```

**Response:**
```json
{
  "zone": 1,
  "model_type": "RandomForest",
  "model_path": "models/rf_zone_1_power_consumption.pkl",
  "prediction": 28668.46,
  "timestamp": "2025-11-07T11:02:35",
  "features_used": ["temperature", "humidity", "hora", "minuto", ...]
}
```

**Nota sobre Features**: Los modelos esperan features específicas generadas por `create_ml_features`:
- **Variables meteorológicas**: `temperature`, `humidity`, `wind_speed`, `general_diffuse_flows`, `diffuse_flows`
- **Features temporales**: `hora`, `minuto`, `dia_de_semana`, `dia_del_ano`
- **Lags**: `lag_zone_X_power_consumption_1_hora`, `lag_zone_X_power_consumption_24_horas` (donde X es 1, 2 o 3)
- **Rolling means**: `rolling_mean_zone_X_power_consumption_1_hora`, `rolling_mean_zone_X_power_consumption_24_horas`

#### GET `/health` - Health Check

**Response:**
```json
{
  "status": "healthy",
  "models_available": {
    "VAR": [0],
    "RandomForest": [1, 2, 3],
    "XGBoost": [1, 2, 3]
  },
  "timestamp": "2025-01-15T14:30:00"
}
```

### Modelos Disponibles en la API

Los modelos se cargan desde el directorio `models/` y están versionados con DVC:

| Tipo de Modelo | Ruta del Artefacto | Versión |
|----------------|-------------------|---------|
| VAR | `models/var_model.pkl` | Ver `dvc.lock` |
| Random Forest Zona 1 | `models/rf_zone_1_power_consumption.pkl` | Ver `dvc.lock` |
| Random Forest Zona 2 | `models/rf_zone_2_power_consumption.pkl` | Ver `dvc.lock` |
| Random Forest Zona 3 | `models/rf_zone_3_power_consumption.pkl` | Ver `dvc.lock` |
| XGBoost Zona 1 | `models/xgb_zone_1_power_consumption.pkl` | Ver `dvc.lock` |
| XGBoost Zona 2 | `models/xgb_zone_2_power_consumption.pkl` | Ver `dvc.lock` |
| XGBoost Zona 3 | `models/xgb_zone_3_power_consumption.pkl` | Ver `dvc.lock` |

**Nota**: Las versiones exactas de los modelos se encuentran en el archivo `dvc.lock`, que contiene los hashes MD5 de cada artefacto.

### Ejemplo de Uso con Python

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
)

result = response.json()
print(f"Predicción: {result['prediction']:.2f} kW")
print(f"Modelo usado: {result['model_path']}")
```

### Despliegue con Docker

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

Esta opción levanta tanto la API como el servidor MLFlow para tracking de experimentos.

```bash
# Configurar variables de entorno (crear archivo .env)
cat > .env << EOF
AWS_ACCESS_KEY_ID=tu_access_key
AWS_SECRET_ACCESS_KEY=tu_secret_key
AWS_DEFAULT_REGION=us-east-2
API_PORT=8000
EOF

# Levantar todos los servicios
docker compose up -d

# Ver logs
docker compose logs -f

# Detener servicios
docker compose down
```

**Servicios disponibles:**
- API FastAPI: http://localhost:8000
- MLFlow UI: http://localhost:5001
- Documentación API: http://localhost:8000/docs

### Pruebas automatizadas

Las pruebas nuevas viven en `tests/` y aseguran tanto utilidades de preprocesamiento como el recorrido completo.

**Tests disponibles:**
- `tests/test_preprocessing.py` - Normalización de nombres, medias móviles, outliers
- `tests/test_integration_pipeline.py` - Pipeline completo de LoadData a evaluación
- `tests/test_monitoring.py` - Sistema de drift monitoring (20 tests)

Ejecuta todos los tests:

```bash
pytest -q
```

O tests específicos:

```bash
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
curl "http://localhost:8000/monitoring/drift/status?zone=1&model_type=RandomForest"

# Registrar valor real observado
curl -X POST http://localhost:8000/monitoring/actual \
  -H "Content-Type: application/json" \
  -d '{
    "zone": 1,
    "actual_value": 28500.0,
    "timestamp": "2025-01-15T14:30:00"
  }'

# Ejecutar chequeo manual de drift
curl -X POST "http://localhost:8000/monitoring/drift/check?zone=1&model_type=RandomForest"
```

**Response Example:**
```json
{
  "zone": 1,
  "model_type": "RandomForest",
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

## Despliegue con Docker

El proyecto está completamente containerizado para despliegues reproducibles.

### Comandos Rápidos

```bash
# 1. Build de la imagen
docker build -t ml-service:latest .

# 2. Ejecutar contenedor
docker run -p 8000:8000 ml-service:latest

# 3. O usar Docker Compose (recomendado)
docker-compose up -d
```

### Características de la Imagen Docker

**Multi-Stage Build:**
- Stage 1 (Builder): Compilación de dependencias científicas (~1.5 GB, descartado)
- Stage 2 (Runtime): Imagen final optimizada (~600-800 MB)

**Contenido:**
- Python 3.11-slim
- FastAPI + Uvicorn
- Modelos ML (VAR, RandomForest, XGBoost)
- Sistema completo de monitoreo de drift
- Integración con MLFlow
- Soporte AWS S3

**Seguridad:**
- Usuario no-root (apiuser:1000)
- Sin credenciales hardcodeadas
- Health checks automáticos cada 30s
- Variables de entorno para secrets

**Volúmenes Persistentes:**
- `prediction_logs`: Historial de predicciones para drift monitoring
- `drift_reports`: Reportes de drift generados
- `model_cache`: Cache de modelos descargados
- `mlflow_data`: Base de datos MLFlow

### Docker Compose

Ejecuta API + MLFlow simultáneamente:

```bash
# Configurar variables de entorno
cp .env.example .env
# Editar .env con tus credenciales AWS

# Iniciar servicios
docker-compose up -d

# Verificar estado
docker-compose ps

# Ver logs
docker-compose logs -f api

# Detener servicios
docker-compose down
```

**Servicios disponibles:**
- API: http://localhost:8000
- MLFlow UI: http://localhost:5001
- Swagger UI: http://localhost:8000/docs

### Scripts de Deployment

```bash
# Build con versionado semántico (2.0.0, latest, git-commit)
./scripts/docker-build.sh

# Ejecutar con configuración completa
./scripts/docker-run.sh

# Publicar a DockerHub
docker login
./scripts/docker-push.sh
```

### Tags Versionados (DockerHub)

La imagen se publica con 3 tags para trazabilidad:

1. **Versión específica**: `equipo11/power-tetouan-api:2.0.0` (producción)
2. **Latest**: `equipo11/power-tetouan-api:latest` (última estable)
3. **Git commit**: `equipo11/power-tetouan-api:<hash>` (trazabilidad completa)

### Variables de Entorno

```bash
# AWS Credentials (para acceso a modelos en S3)
AWS_ACCESS_KEY_ID=<tu-access-key>
AWS_SECRET_ACCESS_KEY=<tu-secret-key>
AWS_DEFAULT_REGION=us-east-2

# MLFlow Configuration
MLFLOW_TRACKING_URI=http://mlflow:5000

# Drift Monitoring Configuration (opcional)
DRIFT_CHECK_INTERVAL_HOURS=6
DRIFT_MONITORING_WINDOW_HOURS=24
DRIFT_PERFORMANCE_THRESHOLD=0.15
```

### Verificación Post-Despliegue

```bash
# 1. Health check
curl http://localhost:8000/health

# 2. Verificar modelos disponibles
curl http://localhost:8000/health | jq .models_available

# 3. Predicción de prueba
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "zone": 1,
    "model_type": "RandomForest",
    "features": {"temperature": 23.5, "humidity": 65.2, ...}
  }'

# 4. Estado del drift monitoring
curl "http://localhost:8000/monitoring/drift/status?zone=1&model_type=RandomForest"
```

## Endpoints de la API

### API Principal

- **POST** `/predict` - Realizar predicción de consumo eléctrico
- **GET** `/health` - Health check y modelos disponibles

### Drift Monitoring (Nuevos)

- **GET** `/monitoring/drift/status` - Estado actual del monitoreo de drift
  - Query params: `zone` (int), `model_type` (str)
- **POST** `/monitoring/actual` - Registrar valor real observado
  - Body: `{"zone": int, "actual_value": float, "timestamp": str}`
- **POST** `/monitoring/drift/check` - Ejecutar chequeo manual de drift
  - Query params: `zone` (int), `model_type` (str)

### MLFlow

- **MLFlow UI**: http://localhost:5001 - Tracking de experimentos y modelos

### Documentación Detallada

**Documentación de la API:**
- **Swagger UI**: http://localhost:8000/docs
- **Código fuente**: `api/` (main.py, schemas.py, predictor.py, drift_monitor.py)

**Documentación de Docker:**
- **Guía completa**: [docs/DOCKER_DEPLOYMENT.md](docs/DOCKER_DEPLOYMENT.md)
- **Quick start**: [docs/DOCKER_QUICKSTART.md](docs/DOCKER_QUICKSTART.md)

**Documentación de Drift Monitoring:**
- **Sistema completo**: [docs/DRIFT_MONITORING.md](docs/DRIFT_MONITORING.md)
- **Resumen de implementación**: [docs/RESUMEN_IMPLEMENTACION.md](docs/RESUMEN_IMPLEMENTACION.md)
