# Guia Completa de Docker - Power Tetouan API

Documentacion completa para contenerizar, construir, ejecutar y publicar la API de prediccion de consumo electrico de Tetouan City con sistema de monitoreo de data drift.

**Version**: 2.0.0
**Actualizado**: Enero 2025
**Mantenido por**: Equipo11 MLOps

---

## Tabla de Contenidos

1. [Quick Start](#quick-start)
2. [Requisitos Previos](#requisitos-previos)
3. [Arquitectura del Contenedor](#arquitectura-del-contenedor)
4. [Construccion de la Imagen](#construccion-de-la-imagen)
5. [Ejecucion del Contenedor](#ejecucion-del-contenedor)
6. [Docker Compose](#docker-compose)
7. [Publicacion en DockerHub](#publicacion-en-dockerhub)
8. [Scripts Automatizados](#scripts-automatizados)
9. [Volumenes y Persistencia](#volumenes-y-persistencia)
10. [Variables de Entorno](#variables-de-entorno)
11. [Troubleshooting](#troubleshooting)
12. [Referencias](#referencias)

---

## Quick Start

### Inicio Rapido (3 comandos)

```bash
# 1. Configurar variables de entorno
cp .env.example .env
# Editar .env con credenciales AWS

# 2. Construir imagen
docker build -t power-tetouan-api:latest .

# 3. Ejecutar contenedor
docker run -p 8000:8000 power-tetouan-api:latest
```

API disponible en: http://localhost:8000/docs

### Con Docker Compose (Recomendado para Produccion)

```bash
# Iniciar API + MLFlow
docker-compose up -d

# Ver logs
docker-compose logs -f api

# Detener servicios
docker-compose down
```

---

## Requisitos Previos

### Software Necesario

- **Docker** >= 20.10.0
- **Docker Compose** >= 2.0.0 (incluido en Docker Desktop)
- **Git** (para versionado de imagenes)
- **Bash** (para ejecutar scripts en Linux/Mac/WSL)

### Instalacion de Docker

#### Windows/Mac
1. Descargar Docker Desktop: https://www.docker.com/products/docker-desktop
2. Instalar y reiniciar
3. Verificar instalacion:
```bash
docker --version
docker-compose --version
```

#### Linux (Ubuntu/Debian)
```bash
# Instalar Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Agregar usuario al grupo docker
sudo usermod -aG docker $USER
newgrp docker

# Instalar Docker Compose
sudo apt-get install docker-compose-plugin

# Verificar
docker --version
docker compose version
```

### Credenciales AWS

Necesarias para descargar modelos desde S3 via MLFlow:

```bash
# Configurar en archivo .env
AWS_ACCESS_KEY_ID=your_key_here
AWS_SECRET_ACCESS_KEY=your_secret_here
AWS_DEFAULT_REGION=us-east-2
```

---

## Arquitectura del Contenedor

### Dockerfile Multi-Stage Build

El Dockerfile utiliza construccion multi-etapa para optimizar el tamano de la imagen final:

```
Etapa 1: Builder (python:3.11-slim)
  - Instala compiladores (gcc, g++, gfortran, make)
  - Compila dependencias cientificas (numpy, scipy, scikit-learn)
  - Instala requirements-api.txt
  - Tamano: ~1.5 GB (descartado despues)

Etapa 2: Runtime (python:3.11-slim)
  - Copia solo dependencias compiladas desde builder
  - Instala librerias de runtime (libgomp, libopenblas, liblapack)
  - Copia codigo fuente (api/, src/)
  - Crea directorios para logs y monitoreo
  - Usuario no-root (apiuser:1000) para seguridad
  - Expone puerto 8000
  - Tamano final: ~1.1 GB
```

### Componentes Incluidos

- **FastAPI** - Framework web para API REST
- **Uvicorn** - Servidor ASGI de alto rendimiento
- **Pydantic** - Validacion de datos y schemas
- **Scikit-learn, XGBoost, Statsmodels** - Librerias de Machine Learning
- **Sistema de monitoreo de drift** - Implementacion OOP personalizada
- **MLFlow client** - Para descargar modelos desde S3
- **Boto3** - Cliente AWS para acceso a S3

### Seguridad

- Multi-stage build para minimizar superficie de ataque
- Usuario no-root (apiuser) para ejecucion
- Sin credenciales hardcodeadas
- Health checks automaticos
- Logs estructurados con timestamps

---

## Construccion de la Imagen

### Opcion 1: Script Automatizado (Recomendado)

```bash
# Dar permisos de ejecucion
chmod +x scripts/docker-build.sh

# Construir con configuracion por defecto
./scripts/docker-build.sh

# Construir con version especifica
VERSION=2.1.0 ./scripts/docker-build.sh

# Construir con registry personalizado
DOCKER_REGISTRY=mycompany VERSION=2.0.0 ./scripts/docker-build.sh
```

El script crea automaticamente 3 tags:
- `equipo11/power-tetouan-api:2.0.0` (version especifica)
- `equipo11/power-tetouan-api:latest` (ultima estable)
- `equipo11/power-tetouan-api:<git-commit>` (trazabilidad)

**Logs guardados en**: `logs/docker/build_YYYYMMDD_HHMMSS.log`

### Opcion 2: Docker Build Manual

```bash
# Build basico
docker build -t power-tetouan-api:latest .

# Build con tags versionados
docker build \
  --build-arg VERSION=2.0.0 \
  --build-arg BUILD_DATE=$(date -u +"%Y-%m-%dT%H:%M:%SZ") \
  -t equipo11/power-tetouan-api:2.0.0 \
  -t equipo11/power-tetouan-api:latest \
  .
```

### Opcion 3: Docker Compose Build

```bash
# Build desde docker-compose.yml
docker-compose build

# Build con rebuild completo (sin cache)
docker-compose build --no-cache

# Build con argumentos personalizados
VERSION=2.0.0 docker-compose build
```

### Verificar Build Exitoso

```bash
# Listar imagenes construidas
docker images | grep power-tetouan-api

# Inspeccionar metadatos de la imagen
docker inspect equipo11/power-tetouan-api:latest

# Ver capas y tamano de cada una
docker history equipo11/power-tetouan-api:latest

# Ver tamano total
docker images equipo11/power-tetouan-api:latest
```

---

## Ejecucion del Contenedor

### Opcion 1: Script Automatizado (Recomendado)

```bash
# Configurar variables en .env
cp .env.example .env
# Editar .env con tus credenciales AWS

# Ejecutar contenedor
chmod +x scripts/docker-run.sh
./scripts/docker-run.sh
```

El script:
- Carga variables desde .env automaticamente
- Detiene y elimina contenedor existente si hay uno
- Inicia nuevo contenedor en modo daemon
- Muestra logs iniciales y endpoints disponibles
- Guarda log de ejecucion en `logs/docker/run_YYYYMMDD_HHMMSS.log`

### Opcion 2: Docker Run Manual

#### Ejecucion Basica
```bash
docker run -p 8000:8000 power-tetouan-api:latest
```

#### Con Configuracion Completa
```bash
docker run -d \
  --name power-tetouan-api \
  -p 8000:8000 \
  -e AWS_ACCESS_KEY_ID="${AWS_ACCESS_KEY_ID}" \
  -e AWS_SECRET_ACCESS_KEY="${AWS_SECRET_ACCESS_KEY}" \
  -e AWS_DEFAULT_REGION="us-east-2" \
  -e MLFLOW_TRACKING_URI="http://mlflow:5000" \
  -e LOG_LEVEL="info" \
  -e DRIFT_CHECK_INTERVAL_HOURS="6" \
  -e DRIFT_MONITORING_WINDOW_HOURS="24" \
  --restart unless-stopped \
  equipo11/power-tetouan-api:latest
```

#### Con Volumenes Persistentes
```bash
docker run -d \
  --name power-tetouan-api \
  -p 8000:8000 \
  -v prediction_logs:/app/logs/predictions \
  -v drift_reports:/app/reports \
  -v model_cache:/app/models \
  -e AWS_ACCESS_KEY_ID="${AWS_ACCESS_KEY_ID}" \
  -e AWS_SECRET_ACCESS_KEY="${AWS_SECRET_ACCESS_KEY}" \
  --restart unless-stopped \
  equipo11/power-tetouan-api:latest
```

### Verificar Ejecucion

```bash
# Ver contenedores en ejecucion
docker ps

# Ver logs en tiempo real
docker logs -f power-tetouan-api

# Verificar health check
curl http://localhost:8000/health

# Verificar respuesta formateada
curl http://localhost:8000/health | python -m json.tool

# Probar endpoint de documentacion
curl http://localhost:8000/docs
```

### Endpoints Disponibles

Una vez corriendo, la API expone:

| Endpoint | URL | Descripcion |
|----------|-----|-------------|
| Swagger UI | http://localhost:8000/docs | Documentacion interactiva de la API |
| Health Check | http://localhost:8000/health | Estado del servicio y modelos |
| Prediccion | POST /predict | Realizar prediccion de consumo |
| Drift Status | GET /monitoring/drift/status | Estado del monitoreo de drift |
| Drift Check | POST /monitoring/drift/check | Ejecutar analisis de drift |
| Actual Values | POST /monitoring/actual | Registrar valores reales |

---

## Docker Compose

### Arquitectura de Servicios

Docker Compose orquesta dos servicios principales:

1. **MLFlow Server** (puerto 5001)
   - Tracking server para modelos ML
   - Backend: SQLite local
   - Artifacts: S3 (itesm-mna bucket)
   - Healthcheck automatico

2. **API Service** (puerto 8000)
   - FastAPI con sistema de drift
   - Conecta a MLFlow via red Docker
   - Volumenes persistentes para logs y reportes
   - Healthcheck automatico

Ambos servicios se comunican via red `tetouan-network`.

### Comandos Docker Compose

#### Gestion de Servicios

```bash
# Iniciar todos los servicios en background
docker-compose up -d

# Ver estado de servicios
docker-compose ps

# Ver logs de todos los servicios
docker-compose logs -f

# Ver logs de un servicio especifico
docker-compose logs -f api
docker-compose logs -f mlflow

# Reiniciar un servicio
docker-compose restart api

# Detener servicios sin eliminar contenedores
docker-compose stop

# Detener y eliminar contenedores
docker-compose down

# Detener, eliminar contenedores Y volumenes
docker-compose down -v

# Rebuild y restart
docker-compose up -d --build

# Ver uso de recursos
docker-compose top

# Escalar servicio API (multiples replicas)
docker-compose up -d --scale api=3
```

#### Debugging y Monitoreo

```bash
# Shell interactivo en contenedor de API
docker-compose exec api /bin/bash

# Ejecutar comando en contenedor
docker-compose exec api env | grep AWS

# Ver configuracion de red
docker network inspect tetouan-network

# Ver volumenes creados
docker volume ls | grep tetouan
```

### Configuracion de Variables

Configurar en archivo `.env`:

```bash
# Docker Registry y Version
DOCKER_REGISTRY=equipo11
VERSION=2.0.0

# Puertos
API_PORT=8000
MLFLOW_PORT=5001

# AWS Credentials (obligatorio)
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_DEFAULT_REGION=us-east-2

# API Configuration
LOG_LEVEL=info

# Drift Monitoring
DRIFT_CHECK_INTERVAL_HOURS=6
DRIFT_MONITORING_WINDOW_HOURS=24
DRIFT_PERFORMANCE_THRESHOLD=0.15
```

### Acceso a Servicios

Con docker-compose activo:

- **API Swagger**: http://localhost:8000/docs
- **API Health**: http://localhost:8000/health
- **MLFlow UI**: http://localhost:5001
- **Drift Status**: http://localhost:8000/monitoring/drift/status?zone=1

---

## Publicacion en DockerHub

### Paso 1: Crear Cuenta en DockerHub

1. Registrarse en: https://hub.docker.com/signup
2. Verificar email
3. Crear repository (opcional, se crea automaticamente al hacer push)

### Paso 2: Login desde CLI

```bash
# Login interactivo
docker login

# Login con credenciales directas
docker login -u your_username -p your_password

# Login con token (mas seguro, recomendado)
echo $DOCKER_TOKEN | docker login -u your_username --password-stdin
```

### Paso 3: Taggear Imagen

```bash
# Formato: registry/username/repository:tag
docker tag power-tetouan-api:latest equipo11/power-tetouan-api:2.0.0
docker tag power-tetouan-api:latest equipo11/power-tetouan-api:latest
docker tag power-tetouan-api:latest equipo11/power-tetouan-api:$(git rev-parse --short HEAD)
```

### Paso 4: Push a DockerHub

#### Opcion A: Script Automatizado

```bash
chmod +x scripts/docker-push.sh
./scripts/docker-push.sh
```

El script realiza:
- Verifica que usuario este logueado
- Verifica que la imagen existe
- Hace push de 3 tags: version, latest y git commit
- Muestra URLs y comandos pull
- Guarda log en `logs/docker/push_YYYYMMDD_HHMMSS.log`

#### Opcion B: Push Manual

```bash
# Push version especifica
docker push equipo11/power-tetouan-api:2.0.0

# Push latest
docker push equipo11/power-tetouan-api:latest

# Push commit hash
docker push equipo11/power-tetouan-api:a1b2c3d

# Push todas las tags a la vez
docker push --all-tags equipo11/power-tetouan-api
```

### Paso 5: Verificar Publicacion

```bash
# Ver en navegador
https://hub.docker.com/r/equipo11/power-tetouan-api

# Pull desde otro servidor para verificar
docker pull equipo11/power-tetouan-api:2.0.0

# Ver informacion de la imagen remota
docker manifest inspect equipo11/power-tetouan-api:latest
```

### Estrategia de Versionado

Utilizamos Semantic Versioning (SemVer):

```
MAJOR.MINOR.PATCH

Ejemplo: 2.0.0
  2 = MAJOR (cambios incompatibles con API anterior)
  0 = MINOR (nueva funcionalidad compatible)
  0 = PATCH (bug fixes compatibles)
```

Para cada release crear 3 tags:

1. **Version exacta** (inmutable): `equipo11/power-tetouan-api:2.0.0`
2. **Latest** (mutable): `equipo11/power-tetouan-api:latest`
3. **Git commit** (trazabilidad): `equipo11/power-tetouan-api:a1b2c3d`

### Workflow de Release

```bash
# 1. Desarrollo en rama feature
git checkout -b feature/new-drift-detector
# ... hacer cambios y commits ...

# 2. Merge a develop
git checkout develop
git merge feature/new-drift-detector

# 3. Tag de release
git tag v2.1.0
git push origin v2.1.0

# 4. Build con tags
docker build \
  -t equipo11/power-tetouan-api:2.1.0 \
  -t equipo11/power-tetouan-api:latest \
  -t equipo11/power-tetouan-api:$(git rev-parse --short HEAD) \
  .

# 5. Push a DockerHub
docker push --all-tags equipo11/power-tetouan-api
```

---

## Scripts Automatizados

Los scripts en `scripts/` automatizan las tareas comunes de Docker con logging completo.

### docker-build.sh

Construye la imagen con tags versionados automaticamente.

```bash
# Uso basico
./scripts/docker-build.sh

# Con variables personalizadas
DOCKER_REGISTRY=mycompany VERSION=2.1.0 ./scripts/docker-build.sh
```

Funcionalidad:
- Detecta version desde variable de entorno o usa default (2.0.0)
- Detecta git commit hash para tag adicional
- Verifica que Dockerfile y requirements-api.txt existen
- Crea 3 tags: version, latest, commit
- Guarda log con timestamp en `logs/docker/build_YYYYMMDD_HHMMSS.log`
- Muestra resumen de imagenes creadas

### docker-run.sh

Ejecuta el contenedor con configuracion completa.

```bash
# Uso basico (carga .env automaticamente)
./scripts/docker-run.sh

# Con puerto personalizado
API_PORT=8001 ./scripts/docker-run.sh
```

Funcionalidad:
- Carga variables desde .env si existe
- Detiene y elimina contenedor existente
- Inicia contenedor en modo daemon (-d)
- Configura restart policy (unless-stopped)
- Pasa variables de entorno necesarias
- Espera a que contenedor este healthy
- Muestra logs iniciales y endpoints disponibles
- Guarda log en `logs/docker/run_YYYYMMDD_HHMMSS.log`

### docker-push.sh

Publica la imagen en DockerHub con todos los tags.

```bash
# Uso basico
./scripts/docker-push.sh

# Con registry personalizado
DOCKER_REGISTRY=mycompany VERSION=2.0.0 ./scripts/docker-push.sh
```

Funcionalidad:
- Verifica login en DockerHub
- Verifica que imagen existe localmente
- Hace push de 3 tags: version, latest, commit
- Muestra URLs de DockerHub y comandos pull
- Guarda log en `logs/docker/push_YYYYMMDD_HHMMSS.log`

### Logs de Scripts

Todos los scripts guardan logs automaticamente en `logs/docker/`:

```bash
# Estructura de logs
logs/docker/
  build_20250114_153045.log
  run_20250114_153120.log
  push_20250114_153200.log
  README.md

# Ver ultimo log de build
tail -f logs/docker/build_*.log | tail -1

# Ver todos los logs de hoy
ls -lh logs/docker/*_$(date +%Y%m%d)*.log

# Limpiar logs antiguos (mas de 30 dias)
find logs/docker -name "*.log" -mtime +30 -delete
```

Los logs NO se versionan en Git (excluidos en .gitignore).

---

## Volumenes y Persistencia

### Volumenes Definidos

Docker Compose define 4 volumenes persistentes:

#### 1. prediction_logs

Almacena logs de predicciones para monitoreo de drift.

```bash
# Ubicacion en contenedor
/app/logs/predictions/

# Archivos generados
predictions_zone_1_RandomForest.jsonl
predictions_zone_2_XGBoost.jsonl
actuals_zone_1.jsonl

# Ver contenido
docker exec power-tetouan-api ls -lh /app/logs/predictions/
```

#### 2. drift_reports

Reportes de deteccion de drift y alertas.

```bash
# Ubicacion en contenedor
/app/reports/drift_monitoring/
/app/reports/realtime_drift_monitoring/

# Archivos generados
drift_monitoring_report.json
drift_alerts.json
drift_actual_vs_pred.png
feature_distributions.png

# Ver reportes
docker exec power-tetouan-api ls -lh /app/reports/drift_monitoring/
```

#### 3. model_cache

Cache local de modelos descargados desde MLFlow/S3.

```bash
# Ubicacion en contenedor
/app/models/

# Archivos (opcionales, segun configuracion)
rf_zone_1_power_consumption.pkl
xgb_zone_2_power_consumption.pkl

# Ver cache
docker exec power-tetouan-api ls -lh /app/models/
```

#### 4. mlflow_data

Base de datos SQLite y metadatos de MLFlow.

```bash
# Ubicacion en contenedor
/mlflow/

# Archivos
mlflow.db  # Base de datos SQLite

# Ver tamano
docker exec mlflow-server ls -lh /mlflow/
```

### Gestion de Volumenes

```bash
# Listar volumenes existentes
docker volume ls

# Filtrar volumenes del proyecto
docker volume ls | grep -E "prediction_logs|drift_reports|model_cache|mlflow_data"

# Inspeccionar volumen especifico
docker volume inspect prediction_logs

# Ver ubicacion real en host
docker volume inspect prediction_logs | grep Mountpoint

# Ver uso de espacio por volumen
docker system df -v
```

### Backup de Volumenes

```bash
# Backup de prediction_logs
docker run --rm \
  -v prediction_logs:/data \
  -v $(pwd):/backup \
  alpine tar czf /backup/prediction_logs_backup_$(date +%Y%m%d).tar.gz /data

# Backup de drift_reports
docker run --rm \
  -v drift_reports:/data \
  -v $(pwd):/backup \
  alpine tar czf /backup/drift_reports_backup_$(date +%Y%m%d).tar.gz /data

# Backup de mlflow_data
docker run --rm \
  -v mlflow_data:/data \
  -v $(pwd):/backup \
  alpine tar czf /backup/mlflow_data_backup_$(date +%Y%m%d).tar.gz /data
```

### Restaurar Volumenes

```bash
# Restaurar prediction_logs desde backup
docker run --rm \
  -v prediction_logs:/data \
  -v $(pwd):/backup \
  alpine tar xzf /backup/prediction_logs_backup_20250114.tar.gz -C /

# Restaurar drift_reports
docker run --rm \
  -v drift_reports:/data \
  -v $(pwd):/backup \
  alpine tar xzf /backup/drift_reports_backup_20250114.tar.gz -C /
```

### Limpiar Volumenes

```bash
# CUIDADO: Esto elimina datos permanentemente

# Detener servicios primero
docker-compose down

# Eliminar volumen especifico
docker volume rm prediction_logs

# Eliminar todos los volumenes no usados
docker volume prune

# Eliminar volumenes del proyecto (con servicios detenidos)
docker-compose down -v
```

### Montar Volumenes Locales (Desarrollo)

Para desarrollo con hot-reload, descomentar en `docker-compose.yml`:

```yaml
# api service volumes:
  - ./api:/app/api:ro
  - ./src:/app/src:ro
```

O en docker run:

```bash
docker run \
  -v $(pwd)/api:/app/api:ro \
  -v $(pwd)/src:/app/src:ro \
  -p 8000:8000 \
  power-tetouan-api:latest
```

---

## Variables de Entorno

### Variables Requeridas

```bash
# AWS Credentials (OBLIGATORIO)
# Necesarias para descargar modelos desde S3 via MLFlow
AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE
AWS_SECRET_ACCESS_KEY=wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY
AWS_DEFAULT_REGION=us-east-2
```

### Variables de Configuracion de API

```bash
# Nivel de logging
LOG_LEVEL=info  # debug | info | warning | error

# Puerto de la API
API_PORT=8000
```

### Variables de MLFlow

```bash
# URI del tracking server
# Para docker-compose: usar nombre del servicio
MLFLOW_TRACKING_URI=http://mlflow:5000

# Para standalone: usar localhost
# MLFLOW_TRACKING_URI=http://localhost:5001

# Puerto del servidor MLFlow
MLFLOW_PORT=5001
```

### Variables de Monitoreo de Drift

```bash
# Intervalo de chequeo automatico (en horas)
DRIFT_CHECK_INTERVAL_HOURS=6

# Ventana de datos para analisis (en horas)
DRIFT_MONITORING_WINDOW_HOURS=24

# Umbral de degradacion de performance (0.0 - 1.0)
# 0.15 = 15% de degradacion en metricas
DRIFT_PERFORMANCE_THRESHOLD=0.15
```

### Variables de Docker

```bash
# Registry para publicacion
DOCKER_REGISTRY=equipo11

# Version de la imagen
VERSION=2.0.0

# Entorno de ejecucion
DOCKER_ENV=development  # development | production
```

### Archivo .env de Ejemplo

```bash
# Copiar desde ejemplo
cp .env.example .env

# Editar con tus valores
nano .env  # o vim, code, etc.
```

Contenido del `.env`:

```bash
# AWS Configuration
AWS_ACCESS_KEY_ID=tu_access_key_aqui
AWS_SECRET_ACCESS_KEY=tu_secret_key_aqui
AWS_DEFAULT_REGION=us-east-2

# API Server
API_HOST=0.0.0.0
API_PORT=8000
LOG_LEVEL=INFO

# MLFlow Configuration
MLFLOW_TRACKING_URI=http://mlflow:5000
MLFLOW_PORT=5001
USE_MLFLOW=true

# Drift Monitoring
DRIFT_CHECK_INTERVAL_HOURS=6
DRIFT_MONITORING_WINDOW_HOURS=24
DRIFT_PERFORMANCE_THRESHOLD=0.15

# Docker
DOCKER_REGISTRY=equipo11
VERSION=2.0.0
DOCKER_ENV=development
WORKERS=4
RELOAD=false
```

### Validar Variables de Entorno

```bash
# Ver variables dentro del contenedor
docker exec power-tetouan-api env

# Filtrar solo AWS
docker exec power-tetouan-api env | grep AWS

# Filtrar solo drift
docker exec power-tetouan-api env | grep DRIFT

# Verificar que variables estan cargadas
docker inspect power-tetouan-api | grep -A 20 "Env"
```

---

## Troubleshooting

### Problema: Build Falla

**Sintoma**: Error durante `docker build`

**Soluciones**:

```bash
# 1. Error: "unable to resolve image"
# Actualizar imagen base
docker pull python:3.11-slim

# 2. Error: "no space left on device"
# Limpiar imagenes no usadas
docker system df
docker system prune -a

# 3. Error: "requirements-api.txt: no such file"
# Verificar que archivo existe
ls -lh requirements-api.txt

# 4. Dependencias fallan al compilar
# Rebuild sin cache
docker build --no-cache -t power-tetouan-api .

# 5. Timeout durante build
# Aumentar timeout de Docker Desktop
# Settings > Resources > Advanced > Build timeout
```

### Problema: Contenedor se Detiene Inmediatamente

**Sintoma**: Container exits right after starting

**Diagnostico**:

```bash
# Ver logs para identificar error
docker logs power-tetouan-api

# Ver ultimas 50 lineas
docker logs power-tetouan-api --tail 50

# Seguir logs en tiempo real
docker logs -f power-tetouan-api
```

**Causas comunes**:

1. **Puerto ya en uso**
```bash
# Ver que proceso usa puerto 8000
lsof -i :8000  # Linux/Mac
netstat -ano | findstr :8000  # Windows

# Usar otro puerto
docker run -p 8001:8000 power-tetouan-api:latest
```

2. **Variables de entorno faltantes**
```bash
# Verificar que .env existe y esta cargado
cat .env
docker run -e AWS_ACCESS_KEY_ID=... power-tetouan-api:latest
```

3. **Permisos de archivos**
```bash
# Verificar permisos dentro del contenedor
docker exec power-tetouan-api ls -la /app
docker exec power-tetouan-api whoami  # debe ser apiuser
```

### Problema: No Puede Conectar a MLFlow

**Sintoma**: API logs show MLFlow connection errors

**Diagnostico**:

```bash
# 1. Verificar que MLFlow esta corriendo
docker ps | grep mlflow

# 2. Verificar logs de MLFlow
docker logs mlflow-server --tail 50

# 3. Ping desde contenedor API
docker exec power-tetouan-api ping mlflow

# 4. Verificar DNS resolution
docker exec power-tetouan-api nslookup mlflow

# 5. Verificar red Docker
docker network inspect tetouan-network
```

**Soluciones**:

```bash
# Si MLFlow no esta corriendo
docker-compose up -d mlflow

# Si hay problema de DNS, usar IP directa
MLFLOW_IP=$(docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' mlflow-server)
docker run -e MLFLOW_TRACKING_URI=http://$MLFLOW_IP:5000 power-tetouan-api:latest

# Reiniciar servicios con red limpia
docker-compose down
docker network prune
docker-compose up -d
```

### Problema: Modelos No Se Descargan desde S3

**Sintoma**: Errors loading models from S3

**Diagnostico**:

```bash
# 1. Verificar credenciales AWS
docker exec power-tetouan-api env | grep AWS

# 2. Probar acceso a S3 desde contenedor
docker exec power-tetouan-api \
  python -c "import boto3; s3=boto3.client('s3'); print(s3.list_buckets())"

# 3. Verificar permisos de bucket
aws s3 ls s3://itesm-mna/202502-equipo11/mlruns

# 4. Ver logs de MLFlow
docker-compose logs mlflow | grep -i error
```

**Soluciones**:

```bash
# Verificar que credenciales AWS son correctas
# Editar .env con credenciales validas

# Verificar politicas IAM de usuario AWS
# Debe tener permisos s3:GetObject, s3:PutObject

# Reiniciar con nuevas credenciales
docker-compose down
docker-compose up -d
```

### Problema: Health Check Falla

**Sintoma**: Container shows as "unhealthy"

**Diagnostico**:

```bash
# Ver estado de health check
docker inspect --format='{{json .State.Health}}' power-tetouan-api | python -m json.tool

# Probar health endpoint manualmente
curl http://localhost:8000/health

# Desde dentro del contenedor
docker exec power-tetouan-api curl -f http://localhost:8000/health
```

**Soluciones**:

```bash
# Si API no responde, ver logs
docker logs power-tetouan-api

# Si health endpoint falla pero API funciona
# Actualizar healthcheck en Dockerfile o docker-compose.yml

# Deshabilitar healthcheck temporalmente
docker run --no-healthcheck power-tetouan-api:latest
```

### Problema: Logs de Drift No Se Guardan

**Sintoma**: No prediction logs or drift reports generated

**Diagnostico**:

```bash
# 1. Verificar permisos de directorios
docker exec power-tetouan-api ls -la /app/logs/predictions
docker exec power-tetouan-api ls -la /app/reports/drift_monitoring

# 2. Verificar que volumenes estan montados
docker inspect power-tetouan-api | grep -A 10 "Mounts"

# 3. Verificar espacio en disco
docker system df
```

**Soluciones**:

```bash
# Crear directorios manualmente
docker exec power-tetouan-api mkdir -p /app/logs/predictions
docker exec power-tetouan-api mkdir -p /app/reports/drift_monitoring

# Verificar ownership
docker exec power-tetouan-api chown -R apiuser:apiuser /app/logs /app/reports

# Recrear volumenes
docker-compose down -v
docker-compose up -d
```

### Problema: Alto Uso de Disco

**Sintoma**: Docker consuming too much disk space

**Diagnostico**:

```bash
# Ver uso de espacio total
docker system df

# Ver uso detallado
docker system df -v

# Ver imagenes y tamano
docker images

# Ver volumenes y tamano
docker volume ls
```

**Soluciones**:

```bash
# 1. Limpiar builds antiguos
docker builder prune

# 2. Limpiar imagenes no usadas
docker image prune -a

# 3. Limpiar volumenes no usados
docker volume prune

# 4. Limpieza completa (CUIDADO: elimina todo)
docker system prune -a --volumes

# 5. Mantener solo ultimas 3 versiones de imagen
docker images | grep power-tetouan-api | tail -n +4 | awk '{print $3}' | xargs docker rmi

# 6. Configurar limites en Docker Desktop
# Settings > Resources > Advanced > Disk image size
```

### Problema: Docker Compose No Inicia

**Sintoma**: `docker-compose up` fails

**Diagnostico**:

```bash
# Validar sintaxis de docker-compose.yml
docker-compose config

# Ver errores detallados
docker-compose up

# Ver logs de servicios especificos
docker-compose logs mlflow
docker-compose logs api
```

**Soluciones**:

```bash
# 1. Error de version
# Remover linea "version: '3.8'" si aparece warning

# 2. Puertos en uso
# Cambiar puertos en .env
API_PORT=8001
MLFLOW_PORT=5002

# 3. Volumenes no se pueden crear
# Limpiar volumenes existentes
docker-compose down -v

# 4. Network conflicts
# Eliminar redes Docker
docker network prune

# 5. Rebuild desde cero
docker-compose down -v
docker-compose build --no-cache
docker-compose up -d
```

### Comandos Utiles para Debugging

```bash
# Ver todos los contenedores (incluidos detenidos)
docker ps -a

# Ver procesos dentro del contenedor
docker top power-tetouan-api

# Estadisticas de recursos en tiempo real
docker stats power-tetouan-api

# Shell interactivo en contenedor
docker exec -it power-tetouan-api /bin/bash

# Copiar archivo desde contenedor
docker cp power-tetouan-api:/app/logs/predictions/predictions.jsonl ./

# Copiar archivo hacia contenedor
docker cp local_file.txt power-tetouan-api:/app/

# Ver cambios en filesystem del contenedor
docker diff power-tetouan-api

# Exportar contenedor a archivo tar
docker export power-tetouan-api -o power-tetouan-api.tar

# Ver eventos de Docker
docker events

# Inspeccionar configuracion completa
docker inspect power-tetouan-api > container_config.json
```

---

## Referencias

### Documentacion Oficial

- Docker Docs: https://docs.docker.com/
- Docker Compose: https://docs.docker.com/compose/
- Dockerfile Best Practices: https://docs.docker.com/develop/develop-images/dockerfile_best-practices/
- DockerHub: https://hub.docker.com/
- MLFlow Docker: https://www.mlflow.org/docs/latest/docker.html

### Tutoriales

- Multi-Stage Builds: https://docs.docker.com/build/building/multi-stage/
- Docker Security: https://docs.docker.com/engine/security/
- Docker Networking: https://docs.docker.com/network/
- Volume Management: https://docs.docker.com/storage/volumes/

### Herramientas Utiles

- Dive: https://github.com/wagoodman/dive (Explorar capas de imagen)
- Hadolint: https://github.com/hadolint/hadolint (Linter para Dockerfiles)
- Docker Slim: https://github.com/docker-slim/docker-slim (Optimizar imagenes)
- Trivy: https://github.com/aquasecurity/trivy (Escaneo de vulnerabilidades)

### Recursos del Proyecto

- Repositorio Git: https://github.com/equipo11/mlops
- DockerHub: https://hub.docker.com/r/equipo11/power-tetouan-api
- Documentacion de API: Ver `/docs` en el repositorio
- Postman Collection: `docs/postman/predict_collection.postman_collection.json`

### Contacto y Soporte

- Equipo: Equipo11 MLOps
- Curso: ITESM MNA - MLOps
- Periodo: 202502
- Repositorio Issues: Para reportar problemas

---

**Version del Documento**: 2.0.0
**Ultima Actualizacion**: Enero 2025
**Mantenido por**: Equipo11 MLOps

Este documento consolida toda la informacion necesaria para trabajar con Docker en el proyecto Power Tetouan API. Para actualizaciones o correcciones, contactar al equipo de desarrollo.
