#!/bin/bash
# ============================================================================
# Script de Run para Docker - Power Tetouan API
# Ejecuta el contenedor con configuración apropiada
# ============================================================================

set -e  # Exit on error

# Colores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuración
IMAGE_NAME="power-tetouan-api"
DOCKER_REGISTRY="${DOCKER_REGISTRY:-equipo11}"
VERSION="${VERSION:-latest}"
CONTAINER_NAME="power-tetouan-api"
API_PORT="${API_PORT:-8000}"

# Crear directorio de logs si no existe
LOGS_DIR="logs/docker"
mkdir -p "${LOGS_DIR}"

# Archivo de log con timestamp
LOG_FILE="${LOGS_DIR}/run_$(date +%Y%m%d_%H%M%S).log"

# Función para log con timestamp
log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $*"
}

# Redirigir toda la salida al log file (además de mostrarla)
exec > >(tee -a "${LOG_FILE}")
exec 2>&1

# Cargar variables de entorno si existe .env
if [ -f ".env" ]; then
    log "Loading environment variables from .env..."
    set -a
    . .env
    set +a
fi

log "============================================================================"
log "  Running Docker Container: ${IMAGE_NAME}"
log "============================================================================"
log ""
log "Configuration:"
log "  Image: ${DOCKER_REGISTRY}/${IMAGE_NAME}:${VERSION}"
log "  Container Name: ${CONTAINER_NAME}"
log "  API Port: ${API_PORT}"
log "  Log File: ${LOG_FILE}"
log ""

# Detener y eliminar contenedor existente si existe
if docker ps -a | grep -q "${CONTAINER_NAME}"; then
    echo -e "${YELLOW}Stopping existing container...${NC}"
    docker stop "${CONTAINER_NAME}" 2>/dev/null || true
    docker rm "${CONTAINER_NAME}" 2>/dev/null || true
fi

# Ejecutar contenedor
echo -e "${GREEN}Starting container...${NC}"

docker run -d \
    --name "${CONTAINER_NAME}" \
    -p "${API_PORT}:8000" \
    -e AWS_ACCESS_KEY_ID="${AWS_ACCESS_KEY_ID}" \
    -e AWS_SECRET_ACCESS_KEY="${AWS_SECRET_ACCESS_KEY}" \
    -e AWS_DEFAULT_REGION="${AWS_DEFAULT_REGION:-us-east-2}" \
    -e MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-http://localhost:5001}" \
    -e LOG_LEVEL="${LOG_LEVEL:-info}" \
    -e DRIFT_CHECK_INTERVAL_HOURS="${DRIFT_CHECK_INTERVAL_HOURS:-6}" \
    -e DRIFT_MONITORING_WINDOW_HOURS="${DRIFT_MONITORING_WINDOW_HOURS:-24}" \
    --restart unless-stopped \
    "${DOCKER_REGISTRY}/${IMAGE_NAME}:${VERSION}"

echo ""
echo -e "${GREEN}✓ Container started successfully!${NC}"
echo ""

# Esperar a que el contenedor esté saludable
echo -e "${BLUE}Waiting for container to be healthy...${NC}"
sleep 5

# Mostrar logs iniciales
echo ""
echo -e "${BLUE}Container logs (last 20 lines):${NC}"
docker logs --tail 20 "${CONTAINER_NAME}"

echo ""
echo -e "${BLUE}Container status:${NC}"
docker ps | grep "${CONTAINER_NAME}"

log ""
log "============================================================================"
log "  Container is Running!"
log "============================================================================"
log ""
log "API endpoints:"
log "  • Swagger UI: http://localhost:${API_PORT}/docs"
log "  • Health Check: http://localhost:${API_PORT}/health"
log "  • Drift Status: http://localhost:${API_PORT}/monitoring/drift/status?zone=1&model_type=RandomForest"
log ""
log "Useful commands:"
log "  • View logs: docker logs -f ${CONTAINER_NAME}"
log "  • Stop container: docker stop ${CONTAINER_NAME}"
log "  • Restart container: docker restart ${CONTAINER_NAME}"
log "  • Shell access: docker exec -it ${CONTAINER_NAME} /bin/bash"
log ""
log "Execution log saved to: ${LOG_FILE}"
log ""
