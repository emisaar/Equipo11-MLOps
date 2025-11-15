#!/bin/bash
# ============================================================================
# Script de Build para Docker - Power Tetouan API
# Construye la imagen Docker con versionado y tags apropiados
# ============================================================================

set -e  # Exit on error

# Colores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

if [ -f ".env" ]; then
    set -a
    . .env
    set +a
fi

# Configuración
IMAGE_NAME="power-tetouan-api"
DOCKER_REGISTRY="${DOCKER_REGISTRY:-equipo11}"  # DockerHub username o registry
VERSION="${VERSION:-2.0.0}"
BUILD_DATE=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
GIT_COMMIT=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")

# Crear directorio de logs si no existe
LOGS_DIR="logs/docker"
mkdir -p "${LOGS_DIR}"

# Archivo de log con timestamp
LOG_FILE="${LOGS_DIR}/build_$(date +%Y%m%d_%H%M%S).log"

# Función para log con timestamp
log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $*"
}

# Redirigir toda la salida al log file (además de mostrarla)
exec > >(tee -a "${LOG_FILE}")
exec 2>&1

log "============================================================================"
log "  Building Docker Image: ${IMAGE_NAME}"
log "============================================================================"
log ""
log "Configuration:"
log "  Registry: ${DOCKER_REGISTRY}"
log "  Image: ${IMAGE_NAME}"
log "  Version: ${VERSION}"
log "  Build Date: ${BUILD_DATE}"
log "  Git Commit: ${GIT_COMMIT}"
log "  Log File: ${LOG_FILE}"
log ""

# Verificar que estamos en el directorio raíz del proyecto
if [ ! -f "Dockerfile" ]; then
    echo -e "${RED}Error: Dockerfile not found. Please run this script from the project root.${NC}"
    exit 1
fi

# Verificar que requirements.txt existe
if [ ! -f "requirements.txt" ]; then
    echo -e "${RED}Error: requirements.txt not found.${NC}"
    exit 1
fi

# Build de la imagen
echo -e "${GREEN}Building Docker image...${NC}"

docker build \
    --build-arg BUILD_DATE="${BUILD_DATE}" \
    --build-arg VERSION="${VERSION}" \
    --build-arg GIT_COMMIT="${GIT_COMMIT}" \
    -t "${DOCKER_REGISTRY}/${IMAGE_NAME}:${VERSION}" \
    -t "${DOCKER_REGISTRY}/${IMAGE_NAME}:latest" \
    -t "${DOCKER_REGISTRY}/${IMAGE_NAME}:${GIT_COMMIT}" \
    --progress=plain \
    .

echo ""
echo -e "${GREEN}✓ Build completed successfully!${NC}"
echo ""

# Mostrar información de la imagen
echo -e "${BLUE}Image information:${NC}"
docker images | grep "${IMAGE_NAME}" | head -3

echo ""
echo -e "${BLUE}Image tags created:${NC}"
echo "  • ${DOCKER_REGISTRY}/${IMAGE_NAME}:${VERSION}"
echo "  • ${DOCKER_REGISTRY}/${IMAGE_NAME}:latest"
echo "  • ${DOCKER_REGISTRY}/${IMAGE_NAME}:${GIT_COMMIT}"

log ""
log "============================================================================"
log "  Build Complete!"
log "============================================================================"
log ""
log "Next steps:"
log "  1. Test the image:"
log "     docker run -p 8000:8000 ${DOCKER_REGISTRY}/${IMAGE_NAME}:${VERSION}"
log ""
log "  2. Push to registry:"
log "     ./scripts/docker-push.sh"
log ""
log "Log saved to: ${LOG_FILE}"
log ""
