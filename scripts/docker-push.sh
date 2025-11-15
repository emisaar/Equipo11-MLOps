#!/bin/bash
# ============================================================================
# Script de Push para Docker - Power Tetouan API
# Publica la imagen en DockerHub con tags versionados
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
DOCKER_REGISTRY="${DOCKER_REGISTRY:-}"  # Se obtiene del usuario de Docker Hub
VERSION="${VERSION:-2.0.0}"
GIT_COMMIT=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")

# Crear directorio de logs si no existe
LOGS_DIR="logs/docker"
mkdir -p "${LOGS_DIR}"

# Archivo de log con timestamp
LOG_FILE="${LOGS_DIR}/push_$(date +%Y%m%d_%H%M%S).log"

# Función para log con timestamp
log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $*"
}

# Redirigir toda la salida al log file
exec > "${LOG_FILE}" 2>&1

log "============================================================================"
log "  Pushing Docker Image to Registry"
log "============================================================================"
log ""

# Cargar variables de entorno del proyecto si existen
if [ -f ".env" ]; then
    set -a
    . .env
    set +a
fi

# Verificar login en Docker Hub
echo -e "${BLUE}Verificando autenticación en Docker Hub...${NC}"

# Intentar login automático si hay credenciales en entorno
LOGGED_USER=""
if docker info 2>/dev/null | grep -q "Username:"; then
    LOGGED_USER=$(docker info 2>/dev/null | grep "Username:" | awk '{print $2}')
    echo -e "${GREEN}✓ Logueado como: ${LOGGED_USER}${NC}"
else
    if [ -n "${DOCKER_USERNAME:-}" ] && [ -n "${DOCKER_PASSWORD:-}" ]; then
        echo -e "${BLUE}Intentando login automático en Docker Hub...${NC}"
        if echo "${DOCKER_PASSWORD}" | docker login --username "${DOCKER_USERNAME}" --password-stdin >/dev/null; then
            echo -e "${GREEN}Login automático exitoso con ${DOCKER_USERNAME}${NC}"
            LOGGED_USER="${DOCKER_USERNAME}"
        else
            echo -e "${YELLOW}Login automático falló${NC}"
        fi
    fi
fi

if [ -z "$LOGGED_USER" ]; then
    echo -e "${RED}Error: No estás logueado en Docker Hub${NC}"
    echo ""
    echo -e "${YELLOW}Pasos para configurar Docker Hub:${NC}"
    echo ""
    echo '1. Crear cuenta en Docker Hub (si no tienes una):'
    echo "   https://hub.docker.com/signup"
    echo ""
    echo "2. Login desde terminal:"
    echo "   ${GREEN}docker login${NC}"
    echo ""
    echo "3. Ingresar tu username y password cuando se solicite"
    echo ""
    echo "4. Ejecutar este script nuevamente"
    echo ""
    echo -e "${YELLOW}Nota: El repositorio se creará automáticamente en Docker Hub${NC}"
    echo "      al hacer el primer push como: \$DOCKER_REGISTRY/${IMAGE_NAME}"
    echo ""
    exit 1
fi

# Fallback sobre usuario definido manualmente
DOCKER_REGISTRY="${DOCKER_REGISTRY:-${DOCKER_USERNAME:-${LOGGED_USER}}}"
if [ -z "$DOCKER_REGISTRY" ]; then
    echo -e "${RED}Error: No se pudo determinar el Docker registry${NC}"
    echo "Exporta DOCKER_REGISTRY con tu username de Docker Hub."
    exit 1
fi

# Verificar que DOCKER_REGISTRY está configurado
if [ -z "$DOCKER_REGISTRY" ]; then
    echo -e "${RED}Error: No se pudo determinar el Docker registry${NC}"
    echo "Por favor, especifica tu username de Docker Hub:"
    echo "  export DOCKER_REGISTRY=tu_username"
    echo "  ./scripts/docker-push.sh"
    exit 1
fi

log "Configuration:"
log "  Registry: ${DOCKER_REGISTRY}"
log "  Image: ${IMAGE_NAME}"
log "  Version: ${VERSION}"
log "  Git Commit: ${GIT_COMMIT}"
log "  Log File: ${LOG_FILE}"
log ""

# Verificar que la imagen existe localmente
echo -e "${BLUE}Verificando imagen local...${NC}"
IMAGE_TAG_FULL="${DOCKER_REGISTRY}/${IMAGE_NAME}:${VERSION}"
IMAGE_TAG_SIMPLE="${IMAGE_NAME}:${VERSION}"
LOCAL_IMAGE=""
if docker images --format "{{.Repository}}:{{.Tag}}" | grep -q "${IMAGE_TAG_FULL}"; then
    LOCAL_IMAGE="${IMAGE_TAG_FULL}"
elif docker images --format "{{.Repository}}:{{.Tag}}" | grep -q "${IMAGE_TAG_SIMPLE}"; then
    LOCAL_IMAGE="${IMAGE_TAG_SIMPLE}"
fi

if [ -z "$LOCAL_IMAGE" ]; then
    echo -e "${RED}Error: Imagen no encontrada. Construye la imagen primero:${NC}"
    echo ""
    echo "  ${GREEN}./scripts/docker-build.sh${NC}"
    echo ""
    echo "O especifica tu registry al construir:"
    echo "  ${GREEN}DOCKER_REGISTRY=${DOCKER_REGISTRY} ./scripts/docker-build.sh${NC}"
    echo ""
    exit 1
fi

echo -e "${BLUE}Verificando tags de la imagen...${NC}"
NEEDS_RETAG=false
if [ "${LOCAL_IMAGE}" != "${IMAGE_TAG_FULL}" ]; then
    echo -e "${YELLOW}Warning: La imagen necesita ser re-tagueada para el registry${NC}"
    NEEDS_RETAG=true
fi

# Re-taguear si es necesario
if [ "$NEEDS_RETAG" = true ]; then
    echo -e "${BLUE}Re-tagueando imagen para ${DOCKER_REGISTRY}...${NC}"

    docker tag "$LOCAL_IMAGE" "${DOCKER_REGISTRY}/${IMAGE_NAME}:${VERSION}"
    docker tag "$LOCAL_IMAGE" "${DOCKER_REGISTRY}/${IMAGE_NAME}:latest"

    if [ "$GIT_COMMIT" != "unknown" ]; then
        docker tag "$LOCAL_IMAGE" "${DOCKER_REGISTRY}/${IMAGE_NAME}:${GIT_COMMIT}"
    fi

    echo -e "${GREEN}✓ Imagen re-tagueada correctamente${NC}"
fi

echo ""
echo -e "${GREEN}Pushing images to ${DOCKER_REGISTRY}...${NC}"
echo ""

# Push version tag
echo -e "${BLUE}[1/3] Pushing ${DOCKER_REGISTRY}/${IMAGE_NAME}:${VERSION}...${NC}"
if docker push "${DOCKER_REGISTRY}/${IMAGE_NAME}:${VERSION}"; then
    echo -e "${GREEN}✓ Tag ${VERSION} pushed successfully${NC}"
else
    echo -e "${RED}Error pushing version tag${NC}"
    echo ""
    echo -e "${YELLOW}Posibles causas:${NC}"
    echo "1. No tienes permisos para el repositorio ${DOCKER_REGISTRY}/${IMAGE_NAME}"
    echo "2. El repositorio debe ser público o debes tener acceso"
    echo "3. Verifica tu autenticación: docker login"
    echo ""
    echo -e "${YELLOW}Para crear el repositorio en Docker Hub:${NC}"
    echo "1. Ve a: https://hub.docker.com/repository/create"
    echo "2. Nombre del repositorio: ${IMAGE_NAME}"
    echo "3. Visibilidad: Public (recomendado para proyectos open source)"
    echo ""
    exit 1
fi

# Push latest tag
echo -e "${BLUE}[2/3] Pushing ${DOCKER_REGISTRY}/${IMAGE_NAME}:latest...${NC}"
docker push "${DOCKER_REGISTRY}/${IMAGE_NAME}:latest"
echo -e "${GREEN}✓ Tag latest pushed successfully${NC}"

# Push commit tag si está disponible
if [ "$GIT_COMMIT" != "unknown" ]; then
    echo -e "${BLUE}[3/3] Pushing ${DOCKER_REGISTRY}/${IMAGE_NAME}:${GIT_COMMIT}...${NC}"
    docker push "${DOCKER_REGISTRY}/${IMAGE_NAME}:${GIT_COMMIT}"
    echo -e "${GREEN}✓ Tag ${GIT_COMMIT} pushed successfully${NC}"
else
    echo -e "${YELLOW}Skipping commit tag (git not available)${NC}"
fi

echo ""
echo -e "${GREEN}============================================${NC}"
echo -e "${GREEN}  All tags pushed successfully!${NC}"
echo -e "${GREEN}============================================${NC}"
echo ""

# Mostrar URLs de Docker Hub
echo -e "${BLUE}Image available at:${NC}"
echo "  https://hub.docker.com/r/${DOCKER_REGISTRY}/${IMAGE_NAME}"
echo ""
echo -e "${BLUE}Pull commands:${NC}"
echo "  ${YELLOW}docker pull ${DOCKER_REGISTRY}/${IMAGE_NAME}:${VERSION}${NC}"
echo "  ${YELLOW}docker pull ${DOCKER_REGISTRY}/${IMAGE_NAME}:latest${NC}"
if [ "$GIT_COMMIT" != "unknown" ]; then
    echo "  ${YELLOW}docker pull ${DOCKER_REGISTRY}/${IMAGE_NAME}:${GIT_COMMIT}${NC}"
fi

log ""
log "============================================================================"
log "  Push Complete!"
log "============================================================================"
log ""
log "Execution log saved to: ${LOG_FILE}"
log ""
