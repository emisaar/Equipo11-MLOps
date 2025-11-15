#!/bin/bash
# ============================================================================
# Script Completo de Docker - Build, Run y Push
# Workflow completo: verificación → login → build → test → push
# ============================================================================

set -e

# Colores
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuración
IMAGE_NAME="power-tetouan-api"
VERSION="${VERSION:-2.0.0}"
PORT="${PORT:-8000}"

echo ""
echo -e "${CYAN}============================================================================${NC}"
echo -e "${CYAN}  DOCKER WORKFLOW COMPLETO - Power Tetouan API${NC}"
echo -e "${CYAN}============================================================================${NC}"
echo ""

# ============================================================================
# FASE 1: VERIFICACIONES
# ============================================================================

echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}  FASE 1: VERIFICACIONES${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

# Verificar Docker instalado
echo -e "${BLUE}[1/3] Verificando Docker...${NC}"
if ! command -v docker &> /dev/null; then
    echo -e "${RED}✗ Docker no está instalado${NC}"
    echo "  Instalar desde: https://docs.docker.com/get-docker/"
    exit 1
fi
DOCKER_VERSION=$(docker --version)
echo -e "${GREEN}✓ Docker instalado: ${DOCKER_VERSION}${NC}"
echo ""

# Verificar Docker daemon
echo -e "${BLUE}[2/3] Verificando Docker daemon...${NC}"
if ! docker info &> /dev/null; then
    echo -e "${RED}✗ Docker daemon no está corriendo${NC}"
    echo "  Inicia Docker Desktop o ejecuta: sudo systemctl start docker"
    exit 1
fi
echo -e "${GREEN}✓ Docker daemon corriendo${NC}"
echo ""

# Verificar autenticación
echo -e "${BLUE}[3/3] Verificando autenticación en Docker Hub...${NC}"
if docker info 2>/dev/null | grep -q "Username:"; then
    USERNAME=$(docker info 2>/dev/null | grep "Username:" | awk '{print $2}')
    echo -e "${GREEN}✓ Autenticado como: ${USERNAME}${NC}"
    LOGGED_IN=true
else
    echo -e "${YELLOW}! No estás autenticado${NC}"
    LOGGED_IN=false
fi
echo ""

# ============================================================================
# FASE 2: AUTENTICACIÓN
# ============================================================================

if [ "$LOGGED_IN" = false ]; then
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}  FASE 2: AUTENTICACIÓN EN DOCKER HUB${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""

    echo -e "${YELLOW}Es necesario autenticarte en Docker Hub${NC}"
    echo ""
    echo "¿Tienes una cuenta en Docker Hub? (s/n)"
    read -r HAS_ACCOUNT

    if [[ "$HAS_ACCOUNT" == "n" || "$HAS_ACCOUNT" == "N" ]]; then
        echo ""
        echo -e "${YELLOW}Pasos para crear cuenta:${NC}"
        echo "  1. Ve a: ${CYAN}https://hub.docker.com/signup${NC}"
        echo "  2. Completa el formulario de registro"
        echo "  3. Verifica tu email"
        echo "  4. Vuelve aquí y ejecuta este script nuevamente"
        echo ""
        exit 0
    fi

    echo ""
    echo -e "${BLUE}Ejecutando docker login...${NC}"
    echo ""

    if docker login; then
        echo ""
        echo -e "${GREEN}✓ Login exitoso${NC}"
        USERNAME=$(docker info 2>/dev/null | grep "Username:" | awk '{print $2}')
        echo -e "${GREEN}Autenticado como: ${USERNAME}${NC}"
        LOGGED_IN=true
    else
        echo ""
        echo -e "${RED}✗ Login falló${NC}"
        exit 1
    fi
    echo ""
fi

# Configurar DOCKER_REGISTRY con el usuario logueado
export DOCKER_REGISTRY="${DOCKER_REGISTRY:-$USERNAME}"

# ============================================================================
# FASE 3: BUILD DE IMAGEN
# ============================================================================

echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}  FASE 3: BUILD DE IMAGEN${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

# Verificar si ya existe la imagen
if docker images | grep -q "${DOCKER_REGISTRY}/${IMAGE_NAME}:${VERSION}"; then
    echo -e "${GREEN}✓ Imagen encontrada: ${DOCKER_REGISTRY}/${IMAGE_NAME}:${VERSION}${NC}"
else
    echo -e "${YELLOW}Imagen no encontrada. Se construirá ahora.${NC}"
fi

echo ""
echo -e "${BLUE}Construyendo imagen ${DOCKER_REGISTRY}/${IMAGE_NAME}:${VERSION}...${NC}"
echo ""

if [ -f "./scripts/docker-build.sh" ]; then
    bash ./scripts/docker-build.sh
else
    echo -e "${YELLOW}Script docker-build.sh no encontrado, usando docker build directo...${NC}"
    docker build \
        --build-arg VERSION="${VERSION}" \
        -t "${DOCKER_REGISTRY}/${IMAGE_NAME}:${VERSION}" \
        -t "${DOCKER_REGISTRY}/${IMAGE_NAME}:latest" \
        .
fi

echo ""
echo -e "${GREEN}✓ Imagen construida exitosamente${NC}"
echo ""
echo -e "${CYAN}Información de la imagen:${NC}"
docker images | grep "${IMAGE_NAME}" | head -3
echo ""

# ============================================================================
# FASE 4: TEST/RUN DE IMAGEN
# ============================================================================

echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}  FASE 4: TEST DE IMAGEN${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""


echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}  FASE 4: TEST DE IMAGEN${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}Ejecutando script docker-run.sh${NC}"
if [ -f "./scripts/docker-run.sh" ]; then
    bash ./scripts/docker-run.sh
else
    echo -e "${YELLOW}Script docker-run.sh no encontrado, iniciando docker run manual...${NC}"
    docker run -d         --name "${IMAGE_NAME}"         -p "${PORT}:8000"         "${DOCKER_REGISTRY}/${IMAGE_NAME}:${VERSION}"
fi

echo ""

echo ""

# ============================================================================
# FASE 5: PUSH A DOCKER HUB
# ============================================================================

echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}  FASE 5: PUSH A DOCKER HUB${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

echo -e "${BLUE}Ejecutando script docker-push.sh${NC}"
if [ -f "./scripts/docker-push.sh" ]; then
    bash ./scripts/docker-push.sh
else
    echo -e "${YELLOW}Script docker-push.sh no encontrado, usando docker push manual...${NC}"
    echo -e "${BLUE}Pushing ${DOCKER_REGISTRY}/${IMAGE_NAME}:${VERSION}...${NC}"
    docker push "${DOCKER_REGISTRY}/${IMAGE_NAME}:${VERSION}"
    echo -e "${GREEN}✓ Tag ${VERSION} pushed${NC}"
    echo -e "${BLUE}Pushing ${DOCKER_REGISTRY}/${IMAGE_NAME}:latest...${NC}"
    docker push "${DOCKER_REGISTRY}/${IMAGE_NAME}:latest"
    echo -e "${GREEN}✓ Tag latest pushed${NC}"
fi
# ============================================================================
# RESUMEN FINAL
# ============================================================================

echo ""
echo -e "${CYAN}============================================================================${NC}"
echo -e "${GREEN}  WORKFLOW COMPLETADO${NC}"
echo -e "${CYAN}============================================================================${NC}"
echo ""

echo -e "${CYAN}Resumen:${NC}"
echo "  ✓ Docker verificado"
echo "  ✓ Autenticación: ${USERNAME}"
echo "  ✓ Imagen: ${DOCKER_REGISTRY}/${IMAGE_NAME}:${VERSION}"
echo "  ✓ Run/test ejecutado: ${IMAGE_NAME}"
if docker ps | grep -q "${IMAGE_NAME}"; then
    echo "  ⚠ Contenedor corriendo en puerto ${PORT}"
fi
echo "  ✓ Push completado"

echo ""
echo -e "${CYAN}Comandos útiles:${NC}"
echo "  Ver imágenes:        docker images | grep ${IMAGE_NAME}"
echo "  Ver contenedores:    docker ps -a | grep ${IMAGE_NAME}"
echo "  Detener contenedor:  docker stop ${IMAGE_NAME}"
echo "  Eliminar contenedor: docker rm ${IMAGE_NAME}"
echo "  Logs del contenedor: docker logs ${IMAGE_NAME}"
echo ""
