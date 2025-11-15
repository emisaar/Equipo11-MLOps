#!/bin/bash
# ============================================================================
# Script Completo de Docker - Build, Run y Push
# Workflow completo: verificación → login → build → test → push
# ============================================================================

set -e
# Cargar variables de entorno del proyecto si existen
if [ -f ".env" ]; then
    set -a
    . .env
    set +a
fi

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
DOCKER_CHECK_TIMEOUT="${DOCKER_CHECK_TIMEOUT:-20}"

echo ""
echo -e "${CYAN}============================================================================${NC}"
echo -e "${CYAN}  DOCKER WORKFLOW COMPLETO - Power Tetouan API${NC}"
echo -e "${CYAN}============================================================================${NC}"
echo ""

# ============================================================================
# FASE 1: VERIFICACIONES
# ============================================================================

check_docker_daemon() {
    local tmp_log
    tmp_log=$(mktemp -t docker-info-check.XXXXXX)

    docker info >"$tmp_log" 2>&1 &
    local info_pid=$!
    local elapsed=0

    while kill -0 "${info_pid}" 2>/dev/null; do
        if [ "${elapsed}" -ge "${DOCKER_CHECK_TIMEOUT}" ]; then
            kill -TERM "${info_pid}" 2>/dev/null || true
            wait "${info_pid}" 2>/dev/null || true
            echo "Docker info output (last attempt):"
            cat "$tmp_log"
            rm -f "$tmp_log"
            return 2
        fi
        sleep 1
        elapsed=$((elapsed + 1))
    done

    local result=0
    wait "${info_pid}" >/dev/null 2>&1 || result=$?
    rm -f "$tmp_log"
    return $result
}

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
if ! check_docker_daemon; then
    echo -e "${RED}✗ Docker daemon no respondió en ${DOCKER_CHECK_TIMEOUT}s${NC}"
    echo "  Revisa que Docker Desktop esté iniciado o ejecuta: sudo systemctl start docker"
    echo "  También puedes validar manualmente con: docker info"
    if [ -n "${DOCKER_CONTEXT:-}" ]; then
        echo "  Contexto actual: ${DOCKER_CONTEXT}"
    fi
    exit 1
fi
echo -e "${GREEN}✓ Docker daemon corriendo${NC}"
echo ""

# ============================================================================
# FASE 2: AUTENTICACIÓN
# ============================================================================

# Verificar autenticación
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}  FASE 2: AUTENTICACIÓN EN DOCKER HUB${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo -e "${BLUE}[3/3] Verificando autenticación en Docker Hub...${NC}"
if docker info 2>/dev/null | grep -q "Username:"; then
    USERNAME=$(docker info 2>/dev/null | grep "Username:" | awk '{print $2}')
    echo -e "${GREEN}✓ Autenticado como: ${USERNAME}${NC}"
    LOGGED_IN=true
else
    echo -e "${YELLOW}! No estás autenticado${NC}"
    if [ -n "${DOCKER_USERNAME:-}" ] && [ -n "${DOCKER_PASSWORD:-}" ]; then
        echo -e "${BLUE}Intentando login automático en Docker Hub...${NC}"
        if echo "${DOCKER_PASSWORD}" | docker login --username "${DOCKER_USERNAME}" --password-stdin >/dev/null; then
            echo -e "${GREEN}Login automático exitoso con ${DOCKER_USERNAME}${NC}"
            LOGGED_USER="${DOCKER_USERNAME}"
            LOGGED_IN=true
        else
            echo -e "${YELLOW}Login automático falló${NC}"
            LOGGED_IN=false
        fi
    fi    
fi
echo ""

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
