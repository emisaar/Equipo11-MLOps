#!/bin/bash
# Script de verificación del despliegue del modelo champion
# Verifica que todos los componentes estén correctamente configurados

set -e

# Colores
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo ""
echo -e "${BLUE}============================================================================${NC}"
echo -e "${BLUE}  VERIFICACIÓN DE DESPLIEGUE - Modelo Champion${NC}"
echo -e "${BLUE}============================================================================${NC}"
echo ""

# Variables de conteo
CHECKS_PASSED=0
CHECKS_FAILED=0
WARNINGS=0

# Funcion para check exitoso
check_ok() {
    echo -e "${GREEN}[OK]${NC} $1"
    ((CHECKS_PASSED++))
}

# Funcion para check fallido
check_fail() {
    echo -e "${RED}[ERROR]${NC} $1"
    ((CHECKS_FAILED++))
}

# Funcion para advertencia
check_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
    ((WARNINGS++))
}

# ============================================================================
# VERIFICACIÓN 1: Directorio de modelos
# ============================================================================
echo -e "${BLUE}[1/6] Verificando directorio de modelos...${NC}"

if [ -d "models" ]; then
    check_ok "Directorio models/ existe"
else
    check_fail "Directorio models/ no existe"
fi

# ============================================================================
# VERIFICACIÓN 2: Modelo champion
# ============================================================================
echo ""
echo -e "${BLUE}[2/6] Verificando modelo champion...${NC}"

CHAMPION_FILES=$(ls models/*_champion.pkl 2>/dev/null | wc -l)

if [ "$CHAMPION_FILES" -gt 0 ]; then
    check_ok "Modelo champion encontrado"
    for file in models/*_champion.pkl; do
        if [ -f "$file" ]; then
            SIZE=$(du -h "$file" | cut -f1)
            echo "     └─ $(basename "$file") (${SIZE})"
        fi
    done
else
    check_warn "No se encontró modelo champion"
    echo "     └─ Ejecuta: dvc repro evaluate_models"
fi

# ============================================================================
# VERIFICACIÓN 3: Scripts de pipeline
# ============================================================================
echo ""
echo -e "${BLUE}[3/6] Verificando scripts de pipeline...${NC}"

if [ -f "pipeline/deploy.py" ]; then
    check_ok "Script de despliegue existe"
else
    check_fail "Script pipeline/deploy.py no encontrado"
fi

if [ -f "scripts/setup-docker-hub.sh" ]; then
    check_ok "Script de Docker Hub existe"
else
    check_fail "Script scripts/setup-docker-hub.sh no encontrado"
fi

# ============================================================================
# VERIFICACIÓN 4: DVC configuration
# ============================================================================
echo ""
echo -e "${BLUE}[4/6] Verificando configuración DVC...${NC}"

if [ -f "dvc.yaml" ]; then
    check_ok "Archivo dvc.yaml existe"

    if grep -q "deploy_champion:" dvc.yaml; then
        check_ok "Stage deploy_champion configurado en dvc.yaml"
    else
        check_fail "Stage deploy_champion no encontrado en dvc.yaml"
    fi
else
    check_fail "Archivo dvc.yaml no encontrado"
fi

# ============================================================================
# VERIFICACIÓN 5: Configuración de API
# ============================================================================
echo ""
echo -e "${BLUE}[5/6] Verificando configuración de API...${NC}"

if [ -f "api/predictor.py" ]; then
    check_ok "Archivo api/predictor.py existe"

    if grep -q "predict_with_champion" api/predictor.py; then
        check_ok "Método predict_with_champion implementado"
    else
        check_fail "Método predict_with_champion no encontrado"
    fi
else
    check_fail "Archivo api/predictor.py no encontrado"
fi

if [ -f "api/schemas.py" ]; then
    check_ok "Archivo api/schemas.py existe"
else
    check_fail "Archivo api/schemas.py no encontrado"
fi

# ============================================================================
# VERIFICACIÓN 6: Docker
# ============================================================================
echo ""
echo -e "${BLUE}[6/6] Verificando Docker...${NC}"

if command -v docker &> /dev/null; then
    check_ok "Docker está instalado"

    if docker info &> /dev/null; then
        check_ok "Docker daemon está corriendo"

        # Verificar si existe la imagen
        if docker images | grep -q "power-tetouan-api"; then
            check_ok "Imagen power-tetouan-api encontrada"

            # Mostrar información de la imagen
            echo "     └─ Imágenes disponibles:"
            docker images | grep "power-tetouan-api" | awk '{print "        - " $1 ":" $2 " (" $7 " " $8 ")"}'
        else
            check_warn "Imagen power-tetouan-api no encontrada"
            echo "     └─ Ejecuta: bash scripts/setup-docker-hub.sh"
        fi
    else
        check_warn "Docker daemon no está corriendo"
        echo "     └─ Inicia Docker Desktop"
    fi
else
    check_warn "Docker no está instalado"
fi

# ============================================================================
# RESUMEN
# ============================================================================
echo ""
echo -e "${BLUE}============================================================================${NC}"
echo -e "${BLUE}  RESUMEN DE VERIFICACIÓN${NC}"
echo -e "${BLUE}============================================================================${NC}"
echo ""

echo -e "Checks exitosos:  ${GREEN}${CHECKS_PASSED}${NC}"
echo -e "Checks fallidos:  ${RED}${CHECKS_FAILED}${NC}"
echo -e "Advertencias:     ${YELLOW}${WARNINGS}${NC}"

echo ""

if [ $CHECKS_FAILED -eq 0 ]; then
    echo -e "${GREEN}[OK] Sistema listo para despliegue${NC}"
    echo ""
    echo "Proximos pasos:"
    echo "  1. Entrenar y evaluar modelos: dvc repro evaluate_models"
    echo "  2. Desplegar modelo champion:  dvc repro deploy_champion"
    echo "  3. Verificar API:              curl http://localhost:8000/health"
    echo ""
    exit 0
else
    echo -e "${RED}[ERROR] Hay problemas que deben resolverse antes del despliegue${NC}"
    echo ""
    exit 1
fi
