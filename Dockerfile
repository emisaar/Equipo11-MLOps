# ============================================================================
# Dockerfile para API de Predicción de Consumo Eléctrico - Tetouan City
# Incluye sistema de monitoreo de data drift
# ============================================================================

# ============================================================================
# Etapa 1: Builder - Compilación de dependencias
# ============================================================================
FROM python:3.11-slim AS builder

LABEL maintainer="Equipo11 MLOps <equipo11@tec.mx>"
LABEL description="API FastAPI para predicción de consumo eléctrico con monitoreo de drift"
LABEL version="2.0.0"

WORKDIR /build

# Instalar dependencias de compilación necesarias para paquetes científicos
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libgomp1 \
    make \
    gfortran \
    libopenblas-dev \
    liblapack-dev \
    && rm -rf /var/lib/apt/lists/*

# Copiar archivos de dependencias
COPY pyproject.toml requirements-api.txt ./

# Actualizar pip y instalar dependencias
# Usar requirements-api.txt para builds optimizados (solo dependencias de producción)
# Construir wheelhouse para reutilizarlo en la imagen final
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip wheel --wheel-dir /wheels --no-cache-dir -r requirements-api.txt

# ============================================================================
# Etapa 2: Runtime - Imagen final optimizada
# ============================================================================
FROM python:3.11-slim

WORKDIR /app

# Metadata
LABEL maintainer="Equipo11 MLOps <team@example.com>"
LABEL description="API FastAPI para predicción de consumo eléctrico con monitoreo de drift"
LABEL version="2.0.0"

# Instalar dependencias de runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    libopenblas0 \
    liblapack3 \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copiar archivos de configuración necesarios para instalación
COPY pyproject.toml requirements-api.txt ./

# Copiar wheelhouse generado en el builder stage
COPY --from=builder /wheels /wheels

# Instalar dependencias desde los wheels (sin acceder a internet en esta etapa)
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir --no-index --find-links=/wheels -r requirements-api.txt && \
    rm -rf /wheels

# Copiar código fuente del proyecto
COPY api ./api
COPY src ./src

# Crear directorios necesarios para monitoreo de drift y modelos
# NOTA: /app/models se usa como cache local, los modelos se cargan dinámicamente desde MLflow/S3
RUN mkdir -p \
    /app/logs/predictions \
    /app/reports/drift_monitoring \
    /app/reports/realtime_drift_monitoring \
    /app/models \
    && chmod -R 755 /app/logs /app/reports /app/models

# Instalar proyecto en modo editable ANTES de cambiar al usuario no-root
RUN pip install --no-cache-dir -e .

# Crear usuario no-root para seguridad
RUN useradd -m -u 1000 -s /bin/bash apiuser && \
    chown -R apiuser:apiuser /app

# Cambiar a usuario no-root
USER apiuser

# Exponer puerto de la API
EXPOSE 8000

# Variables de entorno
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    API_PORT=8000 \
    LOG_LEVEL=info

# Health check mejorado
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Comando por defecto - FastAPI con Uvicorn
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "info"]
