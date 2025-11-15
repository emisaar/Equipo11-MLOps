# Test App - API de Predicción de Consumo Eléctrico

Aplicación de testing para probar la API de predicción de consumo eléctrico desplegada en Docker.

## Características

Esta aplicación simula un cliente real que:

1. **Hace predicciones continuamente** usando el modelo champion desplegado
2. **Registra valores reales** para monitoreo de performance
3. **Introduce drift gradual** en los datos para probar el sistema de detección
4. **Monitorea el estado** del drift y genera alertas
5. **Genera reportes visuales** de las predicciones y errores

## Requisitos

- Python 3.8+
- API desplegada y corriendo (Docker o local)

## Instalación

### 1. Crear entorno virtual

```bash
# Navegar al directorio de la app
cd test_app

# Crear entorno virtual
python -m venv venv

# Activar entorno virtual
# Windows:
venv\Scripts\activate

# Linux/Mac:
source venv/bin/activate
```

### 2. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 3. Configurar variables de entorno

```bash
# Copiar el archivo de ejemplo
cp .env.example .env

# Editar .env con tu configuración
# (opcional, los valores por defecto funcionan con Docker local)
```

## Configuración

El archivo `.env` contiene la siguiente configuración:

```env
# URL de la API (ajustar si la API está en otra máquina)
API_BASE_URL=http://localhost:8000

# Intervalo entre predicciones (segundos)
PREDICTION_INTERVAL=2

# Número de predicciones antes de introducir drift
DRIFT_START_AFTER=50

# Intensidad del drift (0.0 a 1.0)
DRIFT_INTENSITY=0.3

# Zonas a simular (separadas por comas)
SIMULATE_ZONES=1,2,3

# Total de predicciones a realizar
TOTAL_PREDICTIONS=200

# Tipo de drift: temperature, humidity, seasonal, all
DRIFT_TYPE=all

# Drift gradual o abrupto
DRIFT_GRADUAL=true
```

## Uso

### Ejecución Básica

```bash
python main.py
```

### Ejemplo de Salida

```
================================================================================
                    VERIFICANDO CONEXION CON LA API
================================================================================

[OK] API conectada: http://localhost:8000
[INFO]   Status: healthy
[INFO]   Modelos disponibles:
[INFO]     - powerTetouan_RF_zone_1_power_consumption: champion v3

================================================================================
                 EJECUTANDO 200 PREDICCIONES (Zona 1)
================================================================================

[WARN] Drift se introducira gradualmente a partir de la prediccion 50
[INFO]   Tipo de drift: all
[INFO]   Intensidad maxima: 30.0%

  [  1/200] Pred: 25432.18 kW, Real: 25689.34 kW, Error:  1.01%
  [ 10/200] Pred: 24891.52 kW, Real: 24567.89 kW, Error:  1.30%
  ...
  [ 50/200] Pred: 26123.45 kW, Real: 26345.12 kW, Error:  0.85%
  [ 60/200] Pred: 27456.78 kW, Real: 28912.34 kW, Error:  5.30% [DRIFT]
  ...

[OK] 200 predicciones completadas

================================================================================
                      VERIFICANDO ESTADO DE DRIFT
================================================================================

  Zona 1:
    Necesita chequeo: True
    Ultimo chequeo: 2025-11-15T10:30:00
    Proximo chequeo en: 0.5 horas
    Alertas totales: 3
    Requiere accion: True
[WARN]       ¡ALERTAS CRITICAS DETECTADAS!
```

## Estructura de la Aplicación

```
test_app/
├── main.py              # Punto de entrada principal
├── api_client.py        # Cliente para interactuar con la API
├── data_generator.py    # Generador de datos sintéticos
├── visualizer.py        # Visualización de resultados
├── requirements.txt     # Dependencias
├── .env.example         # Ejemplo de configuración
├── .env                 # Configuración (crear)
├── README.md            # Esta documentación
└── test_results/        # Directorio con reportes (generado)
    ├── predictions_timeline.png
    └── prediction_errors.png
```

## Simulación de Drift

La aplicación puede simular diferentes tipos de drift:

### Tipos de Drift

1. **Temperature Drift**: Shift en temperatura (+10°C máximo)
2. **Humidity Drift**: Shift en humedad (+20% máximo)
3. **Seasonal Drift**: Cambio en radiación solar (+50% máximo)
4. **All**: Combinación de todos los anteriores

### Parámetros de Drift

- **DRIFT_START_AFTER**: Predicción en la cual comenzar el drift
- **DRIFT_INTENSITY**: Intensidad del drift (0.0 = sin drift, 1.0 = máximo drift)
- **DRIFT_TYPE**: Tipo de drift a introducir
- **DRIFT_GRADUAL**: Si true, el drift aumenta gradualmente; si false, es abrupto

## Resultados

### Reportes Visuales

La aplicación genera gráficas en el directorio `test_results/`:

1. **predictions_timeline.png**:
   - Predicciones vs valores reales
   - Marca la región donde se introdujo drift

2. **prediction_errors.png**:
   - Error de predicción a lo largo del tiempo
   - Distribución de errores (normal vs drift)

### Logs

Los logs detallados se guardan en `test_app.log`

## Verificación del Sistema de Monitoreo

### Chequeo Manual

La app ejecuta chequeos de drift y muestra:

- Total de alertas detectadas
- Alertas críticas
- Alertas de prioridad alta
- Recomendaciones del sistema

### Ejemplo de Recomendaciones

```
[OK] Zona 1: Chequeo de drift completado
    Alertas totales: 5
    Alertas criticas: False
    Alertas altas: True
    Requiere accion: True

    Recomendaciones:
      - Schedule model retraining within 24-48 hours
      - Review feature engineering pipeline for potential issues
      - Monitor prediction errors closely
```

## Troubleshooting

### API no disponible

```
[ERROR] No se pudo conectar con la API: Connection refused
  URL: http://localhost:8000
  Asegurate de que la API este corriendo:
    docker ps | grep power-tetouan-api
```

**Solución**:
```bash
# Verificar que Docker esté corriendo
docker ps

# Si no está, iniciar la API
docker-compose up -d
```

### Error en predicciones

```
[ERROR] Error en prediccion 1: 400 Bad Request
```

**Solución**: Verificar que las features generadas coincidan con las esperadas por el modelo

## Personalización

### Ajustar Generación de Datos

Editar `data_generator.py` para modificar:
- Distribuciones de las features
- Rangos de valores
- Patrones de consumo

### Añadir Nuevos Tests

Editar `main.py` clase `TestRunner` para:
- Añadir nuevos escenarios de prueba
- Modificar la lógica de drift
- Integrar nuevos tipos de análisis

## Contribuir

Para reportar bugs o sugerir mejoras, contactar al equipo de MLOps.

---

**Equipo 11 - MLOps Project**
*Sistema de Predicción de Consumo Eléctrico - Tetouan City*
