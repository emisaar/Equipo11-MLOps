# Implementación del Sistema de Drift Monitoring

**Proyecto:** Predicción de Consumo Eléctrico - Tetouan City
**Versión:** 1.0.0
**Fecha:** Noviembre 2025
**Equipo:** Equipo11 MLOps

## Resumen
Se verificó e implementó el sistema de monitoreo de drift para la API de predicción de consumo eléctrico, enfocado en un modelo champion único (zona 3).

---

## 1. Arquitectura Implementada

### Componentes Principales

#### 1.1 PredictionLogger (`api/drift_monitor.py`)
- **Propósito**: Registra predicciones y valores reales para monitoreo
- **Almacenamiento**: JSONL files por zona y modelo
- **Funcionalidades**:
  - `log_prediction()`: Registra predicción con features
  - `log_actual_value()`: Registra valor real observado
  - `load_predictions()`: Carga predicciones históricas
  - `merge_predictions_with_actuals()`: Une predicciones con valores reales

#### 1.2 RealTimeDriftMonitor (`api/drift_monitor.py`)
- **Propósito**: Monitorea drift en tiempo real
- **Pipeline**: `DriftMonitoringPipeline` de `src/monitoring`
- **Funcionalidades**:
  - `check_drift()`: Ejecuta detección de drift
  - `should_check_drift()`: Determina si es tiempo de chequear
  - `get_drift_status()`: Obtiene estado actual del monitoreo

**Características habilitadas**:
```python
self.pipeline = DriftMonitoringPipeline(
    output_dir=self.output_dir,
    enable_statistical=True,   # Detección estadística (KS test, etc.)
    enable_timeseries=True,    # Análisis de series de tiempo (ACF, ADF)
    enable_performance=True,   # Monitoreo de performance (RMSE, MAE)
)
```

### 1.2 Integración FastAPI

**Endpoints de Monitoreo**:

1. `/monitoring/actual` (POST)
   - Registra valores reales observados
   - Permite tracking de performance del modelo

2. `/monitoring/drift/status` (GET)
   - Consulta estado del monitoreo
   - Retorna última verificación y próximo chequeo

3. `/monitoring/drift/check` (POST)
   - Ejecuta chequeo manual de drift
   - Genera reporte completo

---

## 2. Test App - Simulador de Drift

### 2.1 Componentes

**Archivos principales**:
- `test_app/main.py`: Ejecutor principal del test
- `test_app/data_generator.py`: Generador de datos sintéticos con drift
- `test_app/api_client.py`: Cliente HTTP para la API
- `test_app/visualizer.py`: Generación de gráficas

### 2.2 Flujo de Ejecución

1. **Verificación de API** (`check_api_health()`)
   - Conecta con API en http://localhost:8000
   - Detecta modelo champion automáticamente
   - Infiere zona del modelo (zona 3 para RF_zone_3)

2. **Generación de Predicciones** (`run_predictions()`)
   - Genera 200 muestras sintéticas
   - Introduce drift gradual a partir de muestra 50
   - Tipos de drift: temperature, humidity, seasonal, all
   - Intensidad configurable (default: 30%)

3. **Registro de Datos**
   - Cada predicción se registra en la API
   - Valor real simulado se registra mediante `/monitoring/actual`
   - Bias añadido durante drift para simular degradación

4. **Consulta de Drift** (`check_drift_status()`)
   - Verifica estado del monitoreo
   - Muestra última verificación y próxima programada

5. **Chequeo Manual** (`trigger_drift_checks()`)
   - Ejecuta detección de drift bajo demanda
   - Genera reporte completo si hay datos suficientes

6. **Visualización** (`generate_report()`)
   - Genera gráficas de predicciones vs reales
   - Muestra errores a lo largo del tiempo
   - Marca períodos con drift activo

### 2.3 Generación de Drift

**DataGenerator** simula drift realista:

```python
# Drift en temperatura (simula cambio climático)
if drift_type in ['temperature', 'all']:
    shift = intensity * 10.0  # Hasta +10°C
    features['temperature'] += shift

# Drift en humedad
if drift_type in ['humidity', 'all']:
    shift = intensity * 20.0  # Hasta +20%
    features['humidity'] += shift

# Drift estacional (radiación solar)
if drift_type in ['seasonal', 'all']:
    scale = 1.0 + (intensity * 0.5)  # Hasta 50% más radiación
    features['general_diffuse_flows'] *= scale
    features['diffuse_flows'] *= scale
```

**Drift gradual**:
- Comienza en muestra configurada (default: 50)
- Intensidad aumenta linealmente
- Progress = (muestra_actual - inicio_drift) / (total - inicio_drift)
- Intensidad actual = min(1.0, progress * intensidad_máxima)

---

## 3. Modelo Champion y Features

### Modelo Actual
**Archivo**: `models/powerTetouan_RF_zone_3_power_consumption_version_XX_champion.pkl`
**Tipo**: Random Forest para zona 3
**Alias MLFlow**: champion (versión 3)

### Features Requeridas

**Meteorológicas** (5):
- `temperature`
- `humidity`
- `wind_speed`
- `general_diffuse_flows`
- `diffuse_flows`

**Temporales** (4):
- `hora`
- `minuto`
- `dia_de_semana`
- `dia_del_ano`

**Lags y Rolling Means** (4):
- `lag_zone_3_power_consumption_1_hora`
- `lag_zone_3_power_consumption_24_horas`
- `rolling_mean_zone_3_power_consumption_1_hora`
- `rolling_mean_zone_3_power_consumption_24_horas`

**Total**: 13 features

---

## 4. Resultados del Test

### Ejecución Completada

**Configuración**:
- Total predicciones: 200
- Drift inicio: muestra 50
- Tipo de drift: all (temperature + humidity + seasonal)
- Intensidad máxima: 30%
- Intervalo: 2 segundos entre predicciones

**Resultados Observados**:

```
Predicciones sin drift (1-50):
  Error promedio: ~1%

Predicciones con drift (51-200):
  Error aumenta gradualmente
  Muestra 60:  2.49%
  Muestra 70:  4.72%
  Muestra 80:  3.97%
  Muestra 90:  5.02%
  Muestra 110: 8.03%
```

**Visualizaciones Generadas**:
- `test_results/predictions_timeline.png` (1.07 MB)
- `test_results/prediction_errors.png` (588 KB)

### Drift Monitoring

**Estado del Chequeo**:
```
status=insufficient_data
```

**Explicación**:
- Primera ejecución del sistema
- No hay datos históricos de referencia
- Se requieren al menos 100 muestras en período de referencia
- Se requieren al menos 100 muestras en período actual

---

## 5. Mejoras Implementadas

### 5.1 drift_monitor.py

**Cambios realizados**:

1. **Logger añadido**
```python
import logging
logger = logging.getLogger(__name__)
```

2. **Creación automática de directorio de reportes**
```python
# Create output directory if it doesn't exist
self.output_dir = Path("reports/realtime_drift_monitoring")
self.output_dir.mkdir(parents=True, exist_ok=True)
```

3. **Timeseries analysis habilitado**
```python
self.pipeline = DriftMonitoringPipeline(
    output_dir=self.output_dir,
    enable_statistical=True,
    enable_timeseries=True,  # Habilitado para detección comprehensiva
    enable_performance=True,
)
```

4. **Manejo robusto de errores en get_drift_status()**
```python
try:
    report_data = json.loads(report_file.read_text())
    latest_report = report_data.get("summary")
except Exception as e:
    logger.warning(f"Error loading drift report: {e}")
```

---

## 6. Limitaciones Actuales y Recomendaciones

### 6.1 Datos Insuficientes

**Problema**: Primera ejecución retorna `insufficient_data`

**Soluciones**:

1. **Opción A: Ejecutar múltiples veces**
   ```bash
   # Acumular datos históricos
   cd test_app
   python main.py  # Primera ejecución
   sleep 300       # Esperar 5 minutos
   python main.py  # Segunda ejecución
   ```

2. **Opción B: Reducir umbrales para testing**
   ```python
   # En drift_monitor.py línea 343
   if len(reference_data) < 50 or len(current_data) < 50:  # Reducido de 100
       return None
   ```

3. **Opción C: Usar datos de referencia pre-generados**
   ```python
   # Generar archivo de referencia
   # Guardar en data/reference/baseline_zone_3.parquet

   drift_monitor = RealTimeDriftMonitor(
       prediction_logger=prediction_logger,
       reference_data_path=Path("data/reference/baseline_zone_3.parquet")
   )
   ```

### 6.2 Logs de Predicciones en Docker

**Problema**: Los logs se almacenan dentro del contenedor Docker

**Solución**: Montar volumen en docker-compose.yml

```yaml
api:
  volumes:
    - prediction_logs:/app/logs/predictions
    - ./reports:/app/reports  # Montar reportes en host

volumes:
  prediction_logs:
    driver: local
```

Esto permite:
- Acceder a logs desde el host
- Persistencia entre reinicios de contenedores
- Análisis externo de datos

### 6.3 Ventana de Monitoreo

**Configuración actual**:
- Ventana de monitoreo: 24 horas
- Intervalo de chequeo: 6 horas
- Tolerancia para merge: 10 minutos

**Para testing**:
```python
drift_monitor = RealTimeDriftMonitor(
    prediction_logger=prediction_logger,
    monitoring_window_hours=1,    # 1 hora para testing
    check_interval_hours=0.5,     # Cada 30 min
)
```

---

## 7. Próximos Pasos

### 7.1 Para Producción

1. **Generar datos de referencia baseline**
   - Ejecutar modelo en condiciones normales por 1 semana
   - Guardar dataset de referencia

2. **Configurar alertas automáticas**
   - Integrar con Slack/Email
   - Definir umbrales de severidad

3. **Dashboards de monitoreo**
   - Grafana para visualización en tiempo real
   - Métricas de drift trending

### 7.2 Para Testing

1. **Script de inicialización de datos**
   ```bash
   # Crear bootstrap_drift_monitoring.sh
   # Genera datos de referencia sintéticos
   ```

2. **Tests automatizados**
   ```python
   # test_drift_detection.py
   # Verifica que drift se detecta correctamente
   ```

3. **Documentación de interpretación**
   - Guía de lectura de reportes
   - Acciones recomendadas por tipo de drift

---

## 8. Estructura de Directorios

```
Equipo11-MLOps/
├── api/
│   ├── drift_monitor.py            # Sistema de monitoreo
│   ├── main.py                     # Endpoints de FastAPI
│   └── predictor.py                # Carga del modelo champion
│
├── src/
│   └── monitoring/
│       ├── pipeline.py             # DriftMonitoringPipeline
│       ├── statistical.py          # Tests estadísticos
│       ├── timeseries.py           # Análisis temporal
│       └── performance.py          # Métricas de performance
│
├── test_app/
│   ├── main.py                     # Ejecutor de tests
│   ├── data_generator.py           # Generador con drift
│   ├── api_client.py               # Cliente HTTP
│   ├── visualizer.py               # Gráficas
│   └── test_results/               # Reportes generados
│       ├── predictions_timeline.png
│       └── prediction_errors.png
│
├── logs/predictions/               # Logs de predicciones (Docker)
│   ├── predictions_zone_3_Champion.jsonl
│   └── actuals_zone_3.jsonl
│
├── reports/realtime_drift_monitoring/  # Reportes de drift (Docker)
│   ├── drift_monitoring_report.json
│   ├── statistical_drift_report.json
│   ├── timeseries_drift_report.json
│   └── performance_monitoring_report.json
│
└── models/
    └── powerTetouan_RF_zone_3_power_consumption_version_XX_champion.pkl  # Modelo champion
```

---

## 9. Comandos de Verificación

```bash
# 1. Verificar API
curl http://localhost:8000/health | python -m json.tool

# 2. Ejecutar test de drift
cd test_app
python main.py

# 3. Ver resultados
ls -la test_results/

# 4. Consultar estado de drift
curl "http://localhost:8000/monitoring/drift/status?zone=3" | python -m json.tool

# 5. Forzar chequeo de drift
curl -X POST "http://localhost:8000/monitoring/drift/check?zone=3" | python -m json.tool

# 6. Ver logs dentro del contenedor (Windows: usar PowerShell)
docker exec power-tetouan-api ls /app/logs/predictions/
docker exec power-tetouan-api ls /app/reports/realtime_drift_monitoring/
```

---

## 10. Conclusiones

### Implementación Exitosa

1. **Sistema de drift monitoring funcional**
   - PredictionLogger registra datos correctamente
   - RealTimeDriftMonitor integrado con pipeline completo
   - Reportes se generan en directorio correcto

2. **Test app completo**
   - Genera drift realista y gradual
   - Visualizaciones informativas
   - Interfaz amigable con colores

3. **Modelo champion único**
   - Zona 3 correctamente implementada
   - Features todas generadas correctamente
   - API detecta automáticamente el modelo

### Áreas de Mejora

1. **Datos históricos**
   - Necesita ejecuciones múltiples o datos de referencia
   - Considerar reducir umbrales para testing

2. **Acceso a logs en Windows**
   - Comandos docker exec tienen problemas con Git Bash
   - Usar PowerShell o montar volúmenes

3. **Documentación de interpretación**
   - Añadir guía de lectura de reportes
   - Ejemplos de acciones correctivas

---

## Referencias

- Drift_Monitoring.md: Arquitectura teórica del sistema
- Docker.md: Configuración de contenedores
- src/monitoring/: Implementación de algoritmos de detección
