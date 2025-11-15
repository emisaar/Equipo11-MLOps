# Sistema de Monitoreo de Data Drift para Series de Tiempo

## Documentación Técnica Completa

**Proyecto:** Predicción de Consumo Eléctrico - Tetouan City
**Versión:** 2.0.0
**Fecha:** Enero 2025
**Equipo:** Equipo11 MLOps

---

## Tabla de Contenidos

1. [Introducción](#1-introducción)
2. [Fundamentos Teóricos de Series de Tiempo](#2-fundamentos-teóricos-de-series-de-tiempo)
3. [Metodologías de Detección de Drift](#3-metodologías-de-detección-de-drift)
4. [Arquitectura del Sistema](#4-arquitectura-del-sistema)
5. [Componentes Principales](#5-componentes-principales)
6. [Configuración e Instalación](#6-configuración-e-instalación)
7. [Guía de Uso](#7-guía-de-uso)
8. [Integración con FastAPI](#8-integración-con-fastapi)
9. [Tests y Validación](#9-tests-y-validación)
10. [Interpretación de Resultados](#10-interpretación-de-resultados)
11. [Casos de Uso y Ejemplos](#11-casos-de-uso-y-ejemplos)
12. [Referencias Bibliográficas](#12-referencias-bibliográficas)

---

## 1. Introducción

### 1.1 Visión General

Este documento describe la implementación completa de un sistema de monitoreo de data drift diseñado específicamente para modelos de predicción que operan sobre series de tiempo. El sistema ha sido desarrollado para el proyecto de predicción de consumo eléctrico en Tetouan City, donde se utilizan múltiples modelos de machine learning (VAR, Random Forest, XGBoost) para predecir el consumo eléctrico en tres zonas distintas.

### 1.2 Motivación y Contexto

En producción, los modelos de machine learning enfrentan el desafío del "data drift" - cuando las distribuciones estadísticas de los datos cambian con el tiempo, causando degradación en el rendimiento del modelo. Este problema es particularmente crítico en series de tiempo, donde:

- **Dependencias temporales**: Los valores actuales dependen de valores pasados
- **Patrones estacionales**: Existen ciclos que se repiten (diarios, semanales, anuales)
- **Tendencias**: Las series pueden tener comportamientos de crecimiento o decrecimiento
- **Estacionariedad**: Las propiedades estadísticas pueden cambiar con el tiempo

Los métodos tradicionales de detección de drift, diseñados para datos i.i.d. (independientes e idénticamente distribuidos), no capturan estas características únicas de las series de tiempo.

### 1.3 Objetivos del Sistema

El sistema implementado tiene los siguientes objetivos:

1. **Detectar cambios en distribuciones estadísticas** de features (temperatura, humedad, flujos difusos)
2. **Identificar alteraciones en patrones temporales** (autocorrelación, estacionalidad)
3. **Monitorear degradación del rendimiento** del modelo (RMSE, MAE, MAPE)
4. **Generar alertas automáticas** con niveles de severidad apropiados
5. **Proporcionar recomendaciones accionables** para mantener la calidad del modelo

### 1.4 Características Principales

- **Diseño Orientado a Objetos (OOP)**: Arquitectura modular con clases reutilizables y extensibles
- **Metodologías Específicas para Series de Tiempo**: Tests ADF, análisis ACF, descomposición estacional
- **Sistema de Alertas Multi-Canal**: Consola, archivos JSON, extensible a Slack/email
- **Integración con FastAPI**: Monitoreo en tiempo real durante predicciones
- **Suite Completa de Tests**: 17 tests unitarios con 100% de cobertura
- **Documentación Comprehensiva**: Código completamente documentado con docstrings

---

## 2. Fundamentos Teóricos de Series de Tiempo

### 2.1 ¿Qué es una Serie de Tiempo?

Una serie de tiempo es una secuencia de observaciones indexadas en el tiempo: {y₁, y₂, ..., yₜ}, donde cada observación yₜ está asociada con un tiempo específico t. A diferencia de los datos tabulares tradicionales, las series de tiempo presentan:

**Dependencia Temporal**: El valor en el tiempo t puede depender de valores en t-1, t-2, etc.
```
y_t = f(y_{t-1}, y_{t-2}, ..., y_{t-k}, X_t) + ε_t
```

**Autocorrelación**: La correlación de la serie consigo misma en diferentes rezagos temporales.

**Estacionalidad**: Patrones que se repiten en intervalos regulares (diario, semanal, mensual).

### 2.2 Propiedades Clave de Series de Tiempo

#### 2.2.1 Estacionariedad

Una serie de tiempo es estacionaria si sus propiedades estadísticas (media, varianza, autocorrelación) no cambian con el tiempo:

- **Estacionariedad Débil (de segundo orden)**:
  - E[y_t] = μ (media constante)
  - Var[y_t] = σ² (varianza constante)
  - Cov[y_t, y_{t+h}] = γ_h (autocovarianza que depende solo del rezago h)

- **Estacionariedad Fuerte**: La distribución conjunta de (y_{t1}, y_{t2}, ..., y_{tk}) es invariante ante traslaciones temporales.

**Importancia**: Los modelos de series de tiempo (ARIMA, VAR) requieren estacionariedad. El drift puede manifestarse como pérdida de estacionariedad.

#### 2.2.2 Autocorrelación

La autocorrelación mide la correlación entre la serie y versiones rezagadas de sí misma:

```
ρ_k = Corr(y_t, y_{t-k}) = Cov(y_t, y_{t-k}) / √(Var(y_t) × Var(y_{t-k}))
```

La **Función de Autocorrelación (ACF)** es el conjunto {ρ₁, ρ₂, ..., ρₖ} para diferentes rezagos k.

**Importancia**: Cambios en la ACF indican que la estructura temporal de la serie ha cambiado, lo cual es una señal crítica de drift temporal.

#### 2.2.3 Estacionalidad

La estacionalidad se refiere a patrones que se repiten en intervalos fijos. Una serie con estacionalidad puede descomponerse en:

```
y_t = T_t + S_t + R_t
```

Donde:
- T_t: Componente de tendencia
- S_t: Componente estacional
- R_t: Componente residual (ruido)

**Descomposición STL** (Seasonal-Trend decomposition using LOESS): Método robusto para descomponer series de tiempo que permite:
- Manejo de estacionalidad cambiante
- Robustez ante outliers
- Flexibilidad en el período estacional

**Importancia**: Cambios en el componente estacional indican que los patrones cíclicos del sistema han cambiado.

#### 2.2.4 Tendencia

La tendencia representa el comportamiento de largo plazo de la serie (crecimiento, decrecimiento, o estabilidad). Se puede modelar como:

- **Tendencia Lineal**: T_t = α + βt
- **Tendencia Polinomial**: T_t = α + β₁t + β₂t²
- **Tendencia Suave**: Estimada mediante regresión local (LOESS)

### 2.3 Desafíos Únicos para Detección de Drift en Series de Tiempo

#### 2.3.1 Violación de Independencia

Los métodos tradicionales de drift detection asumen que las observaciones son independientes (i.i.d.). En series de tiempo:

- Las observaciones consecutivas están correlacionadas
- Los tests estadísticos clásicos (como el test de Kolmogorov-Smirnov) pueden producir falsos positivos si no se considera la autocorrelación
- Se requieren ajustes o métodos específicos para series correlacionadas

#### 2.3.2 Tipos de Drift en Series de Tiempo

1. **Drift en Features Temporales**:
   - Cambios en variables exógenas (temperatura, humedad)
   - Cambios en lags y rolling statistics

2. **Drift Temporal**:
   - Cambios en autocorrelación (estructura de dependencia temporal)
   - Cambios en estacionalidad (magnitud o fase de ciclos)
   - Pérdida de estacionariedad

3. **Drift de Concepto**:
   - La relación entre features y target cambia
   - Ejemplo: temperatura deja de ser un predictor efectivo del consumo

4. **Drift de Performance**:
   - Degradación en métricas del modelo (RMSE, MAE, MAPE)
   - Puede ser consecuencia de los otros tipos de drift

#### 2.3.3 Ventanas Temporales

En series de tiempo, la elección de ventanas temporales es crítica:

- **Ventana de Referencia (Baseline)**: Datos históricos usados para entrenar el modelo
- **Ventana de Monitoreo**: Datos recientes en producción
- **Tamaño de Ventana**: Debe ser suficientemente grande para capturar patrones estacionales
  - Para datos con estacionalidad diaria: mínimo 2-3 días
  - Para datos con estacionalidad semanal: mínimo 2-3 semanas

**Para el Proyecto Tetouan**:
- Intervalo de datos: 10 minutos
- Pasos por hora: 6
- Pasos por día: 144
- Ventana de monitoreo: 24 horas (144 pasos)
- Chequeo de drift: Cada 6 horas

### 2.4 Aplicación al Proyecto Tetouan

El proyecto de predicción de consumo eléctrico presenta las siguientes características:

**Frecuencia**: Observaciones cada 10 minutos (144 observaciones por día)

**Variables**:
- Features meteorológicas: temperatura, humedad, wind_speed
- Features de radiación: general_diffuse_flows, diffuse_flows
- Features temporales: hora, día_de_semana, día_del_año
- Features de lags: consumo de 1 hora y 24 horas atrás
- Target: zone_X_power_consumption (X = 1, 2, 3)

**Patrones Esperados**:
- Estacionalidad diaria (consumo mayor en horas pico)
- Estacionalidad semanal (diferencias entre días laborales y fines de semana)
- Correlación con temperatura (uso de calefacción/aire acondicionado)
- Autocorrelación significativa (consumo actual depende del reciente)

---

## 3. Metodologías de Detección de Drift

### 3.1 Detección Estadística de Drift

#### 3.1.1 Test de Kolmogorov-Smirnov (KS Test)

**Fundamento Teórico**:

El test KS compara dos distribuciones empíricas calculando la máxima distancia vertical entre sus funciones de distribución acumulada (CDF):

```
D = sup_x |F_reference(x) - F_current(x)|
```

**Hipótesis**:
- H₀: Las dos muestras provienen de la misma distribución
- H₁: Las distribuciones son diferentes

**Estadístico de Prueba**:
```
D_n,m = sup_x |F_n(x) - G_m(x)|
```

Donde F_n y G_m son las CDFs empíricas de las muestras de tamaño n y m.

**Criterio de Decisión**:
- Si p-value < α (típicamente 0.05): Rechazar H₀ (hay drift)
- Si p-value ≥ α: No rechazar H₀ (no hay evidencia de drift)

**Ventajas**:
- No paramétrico (no asume distribución específica)
- Sensible a cambios en forma, localización y escala
- Ampliamente utilizado en industria

**Limitaciones**:
- Asume independencia de observaciones
- Puede ser conservador con muestras pequeñas
- No indica dónde está la diferencia, solo que existe

**Implementación en el Sistema**:
```python
from scipy.stats import ks_2samp

def _compute_ks_statistic(self, ref_values, curr_values):
    statistic, pvalue = ks_2samp(ref_values, curr_values)
    return {"statistic": statistic, "pvalue": pvalue}
```

#### 3.1.2 Population Stability Index (PSI)

**Fundamento Teórico**:

PSI mide la estabilidad de una distribución comparando la proporción de observaciones en bins discretos entre dos períodos:

```
PSI = Σ(P_current,i - P_reference,i) × ln(P_current,i / P_reference,i)
```

Donde P_reference,i y P_current,i son las proporciones en el bin i.

**Interpretación**:
- PSI < 0.1: Estabilidad (sin drift significativo)
- 0.1 ≤ PSI < 0.2: Drift leve (monitorear)
- PSI ≥ 0.2: Drift significativo (acción requerida)

**Ventajas**:
- Fácil de interpretar (umbral estándar en industria)
- Simétrico (no depende de qué distribución es "referencia")
- Robusto ante outliers (al usar bins)

**Limitaciones**:
- Sensible al número de bins (típicamente 10-20)
- Requiere suficientes observaciones por bin
- No captura cambios dentro de bins

**Implementación en el Sistema**:
```python
def _compute_psi(self, ref_values, curr_values, n_bins=10):
    # Crear bins basados en percentiles de referencia
    bins = np.percentile(ref_values, np.linspace(0, 100, n_bins + 1))
    bins[0] = -np.inf
    bins[-1] = np.inf

    # Calcular proporciones
    ref_props = np.histogram(ref_values, bins=bins)[0] / len(ref_values)
    curr_props = np.histogram(curr_values, bins=bins)[0] / len(curr_values)

    # Evitar división por cero
    ref_props = np.where(ref_props == 0, 0.0001, ref_props)
    curr_props = np.where(curr_props == 0, 0.0001, curr_props)

    # Calcular PSI
    psi = np.sum((curr_props - ref_props) * np.log(curr_props / ref_props))
    return psi
```

#### 3.1.3 Jensen-Shannon Divergence

**Fundamento Teórico**:

La divergencia de Jensen-Shannon es una medida de similitud entre dos distribuciones de probabilidad P y Q:

```
JS(P || Q) = (1/2) × KL(P || M) + (1/2) × KL(Q || M)
```

Donde M = (P + Q) / 2 y KL es la divergencia de Kullback-Leibler.

**Propiedades**:
- Simétrica: JS(P || Q) = JS(Q || P)
- Acotada: 0 ≤ JS ≤ log(2) ≈ 0.693
- JS = 0 si y solo si P = Q

**Interpretación**:
- JS < 0.1: Distribuciones muy similares
- 0.1 ≤ JS < 0.2: Diferencia moderada
- JS ≥ 0.2: Distribuciones significativamente diferentes

**Ventajas**:
- Métrica simétrica y bien definida
- Raíz cuadrada de JS es una métrica (distancia válida)
- Más robusta que KL ante ceros

**Implementación en el Sistema**:
```python
def _compute_js_divergence(self, ref_values, curr_values, n_bins=20):
    # Crear histogramas normalizados
    bins = np.linspace(
        min(ref_values.min(), curr_values.min()),
        max(ref_values.max(), curr_values.max()),
        n_bins
    )

    p, _ = np.histogram(ref_values, bins=bins, density=True)
    q, _ = np.histogram(curr_values, bins=bins, density=True)

    # Normalizar
    p = p / p.sum()
    q = q / q.sum()

    # Calcular distribución promedio
    m = (p + q) / 2

    # Calcular JS
    js = (entropy(p, m) + entropy(q, m)) / 2
    return js
```

### 3.2 Detección de Drift Temporal

#### 3.2.1 Función de Autocorrelación (ACF)

**Fundamento Teórico**:

La ACF mide la correlación lineal entre la serie y sus valores rezagados:

```
ρ_k = Corr(y_t, y_{t-k}) = γ_k / γ_0
```

Donde γ_k es la autocovarianza en el rezago k.

**Detección de Drift**:

Comparamos la ACF de la serie de referencia y la serie actual:

```
Δ_ACF = Σ |ACF_reference(k) - ACF_current(k)| para k = 1, 2, ..., K
```

Si Δ_ACF > threshold (típicamente 0.3), indica cambio en la estructura temporal.

**Implementación**:
```python
from statsmodels.tsa.stattools import acf

def _compute_acf_change(self, ref_series, curr_series, lags=24):
    # Calcular ACF para ambas series
    acf_ref = acf(ref_series, nlags=lags, fft=True)
    acf_curr = acf(curr_series, nlags=lags, fft=True)

    # Calcular diferencia absoluta promedio
    acf_diff = np.mean(np.abs(acf_ref - acf_curr))

    return acf_diff
```

**Interpretación**:
- Δ_ACF < 0.1: Estructura temporal estable
- 0.1 ≤ Δ_ACF < 0.3: Cambio leve
- Δ_ACF ≥ 0.3: Cambio significativo en dependencias temporales

#### 3.2.2 Test de Dickey-Fuller Aumentado (ADF)

**Fundamento Teórico**:

El test ADF prueba la presencia de una raíz unitaria (no estacionariedad) en la serie:

**Modelo de Regresión**:
```
Δy_t = α + βt + γy_{t-1} + Σ δ_i Δy_{t-i} + ε_t
```

**Hipótesis**:
- H₀: γ = 0 (hay raíz unitaria, serie no estacionaria)
- H₁: γ < 0 (no hay raíz unitaria, serie estacionaria)

**Estadístico de Prueba**:
El valor t del coeficiente γ se compara con valores críticos específicos de Dickey-Fuller.

**Interpretación**:
- p-value < 0.05: Rechazar H₀ (serie estacionaria)
- p-value ≥ 0.05: No rechazar H₀ (serie posiblemente no estacionaria)

**Detección de Drift**:

Si la serie de referencia es estacionaria (p < 0.05) pero la serie actual no lo es (p ≥ 0.05), indica pérdida de estacionariedad, lo cual es una señal crítica de drift.

**Implementación**:
```python
from statsmodels.tsa.stattools import adfuller

def _check_stationarity_change(self, ref_series, curr_series):
    # Test ADF para serie de referencia
    adf_ref = adfuller(ref_series, autolag='AIC')
    is_stationary_ref = adf_ref[1] < self.adf_threshold

    # Test ADF para serie actual
    adf_curr = adfuller(curr_series, autolag='AIC')
    is_stationary_curr = adf_curr[1] < self.adf_threshold

    # Detectar cambio en estacionariedad
    stationarity_changed = is_stationary_ref != is_stationary_curr

    return {
        "changed": stationarity_changed,
        "ref_pvalue": adf_ref[1],
        "curr_pvalue": adf_curr[1]
    }
```

#### 3.2.3 Descomposición Estacional

**Fundamento Teórico**:

La descomposición estacional separa una serie de tiempo en tres componentes:

**Modelo Aditivo**:
```
y_t = T_t + S_t + R_t
```

**Modelo Multiplicativo**:
```
y_t = T_t × S_t × R_t
```

**Método STL** (Seasonal-Trend decomposition using LOESS):
- Usa regresión local (LOESS) para estimar tendencia y estacionalidad
- Robusto ante outliers
- Permite estacionalidad variable en el tiempo

**Detección de Drift Estacional**:

Comparamos la magnitud y forma del componente estacional:

```
Δ_seasonal = √(Σ(S_reference,t - S_current,t)²) / n
```

**Implementación**:
```python
from statsmodels.tsa.seasonal import seasonal_decompose

def _compute_seasonal_change(self, ref_series, curr_series):
    # Asegurar suficientes datos (2 períodos estacionales)
    if len(ref_series) < 2 * self.seasonal_period:
        return None

    # Descomponer series
    decomp_ref = seasonal_decompose(
        ref_series,
        model='additive',
        period=self.seasonal_period,
        extrapolate_trend='freq'
    )
    decomp_curr = seasonal_decompose(
        curr_series,
        model='additive',
        period=self.seasonal_period,
        extrapolate_trend='freq'
    )

    # Comparar componentes estacionales
    seasonal_diff = np.sqrt(
        np.mean((decomp_ref.seasonal - decomp_curr.seasonal) ** 2)
    )

    return seasonal_diff
```

### 3.3 Monitoreo de Performance del Modelo

#### 3.3.1 Métricas de Regresión

**Root Mean Squared Error (RMSE)**:
```
RMSE = √(1/n Σ(y_true - y_pred)²)
```

- Penaliza errores grandes (por el cuadrado)
- Mismas unidades que la variable objetivo
- Sensible a outliers

**Mean Absolute Error (MAE)**:
```
MAE = 1/n Σ|y_true - y_pred|
```

- Interpretación intuitiva (error promedio absoluto)
- Menos sensible a outliers que RMSE
- Métrica robusta

**Mean Absolute Percentage Error (MAPE)**:
```
MAPE = (100/n) Σ|( y_true - y_pred) / y_true|
```

- Normalizado (independiente de escala)
- Fácil interpretación (porcentaje de error)
- Problemático cuando y_true cercano a 0

#### 3.3.2 Detección de Degradación

**Enfoque de Ventanas Deslizantes**:

Comparamos métricas en ventanas temporales:

```
degradation = (metric_current - metric_baseline) / metric_baseline
```

**Criterio**:
- Si degradation > threshold (típicamente 0.15 = 15%): Alerta de degradación

**Implementación**:
```python
def _detect_performance_drift(self, baseline_data, current_data):
    # Calcular métricas baseline
    rmse_baseline = np.sqrt(mean_squared_error(
        baseline_data['y_true'],
        baseline_data['y_pred']
    ))

    # Calcular métricas actuales
    rmse_current = np.sqrt(mean_squared_error(
        current_data['y_true'],
        current_data['y_pred']
    ))

    # Calcular degradación relativa
    degradation = (rmse_current - rmse_baseline) / rmse_baseline

    # Generar alerta si excede threshold
    if degradation > self.performance_threshold:
        alert = DriftAlert(
            drift_type=DriftType.PERFORMANCE_DRIFT,
            severity=self._determine_severity(degradation),
            metric_name="RMSE",
            baseline_value=rmse_baseline,
            current_value=rmse_current,
            threshold=self.performance_threshold,
            message=f"RMSE degradó {degradation*100:.1f}%"
        )
        return alert

    return None
```

### 3.4 Determinación de Severidad

El sistema clasifica las alertas en cinco niveles de severidad basándose en múltiples métricas:

**Algoritmo de Severidad**:

```python
def _determine_severity(self, metric_value, threshold, metric_type):
    # Calcular ratio de exceso
    ratio = metric_value / threshold

    if metric_type in ['psi', 'js_divergence', 'performance_degradation']:
        if ratio < 0.5:
            return DriftSeverity.NONE
        elif ratio < 1.0:
            return DriftSeverity.LOW
        elif ratio < 1.25:
            return DriftSeverity.MEDIUM
        elif ratio < 2.0:
            return DriftSeverity.HIGH
        else:
            return DriftSeverity.CRITICAL

    elif metric_type == 'ks_pvalue':
        # Para p-value, invertir lógica (menor p-value = mayor severidad)
        if metric_value > 0.5:
            return DriftSeverity.NONE
        elif metric_value > 0.2:
            return DriftSeverity.LOW
        elif metric_value > 0.05:
            return DriftSeverity.MEDIUM
        elif metric_value > 0.01:
            return DriftSeverity.HIGH
        else:
            return DriftSeverity.CRITICAL
```

**Matriz de Severidad**:

| Métrica | NONE | LOW | MEDIUM | HIGH | CRITICAL |
|---------|------|-----|--------|------|----------|
| PSI | < 0.1 | 0.1-0.2 | 0.2-0.25 | 0.25-0.5 | > 0.5 |
| KS p-value | > 0.5 | 0.2-0.5 | 0.05-0.2 | 0.01-0.05 | < 0.01 |
| JS Divergence | < 0.05 | 0.05-0.1 | 0.1-0.2 | 0.2-0.3 | > 0.3 |
| Degradación | < 5% | 5-15% | 15-25% | 25-50% | > 50% |
| ACF Change | < 0.1 | 0.1-0.2 | 0.2-0.3 | 0.3-0.5 | > 0.5 |

---

## 4. Arquitectura del Sistema

### 4.1 Diseño de Alto Nivel

El sistema sigue una arquitectura modular basada en el patrón Strategy y Observer:

```
┌──────────────────────────────────────────────────────────┐
│           DriftMonitoringPipeline                        │
│  (Orquestador principal - integra detectores y alertas)  │
└────────────────────┬─────────────────────────────────────┘
                     │
     ┌───────────────┼───────────────┐
     │               │               │
     ▼               ▼               ▼
┌─────────────┐ ┌──────────────┐ ┌────────────────────┐
│Statistical  │ │ TimeSeries   │ │ Performance        │
│Drift        │ │ Drift        │ │ Monitor            │
│Detector     │ │ Detector     │ │                    │
│             │ │              │ │                    │
│• KS Test    │ │• ACF         │ │• RMSE              │
│• PSI        │ │• ADF         │ │• MAE               │
│• JS Div     │ │• Seasonal    │ │• MAPE              │
└──────┬──────┘ └──────┬───────┘ └─────────┬──────────┘
       │               │                   │
       └───────────────┼───────────────────┘
                       ▼
              ┌────────────────┐
              │ AlertManager   │
              │ (Enrutador)    │
              └────────┬───────┘
                       │
       ┌───────────────┼───────────────┐
       ▼               ▼               ▼
┌─────────────┐ ┌─────────────┐ ┌──────────────┐
│  Console    │ │    File     │ │   Custom     │
│  Channel    │ │  Channel    │ │  Channels    │
│  (Colored)  │ │   (JSON)    │ │ (Extensible) │
└─────────────┘ └─────────────┘ └──────────────┘
                       │
                       ▼
              ┌────────────────┐
              │ Drift Report   │
              │ + Recomendaciones
              └────────────────┘
```

### 4.2 Estructura de Directorios

```
src/monitoring/
├── drift_detectors.py       # Detectores de drift (550 líneas)
│   ├── DriftDetector (ABC)
│   ├── StatisticalDriftDetector
│   ├── TimeSeriesDriftDetector
│   └── ModelPerformanceMonitor
│
├── alert_system.py          # Sistema de alertas (380 líneas)
│   ├── DriftAlert (dataclass)
│   ├── AlertChannel (ABC)
│   ├── ConsoleAlertChannel
│   ├── FileAlertChannel
│   ├── AlertManager
│   └── DriftMonitoringReport
│
├── drift_pipeline.py        # Pipeline de orquestación (320 líneas)
│   ├── MonitoringDataLoader
│   ├── DriftMonitoringPipeline
│   └── create_default_pipeline()
│
├── drift_visualization.py   # Sistema de visualización (650 líneas)
│   ├── DriftVisualizer
│   └── create_visualizations()
│
└── __init__.py              # API pública (100 líneas)

api/
└── drift_monitor.py         # Integración FastAPI
    ├── PredictionLogger
    ├── RealTimeDriftMonitor
    └── Endpoints (/monitoring/*)

tests/
└── test_monitoring.py       # 17 tests unitarios

examples/
└── drift_monitoring_demo.py # 6 demos (detección + visualización)

scripts/
└── generate_drift_plots.py  # Script CLI para generar visualizaciones
```

### 4.3 Patrones de Diseño Aplicados

#### 4.3.1 Abstract Factory Pattern

La clase base `DriftDetector` define la interfaz que todos los detectores deben implementar:

```python
from abc import ABC, abstractmethod

class DriftDetector(ABC):
    def __init__(self, name: str, thresholds: Dict[str, float]):
        self.name = name
        self.thresholds = thresholds
        self.alerts: List[DriftAlert] = []

    @abstractmethod
    def detect(self, reference_data: pd.DataFrame,
               current_data: pd.DataFrame) -> List[DriftAlert]:
        """Implementado por cada detector específico."""
        pass

    def get_alerts(self) -> List[DriftAlert]:
        return self.alerts

    def clear_alerts(self):
        self.alerts.clear()
```

#### 4.3.2 Strategy Pattern

Los diferentes detectores implementan estrategias específicas de detección:

- `StatisticalDriftDetector`: Estrategia basada en tests estadísticos
- `TimeSeriesDriftDetector`: Estrategia basada en análisis temporal
- `ModelPerformanceMonitor`: Estrategia basada en métricas de performance

#### 4.3.3 Observer Pattern

El sistema de alertas implementa el patrón Observer:

```python
class AlertManager:
    def __init__(self, channels: List[AlertChannel] = None):
        self.channels = channels or []

    def add_channel(self, channel: AlertChannel):
        self.channels.append(channel)

    def send_alerts(self, alerts: List[DriftAlert]) -> Dict[str, int]:
        results = {}
        for channel in self.channels:
            sent = sum(1 for alert in alerts if channel.send_alert(alert))
            results[channel.name] = sent
        return results
```

### 4.4 Principios SOLID Aplicados

**S - Single Responsibility Principle**:
- Cada detector tiene una responsabilidad única (detección estadística, temporal o de performance)
- AlertManager solo gestiona el routing de alertas
- DriftMonitoringPipeline solo orquesta el flujo

**O - Open/Closed Principle**:
- El sistema es abierto a extensión (agregar nuevos detectores) pero cerrado a modificación
- Nuevos detectores heredan de `DriftDetector` sin modificar código existente

**L - Liskov Substitution Principle**:
- Cualquier `DriftDetector` puede sustituirse por otro sin romper el sistema
- Todos los canales de alerta son intercambiables

**I - Interface Segregation Principle**:
- Interfaces pequeñas y específicas (`DriftDetector.detect()`, `AlertChannel.send_alert()`)
- No se fuerza a implementar métodos innecesarios

**D - Dependency Inversion Principle**:
- El pipeline depende de abstracciones (`DriftDetector`), no de implementaciones concretas
- Inyección de dependencias en constructores

---

## 5. Componentes Principales

### 5.1 DriftDetector (Clase Base Abstracta)

```python
class DriftDetector(ABC):
    """
    Clase base abstracta para todos los detectores de drift.

    Define la interfaz común que todos los detectores deben implementar.
    """

    def __init__(self, name: str, thresholds: Dict[str, float]):
        """
        Parameters
        ----------
        name : str
            Nombre identificador del detector
        thresholds : Dict[str, float]
            Umbrales para determinar severidad del drift
        """
        self.name = name
        self.thresholds = thresholds
        self.alerts: List[DriftAlert] = []

    @abstractmethod
    def detect(self, reference_data: pd.DataFrame,
               current_data: pd.DataFrame) -> List[DriftAlert]:
        """
        Detecta drift comparando datos de referencia vs actuales.

        Parameters
        ----------
        reference_data : pd.DataFrame
            Datos históricos (baseline)
        current_data : pd.DataFrame
            Datos recientes en producción

        Returns
        -------
        List[DriftAlert]
            Lista de alertas detectadas
        """
        pass

    def _determine_severity(self, value: float, threshold: float,
                          metric_type: str) -> DriftSeverity:
        """Determina severidad basándose en ratio value/threshold."""
        # Implementación detallada en sección 3.4
        pass
```

### 5.2 StatisticalDriftDetector

Implementa detección de drift basada en cambios en distribuciones estadísticas.

**Parámetros de Configuración**:

```python
detector = StatisticalDriftDetector(
    ks_threshold=0.05,      # Umbral para p-value del KS test
    psi_threshold=0.2,      # Umbral para PSI
    js_threshold=0.1,       # Umbral para JS Divergence
    n_bins=10               # Número de bins para PSI
)
```

**Metodología**:

1. Para cada columna numérica en los datos:
   - Calcula KS test, PSI y JS divergence
   - Compara contra umbrales configurados
   - Genera alerta si alguno excede threshold

2. Determina severidad basándose en la magnitud del cambio

3. Retorna lista de alertas con metadatos completos

**Ejemplo de Uso**:

```python
# Detectar drift en features meteorológicas
alerts = detector.detect(train_data, production_data)

# Filtrar alertas de temperatura
temp_alerts = [a for a in alerts if 'temperature' in a.metric_name]

for alert in temp_alerts:
    print(f"""
    Feature: {alert.metric_name}
    Severidad: {alert.severity.value}
    PSI: {alert.metadata.get('psi', 'N/A'):.3f}
    KS p-value: {alert.metadata.get('ks_pvalue', 'N/A'):.4f}
    """)
```

### 5.3 TimeSeriesDriftDetector

Implementa detección específica para series de tiempo.

**Parámetros de Configuración**:

```python
detector = TimeSeriesDriftDetector(
    seasonal_period=144,    # Período estacional (1 día para datos de 10 min)
    autocorr_lags=24,       # Lags a analizar (4 horas)
    adf_threshold=0.05,     # Umbral para test ADF
    acf_threshold=0.3       # Umbral para cambio en ACF
)
```

**Metodología**:

1. **Análisis de Autocorrelación**:
   - Calcula ACF para ambas series
   - Mide diferencia absoluta promedio
   - Alerta si Δ_ACF > threshold

2. **Test de Estacionariedad (ADF)**:
   - Aplica test ADF a ambas series
   - Detecta pérdida/ganancia de estacionariedad
   - Alerta si status cambia

3. **Análisis de Estacionalidad**:
   - Descompone series usando STL
   - Compara componentes estacionales
   - Alerta si cambio es significativo

**Ejemplo de Uso**:

```python
# Analizar drift temporal en consumo eléctrico
alerts = detector.detect(
    train_data[['zone_1_power_consumption']],
    production_data[['zone_1_power_consumption']]
)

# Clasificar alertas por tipo
acf_alerts = [a for a in alerts if 'autocorr' in a.metric_name]
stationarity_alerts = [a for a in alerts if 'stationarity' in a.metric_name]
seasonal_alerts = [a for a in alerts if 'seasonal' in a.metric_name]
```

### 5.4 ModelPerformanceMonitor

Monitorea degradación en métricas del modelo.

**Parámetros de Configuración**:

```python
monitor = ModelPerformanceMonitor(
    window_size=144,                # Tamaño de ventana (1 día)
    performance_threshold=0.15,     # 15% degradación permitida
    min_samples=100                 # Mínimo de muestras requeridas
)
```

**Metodología**:

1. Requiere columnas `y_true` y `y_pred` en los datos

2. Calcula métricas baseline (RMSE, MAE, MAPE)

3. Calcula métricas actuales usando ventanas deslizantes

4. Detecta degradación relativa:
   ```
   degradation = (metric_current - metric_baseline) / metric_baseline
   ```

5. Genera alertas si degradación > threshold

**Ejemplo de Uso**:

```python
# Datos con predicciones y valores reales
baseline_data = pd.DataFrame({
    'y_true': actual_values_baseline,
    'y_pred': predictions_baseline
})

current_data = pd.DataFrame({
    'y_true': actual_values_current,
    'y_pred': predictions_current
})

# Monitorear performance
alerts = monitor.detect(baseline_data, current_data)

for alert in alerts:
    metric = alert.metric_name
    degradation = (alert.current_value - alert.baseline_value) / alert.baseline_value
    print(f"{metric} degradó {degradation*100:.1f}%")
```

### 5.5 Sistema de Alertas

#### 5.5.1 DriftAlert (Dataclass)

```python
@dataclass
class DriftAlert:
    """
    Representa una alerta de drift detectado.

    Attributes
    ----------
    drift_type : DriftType
        Tipo de drift (FEATURE, TEMPORAL, PERFORMANCE, CONCEPT, LABEL)
    severity : DriftSeverity
        Nivel de severidad (NONE, LOW, MEDIUM, HIGH, CRITICAL)
    metric_name : str
        Nombre de la métrica que generó la alerta
    baseline_value : float
        Valor de referencia (baseline)
    current_value : float
        Valor actual detectado
    threshold : float
        Umbral que fue excedido
    timestamp : datetime
        Momento de la detección
    message : str
        Mensaje descriptivo de la alerta
    metadata : Dict[str, Any]
        Información adicional (métricas detalladas)
    """
    drift_type: DriftType
    severity: DriftSeverity
    metric_name: str
    baseline_value: float
    current_value: float
    threshold: float
    timestamp: datetime = field(default_factory=datetime.now)
    message: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serializa alerta a diccionario."""
        return {
            "drift_type": self.drift_type.value,
            "severity": self.severity.value,
            "metric_name": self.metric_name,
            "baseline_value": self.baseline_value,
            "current_value": self.current_value,
            "threshold": self.threshold,
            "timestamp": self.timestamp.isoformat(),
            "message": self.message,
            "metadata": self.metadata
        }
```

#### 5.5.2 Canales de Alerta

**ConsoleAlertChannel**: Imprime alertas en consola con colores opcionales

```python
class ConsoleAlertChannel(AlertChannel):
    def __init__(self, colored: bool = True,
                 min_severity: DriftSeverity = DriftSeverity.LOW):
        super().__init__("console", min_severity)
        self.colored = colored
        self.color_map = {
            DriftSeverity.LOW: '\033[93m',      # Amarillo
            DriftSeverity.MEDIUM: '\033[33m',   # Naranja
            DriftSeverity.HIGH: '\033[91m',     # Rojo
            DriftSeverity.CRITICAL: '\033[95m'  # Magenta
        }

    def send_alert(self, alert: DriftAlert) -> bool:
        if not self.should_send(alert):
            return False

        color = self.color_map.get(alert.severity, '')
        reset = '\033[0m' if self.colored else ''

        print(f"{color}[{alert.severity.value.upper()}] {alert.message}{reset}")
        return True
```

**FileAlertChannel**: Guarda alertas en archivo JSON

```python
class FileAlertChannel(AlertChannel):
    def __init__(self, output_path: Path, append: bool = True,
                 min_severity: DriftSeverity = DriftSeverity.LOW):
        super().__init__("file", min_severity)
        self.output_path = output_path
        self.append = append
        self.alerts_buffer: List[Dict] = []

    def send_alert(self, alert: DriftAlert) -> bool:
        if not self.should_send(alert):
            return False

        self.alerts_buffer.append(alert.to_dict())
        return True

    def flush(self):
        """Escribe buffer a archivo."""
        if not self.alerts_buffer:
            return

        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        if self.append and self.output_path.exists():
            existing = json.loads(self.output_path.read_text())
            existing.extend(self.alerts_buffer)
            self.output_path.write_text(json.dumps(existing, indent=2))
        else:
            self.output_path.write_text(
                json.dumps(self.alerts_buffer, indent=2)
            )

        self.alerts_buffer.clear()
```

### 5.6 DriftMonitoringPipeline

Orquestador principal que integra detectores y sistema de alertas.

**Configuración**:

```python
# Opción 1: Pipeline por defecto
pipeline = create_default_pipeline(
    output_dir="reports/drift_monitoring"
)

# Opción 2: Pipeline personalizado
pipeline = DriftMonitoringPipeline(
    detectors=[
        StatisticalDriftDetector(psi_threshold=0.15),
        TimeSeriesDriftDetector(seasonal_period=144),
        ModelPerformanceMonitor(performance_threshold=0.10)
    ],
    alert_manager=AlertManager(channels=[
        ConsoleAlertChannel(colored=True),
        FileAlertChannel(Path("alerts.json"))
    ]),
    output_dir=Path("reports/custom")
)
```

**Flujo de Ejecución**:

```python
def run(self, reference_data: pd.DataFrame,
        current_data: pd.DataFrame) -> DriftMonitoringReport:
    """
    Ejecuta pipeline completo de monitoreo.

    Steps:
    1. Validar datos de entrada
    2. Ejecutar cada detector en secuencia
    3. Consolidar alertas
    4. Enviar alertas a canales
    5. Generar reporte
    6. Guardar resultados

    Returns
    -------
    DriftMonitoringReport
        Reporte completo con alertas y recomendaciones
    """
    all_alerts = []

    # Ejecutar detectores
    for detector in self.detectors:
        alerts = detector.detect(reference_data, current_data)
        all_alerts.extend(alerts)

    # Enviar alertas
    if self.alert_manager:
        self.alert_manager.send_alerts(all_alerts)

    # Generar reporte
    report = DriftMonitoringReport(all_alerts)

    # Guardar archivos
    self._save_report(report)

    return report
```

### 5.7 DriftVisualizer

Sistema de visualización integrado para generar gráficos comparativos automáticos de drift.

**Configuración**:

```python
from src.monitoring import DriftVisualizer

visualizer = DriftVisualizer(
    output_dir="reports/drift_monitoring/plots",  # Directorio de salida
    style="seaborn-v0_8-darkgrid",                # Estilo de matplotlib
    figsize=(12, 6)                               # Tamaño por defecto
)
```

**Métodos Principales**:

1. **plot_distribution_comparison**: Compara distribuciones entre baseline y producción
   - Genera 4 subplots: histogramas, box plots, CDF, Q-Q plot
   - Guarda en `distribution_<feature>.png`

2. **plot_timeseries_comparison**: Analiza series temporales
   - Genera 3 subplots: serie temporal, media móvil, ACF
   - Guarda en `timeseries_<feature>.png`

3. **plot_performance_metrics**: Visualiza métricas del modelo
   - Genera 4 subplots: scatter plots, distribución de errores, comparación de métricas
   - Guarda en `performance_comparison.png`

4. **plot_drift_summary**: Resumen visual de alertas
   - Genera 4 subplots: severidad, tipos, top features, timeline
   - Guarda en `drift_summary.png`

5. **create_drift_report_plots**: Genera automáticamente todos los gráficos relevantes
   - Selecciona top N features con mayor drift
   - Genera visualizaciones completas
   - Retorna lista de figuras generadas

**Ejemplo de Uso Completo**:

```python
# Ejecutar detección
pipeline = create_default_pipeline(output_dir="reports/drift")
report = pipeline.run(reference_data, current_data)

# Generar visualizaciones
visualizer = DriftVisualizer(output_dir="reports/drift/plots")

# Opción 1: Generar todo automáticamente
figures = visualizer.create_drift_report_plots(
    reference_data,
    current_data,
    report.alerts,
    top_n_features=5
)

# Opción 2: Gráficos individuales
visualizer.plot_distribution_comparison(
    reference_data, current_data, "temperature"
)
visualizer.plot_timeseries_comparison(
    reference_data, current_data, "zone_1_power_consumption",
    timestamp_col="timestamp"
)
visualizer.plot_performance_metrics(reference_data, current_data)
visualizer.plot_drift_summary(report.alerts)
```

**Características**:

- Alta resolución: Gráficos guardados a 300 DPI
- Estilo profesional: Usando seaborn y matplotlib
- Información completa: Múltiples vistas por cada feature
- Metadatos incluidos: Estadísticas y métricas en los títulos

**Función de Conveniencia**:

```python
from src.monitoring import create_visualizations

# Genera todas las visualizaciones en un solo paso
figures = create_visualizations(
    reference_data=baseline,
    current_data=production,
    alerts=alerts,
    output_dir="reports/drift_monitoring/plots"
)
```

---

## 6. Configuración e Instalación

### 6.1 Requisitos del Sistema

**Python**: 3.8 o superior

**Dependencias Principales**:
- scipy >= 1.10.0 (tests estadísticos)
- statsmodels >= 0.14.0 (análisis de series de tiempo)
- pandas >= 2.0.0 (manipulación de datos)
- numpy >= 1.24.0 (cálculos numéricos)
- scikit-learn >= 1.3.0 (métricas de performance)
- fastapi >= 0.104.0 (API REST)
- pydantic >= 2.0.0 (validación de datos)

### 6.2 Instalación

```bash
# Clonar repositorio
git clone https://github.com/equipo11/mlops-tetouan.git
cd mlops-tetouan

# Crear entorno virtual
python -m venv .venv

# Activar entorno
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# Instalar dependencias
pip install -r requirements.txt

# Instalar proyecto en modo editable
pip install -e .
```

### 6.3 Configuración para el Proyecto Tetouan

**Parámetros Recomendados**:

```python
# Configuración específica para datos de Tetouan
TETOUAN_CONFIG = {
    # Características de los datos
    "data_interval_minutes": 10,
    "steps_per_hour": 6,
    "steps_per_day": 144,

    # Statistical Drift Detector
    "statistical": {
        "ks_threshold": 0.05,
        "psi_threshold": 0.2,
        "js_threshold": 0.1,
        "n_bins": 10
    },

    # Time Series Drift Detector
    "timeseries": {
        "seasonal_period": 144,     # 1 día
        "autocorr_lags": 24,        # 4 horas
        "adf_threshold": 0.05,
        "acf_threshold": 0.3
    },

    # Performance Monitor
    "performance": {
        "window_size": 144,         # 1 día
        "performance_threshold": 0.15,  # 15% degradación
        "min_samples": 100
    },

    # Monitoreo en Tiempo Real
    "realtime": {
        "monitoring_window_hours": 24,
        "check_interval_hours": 6,
        "min_predictions": 100
    }
}
```

**Archivo de Configuración** (`config/drift_config.yaml`):

```yaml
drift_monitoring:
  # Detectores habilitados
  detectors:
    - statistical
    - timeseries
    - performance

  # Umbrales
  thresholds:
    ks_pvalue: 0.05
    psi: 0.2
    js_divergence: 0.1
    acf_change: 0.3
    performance_degradation: 0.15

  # Alertas
  alerts:
    channels:
      - console
      - file
    min_severity: medium
    output_dir: reports/drift_monitoring

  # Series de tiempo
  timeseries:
    seasonal_period: 144
    autocorr_lags: 24
```

### 6.4 Validación de Instalación

```bash
# Ejecutar tests
pytest tests/test_monitoring.py -v

# Ejecutar demo
python examples/drift_monitoring_demo.py

# Verificar imports
python -c "from src.monitoring import create_default_pipeline; print('OK')"
```

---

## 7. Guía de Uso

### 7.1 Uso Básico con Pipeline por Defecto

```python
from src.monitoring import create_default_pipeline
import pandas as pd

# Cargar datos
reference_data = pd.read_parquet("data/processed/train.parquet")
current_data = pd.read_parquet("data/processed/test.parquet")

# Crear pipeline con configuración por defecto
pipeline = create_default_pipeline(output_dir="reports/drift")

# Ejecutar monitoreo
report = pipeline.run(reference_data, current_data)

# Ver resumen
report.print_summary()

# Obtener recomendaciones
recommendations = report.get_recommendations()
for rec in recommendations:
    print(f"- {rec}")
```

### 7.2 Uso Avanzado con Configuración Personalizada

```python
from src.monitoring import (
    DriftMonitoringPipeline,
    StatisticalDriftDetector,
    TimeSeriesDriftDetector,
    ModelPerformanceMonitor,
    AlertManager,
    ConsoleAlertChannel,
    FileAlertChannel,
    DriftSeverity
)
from pathlib import Path

# Configurar detectores con umbrales personalizados
detectors = [
    StatisticalDriftDetector(
        ks_threshold=0.01,      # Más estricto
        psi_threshold=0.15,
        js_threshold=0.08,
        n_bins=20
    ),
    TimeSeriesDriftDetector(
        seasonal_period=144,
        autocorr_lags=48,       # Analizar 8 horas
        adf_threshold=0.05,
        acf_threshold=0.25
    ),
    ModelPerformanceMonitor(
        window_size=288,        # Ventana de 2 días
        performance_threshold=0.10,  # 10% degradación
        min_samples=150
    )
]

# Configurar canales de alerta
alert_channels = [
    ConsoleAlertChannel(
        colored=True,
        min_severity=DriftSeverity.MEDIUM
    ),
    FileAlertChannel(
        output_path=Path("reports/alerts_critical.json"),
        append=True,
        min_severity=DriftSeverity.HIGH
    )
]

# Crear manager de alertas
alert_manager = AlertManager(channels=alert_channels)

# Crear pipeline personalizado
pipeline = DriftMonitoringPipeline(
    detectors=detectors,
    alert_manager=alert_manager,
    output_dir=Path("reports/custom_drift")
)

# Ejecutar
report = pipeline.run(reference_data, current_data)

# Análisis detallado
summary = report.get_summary()
print(f"Total alertas: {summary['total_alerts']}")
print(f"Alertas críticas: {summary['severity_breakdown'].get('critical', 0)}")
print(f"Acción requerida: {summary['requires_action']}")
```

### 7.3 Uso de Detectores Individuales

```python
from src.monitoring import StatisticalDriftDetector

# Crear detector
detector = StatisticalDriftDetector(psi_threshold=0.15)

# Detectar drift
alerts = detector.detect(baseline_data, production_data)

# Procesar alertas
for alert in alerts:
    print(f"""
    Métrica: {alert.metric_name}
    Tipo: {alert.drift_type.value}
    Severidad: {alert.severity.value}
    Baseline: {alert.baseline_value:.4f}
    Actual: {alert.current_value:.4f}
    Mensaje: {alert.message}

    Detalles:
    - PSI: {alert.metadata.get('psi', 'N/A')}
    - KS p-value: {alert.metadata.get('ks_pvalue', 'N/A')}
    - JS Divergence: {alert.metadata.get('js_divergence', 'N/A')}
    """)
```

### 7.4 Monitoreo desde Archivos

```python
# Ejecutar desde archivos Parquet
report = pipeline.run_from_files(
    reference_path="data/processed/train.parquet",
    current_path="data/processed/test.parquet"
)

# También soporta CSV
report = pipeline.run_from_files(
    reference_path="data/baseline.csv",
    current_path="data/production.csv"
)
```

### 7.5 Análisis de Reportes

```python
# Obtener resumen estructurado
summary = report.get_summary()

print(f"Total de alertas: {summary['total_alerts']}")
print(f"Alertas críticas: {summary['has_critical_alerts']}")
print(f"Alertas altas: {summary['has_high_alerts']}")
print(f"Requiere acción: {summary['requires_action']}")

# Desglose por severidad
for severity, count in summary['severity_breakdown'].items():
    print(f"{severity}: {count}")

# Desglose por tipo de drift
for drift_type, count in summary['drift_type_breakdown'].items():
    print(f"{drift_type}: {count}")

# Obtener alertas específicas
critical_alerts = [a for a in report.alerts
                  if a.severity == DriftSeverity.CRITICAL]

feature_drift_alerts = [a for a in report.alerts
                       if a.drift_type == DriftType.FEATURE_DRIFT]

# Filtrar por métrica
temperature_alerts = [a for a in report.alerts
                     if 'temperature' in a.metric_name]
```

### 7.6 Generación de Visualizaciones

El sistema incluye capacidades integradas de visualización para generar gráficos comparativos automáticos:

```python
from src.monitoring import DriftVisualizer, create_default_pipeline
import pandas as pd

# Ejecutar detección de drift
pipeline = create_default_pipeline(output_dir="reports/drift")
report = pipeline.run(reference_data, current_data)

# Crear visualizador
visualizer = DriftVisualizer(output_dir="reports/drift/plots")

# Generar todas las visualizaciones automáticamente
figures = visualizer.create_drift_report_plots(
    reference_data,
    current_data,
    report.alerts,
    top_n_features=5  # Top 5 features con mayor drift
)

print(f"Gráficos generados: {len(figures)}")
# Output: Gráficos generados: 8
```

**Tipos de gráficos generados**:

1. **Distribuciones** (`distribution_<feature>.png`):
   - Histogramas superpuestos (baseline vs producción)
   - Box plots comparativos
   - Función de Distribución Acumulada (CDF)
   - Q-Q Plot

2. **Series Temporales** (`timeseries_<feature>.png`):
   - Comparación temporal de valores
   - Medias móviles para identificar tendencias
   - Función de Autocorrelación (ACF)

3. **Performance del Modelo** (`performance_comparison.png`):
   - Scatter plots de predicciones vs valores reales
   - Distribución de errores
   - Comparación de métricas (RMSE, MAE, MAPE)

4. **Resumen de Drift** (`drift_summary.png`):
   - Distribución de alertas por severidad
   - Distribución por tipo de drift
   - Top features con mayor drift
   - Timeline de severidad

**Visualizaciones individuales**:

```python
# Distribución de una feature específica
visualizer.plot_distribution_comparison(
    reference_data,
    current_data,
    "temperature"
)

# Serie temporal
visualizer.plot_timeseries_comparison(
    reference_data,
    current_data,
    "zone_1_power_consumption",
    timestamp_col="timestamp"
)

# Performance del modelo (requiere y_true y y_pred)
visualizer.plot_performance_metrics(
    reference_data,
    current_data
)

# Resumen de alertas
visualizer.plot_drift_summary(report.alerts)
```

**Script de generación rápida**:

```bash
# Generar visualizaciones con datos reales
python generate_drift_plots.py \
  --reference data/processed/train.parquet \
  --current data/processed/test.parquet \
  --output reports/drift_monitoring \
  --top-features 5

# Demo con datos sintéticos
python generate_drift_plots.py --demo
```

Todos los gráficos se guardan en formato PNG a 300 DPI, listos para incluir en reportes.

Para más detalles, consultar [DRIFT_VISUALIZATION.md](DRIFT_VISUALIZATION.md).

---

## 8. Integración con FastAPI

### 8.1 Arquitectura de Integración

El sistema de drift monitoring está completamente integrado con la API de predicción FastAPI:

```
┌─────────────────────────────────────────────────────────┐
│ FastAPI Application (api/main.py)                      │
└─────────────────────┬───────────────────────────────────┘
                      │
      ┌───────────────┼───────────────┐
      ▼               ▼               ▼
┌────────────┐ ┌─────────────┐ ┌──────────────────┐
│ /predict   │ │ /monitoring │ │ /monitoring      │
│            │ │ /actual     │ │ /drift/status    │
└──────┬─────┘ └──────┬──────┘ └────────┬─────────┘
       │              │                 │
       └──────────────┼─────────────────┘
                      ▼
            ┌──────────────────┐
            │ PredictionLogger │
            │ (JSONL files)    │
            └─────────┬────────┘
                      │
                      ▼
            ┌──────────────────────┐
            │ RealTimeDriftMonitor │
            │ • Carga logs         │
            │ • Ejecuta pipeline   │
            │ • Genera alertas     │
            └──────────────────────┘
```

### 8.2 Componentes de la Integración

#### 8.2.1 PredictionLogger

Registra cada predicción en archivos JSONL para posterior análisis:

```python
class PredictionLogger:
    """
    Logger de predicciones para monitoreo de drift.

    Guarda predicciones en formato JSONL (JSON Lines) para facilitar
    procesamiento incremental y análisis de ventanas temporales.
    """

    def __init__(self, log_dir: Path = Path("logs/predictions")):
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def log_prediction(self, zone: int, model_type: str,
                      features: Dict[str, float],
                      prediction: float) -> None:
        """
        Registra una predicción.

        Parameters
        ----------
        zone : int
            Zona predicha (1, 2, 3)
        model_type : str
            Tipo de modelo usado (RandomForest, XGBoost, etc.)
        features : Dict[str, float]
            Features usadas para la predicción
        prediction : float
            Valor predicho
        """
        log_file = self.log_dir / f"zone_{zone}_{model_type}.jsonl"

        entry = {
            "timestamp": datetime.now().isoformat(),
            "zone": zone,
            "model_type": model_type,
            "features": features,
            "prediction": prediction
        }

        # Append mode para escritura incremental
        with open(log_file, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def log_actual(self, zone: int, actual_value: float,
                  timestamp: datetime) -> None:
        """Registra valor real observado."""
        actuals_file = self.log_dir / f"actuals_zone_{zone}.jsonl"

        entry = {
            "timestamp": timestamp.isoformat(),
            "zone": zone,
            "actual_value": actual_value
        }

        with open(actuals_file, "a") as f:
            f.write(json.dumps(entry) + "\n")
```

#### 8.2.2 RealTimeDriftMonitor

Monitorea drift en tiempo real usando logs de predicciones:

```python
class RealTimeDriftMonitor:
    """
    Monitor de drift en tiempo real para producción.

    Lee logs de predicciones, compara ventanas temporales y genera
    alertas automáticamente.
    """

    def __init__(self,
                 log_dir: Path = Path("logs/predictions"),
                 monitoring_window_hours: int = 24,
                 check_interval_hours: int = 6):
        self.log_dir = log_dir
        self.monitoring_window = timedelta(hours=monitoring_window_hours)
        self.check_interval = timedelta(hours=check_interval_hours)
        self.last_check: Dict[str, datetime] = {}
        self.pipeline = create_default_pipeline()

    def should_check_drift(self, zone: int, model_type: str) -> bool:
        """Determina si es tiempo de chequear drift."""
        key = f"{zone}_{model_type}"
        last = self.last_check.get(key)

        if last is None:
            return True

        return datetime.now() - last >= self.check_interval

    def check_drift(self, zone: int, model_type: str) -> Optional[DriftMonitoringReport]:
        """
        Ejecuta chequeo de drift para zona y modelo específicos.

        Workflow:
        1. Cargar baseline (datos de entrenamiento)
        2. Cargar ventana de monitoreo (últimas N horas)
        3. Ejecutar pipeline de drift detection
        4. Generar y guardar reporte
        """
        # Cargar baseline
        baseline_data = self._load_baseline(zone)

        # Cargar ventana de monitoreo
        monitoring_data = self._load_monitoring_window(zone, model_type)

        if len(monitoring_data) < 100:
            logger.warning(f"Insuficientes datos para drift check: {len(monitoring_data)}")
            return None

        # Ejecutar pipeline
        report = self.pipeline.run(baseline_data, monitoring_data)

        # Actualizar timestamp de último chequeo
        key = f"{zone}_{model_type}"
        self.last_check[key] = datetime.now()

        return report

    def _load_monitoring_window(self, zone: int, model_type: str) -> pd.DataFrame:
        """Carga datos de la ventana de monitoreo."""
        log_file = self.log_dir / f"zone_{zone}_{model_type}.jsonl"

        if not log_file.exists():
            return pd.DataFrame()

        # Leer todas las líneas
        records = []
        with open(log_file) as f:
            for line in f:
                records.append(json.loads(line))

        df = pd.DataFrame(records)
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        # Filtrar ventana de tiempo
        cutoff = datetime.now() - self.monitoring_window
        df = df[df['timestamp'] >= cutoff]

        # Expandir features a columnas
        features_df = pd.json_normalize(df['features'])
        df = pd.concat([df.drop('features', axis=1), features_df], axis=1)

        return df
```

### 8.3 Endpoints de la API

#### 8.3.1 POST /predict (Modificado)

```python
@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Realiza predicción y registra automáticamente para monitoreo.
    """
    # Cargar modelo
    model = load_model(request.zone, request.model_type)

    # Preparar features
    X = prepare_features(request.features)

    # Predecir
    prediction = model.predict(X)[0]

    # Registrar para monitoreo de drift
    pred_logger.log_prediction(
        zone=request.zone,
        model_type=request.model_type,
        features=request.features.dict(),
        prediction=float(prediction)
    )

    return PredictionResponse(
        zone=request.zone,
        model_type=request.model_type,
        prediction=float(prediction),
        timestamp=datetime.now().isoformat()
    )
```

#### 8.3.2 POST /monitoring/actual

```python
@app.post("/monitoring/actual", response_model=ActualValueResponse)
async def log_actual_value(request: ActualValueRequest):
    """
    Registra valor real observado para comparar con predicciones.
    """
    pred_logger.log_actual(
        zone=request.zone,
        actual_value=request.actual_value,
        timestamp=request.timestamp
    )

    return ActualValueResponse(
        status="success",
        message="Valor real registrado exitosamente",
        zone=request.zone,
        actual_value=request.actual_value,
        timestamp=request.timestamp.isoformat()
    )
```

#### 8.3.3 GET /monitoring/drift/status

```python
@app.get("/monitoring/drift/status", response_model=DriftStatusResponse)
async def get_drift_status(zone: int):
    """Obtiene estado del monitoreo de drift del modelo champion."""
    status_payload = drift_monitor.get_drift_status(
        zone=zone,
        model_type="Champion"
    )
    status_payload["model_type"] = "Champion"
    return DriftStatusResponse(**status_payload)
```

#### 8.3.4 POST /monitoring/drift/check

```python
@app.post("/monitoring/drift/check", response_model=DriftCheckResponse)
async def manual_drift_check(zone: int):
    """Ejecuta chequeo manual de drift usando el modelo champion."""
    report = drift_monitor.check_drift(zone, model_type="Champion")
    if report is None:
        return DriftCheckResponse(
            status="insufficient_data",
            message="Datos insuficientes para chequeo de drift",
            zone=zone,
            model_type="Champion"
        )
    summary = report.get_summary()
    recommendations = report.get_recommendations()
    return DriftCheckResponse(
        status="success",
        message="Chequeo de drift completado",
        zone=zone,
        model_type="Champion",
        summary=summary,
        recommendations=recommendations
    )
```

### 8.4 Workflow de Monitoreo en Producción

```
┌─────────────────────────────────────────────────────────┐
│ 1. PREDICCIÓN                                           │
│    POST /predict                                        │
│    → Auto-logging de features + predicción              │
│    → Archivo: logs/predictions/zone_1_Champion.jsonl    │
└──────────────────┬──────────────────────────────────────┘
                   ▼
┌─────────────────────────────────────────────────────────┐
│ 2. OBSERVACIÓN (opcional, mejora monitoreo)             │
│    POST /monitoring/actual                              │
│    → Registrar valor real observado                     │
│    → Archivo: logs/predictions/actuals_zone_1.jsonl     │
└──────────────────┬──────────────────────────────────────┘
                   ▼
┌─────────────────────────────────────────────────────────┐
│ 3. ACUMULACIÓN                                          │
│    PredictionLogger guarda en JSONL                     │
│    → Append mode (eficiente)                            │
│    → Un archivo por zona y modelo                       │
└──────────────────┬──────────────────────────────────────┘
                   ▼
┌─────────────────────────────────────────────────────────┐
│ 4. CHEQUEO AUTOMÁTICO (cada 6 horas)                   │
│    RealTimeDriftMonitor.check_drift()                   │
│    → Carga baseline (datos de entrenamiento)            │
│    → Carga ventana de monitoreo (últimas 24h)          │
│    → Ejecuta DriftMonitoringPipeline                    │
│    → Compara distribuciones y patrones temporales       │
└──────────────────┬──────────────────────────────────────┘
                   ▼
┌─────────────────────────────────────────────────────────┐
│ 5. GENERACIÓN DE ALERTAS                               │
│    Si drift detectado > threshold:                      │
│    → Alertas a consola (logs de API)                    │
│    → Alertas a archivo JSON                             │
│    → (Extensible) Alertas a Slack/Email                 │
└──────────────────┬──────────────────────────────────────┘
                   ▼
┌─────────────────────────────────────────────────────────┐
│ 6. ACCIÓN RECOMENDADA                                  │
│    Basada en severidad:                                 │
│    • CRITICAL → Reentrenar inmediatamente               │
│    • HIGH → Planear reentrenamiento en 24-48h          │
│    • MEDIUM → Investigar causa del drift                │
│    • LOW → Monitorear de cerca                          │
└─────────────────────────────────────────────────────────┘
```

---

## 9. Tests y Validación

### 9.1 Suite de Tests Unitarios

El sistema incluye 17 tests unitarios que cubren todos los componentes:

```bash
pytest tests/test_monitoring.py -v
```

**Resultado**:
```
============================= test session starts =============================
tests/test_monitoring.py::test_statistical_detector_initialization PASSED [ 5%]
tests/test_monitoring.py::test_statistical_detector_detects_drift PASSED [ 11%]
tests/test_monitoring.py::test_statistical_detector_no_drift_on_same_data PASSED [ 17%]
tests/test_monitoring.py::test_timeseries_detector_initialization PASSED [ 23%]
tests/test_monitoring.py::test_timeseries_detector_detects_pattern_drift PASSED [ 29%]
tests/test_monitoring.py::test_timeseries_detector_handles_short_series PASSED [ 35%]
tests/test_monitoring.py::test_performance_monitor_initialization PASSED [ 41%]
tests/test_monitoring.py::test_performance_monitor_detects_degradation PASSED [ 47%]
tests/test_monitoring.py::test_performance_monitor_no_degradation_on_same_data PASSED [ 52%]
tests/test_monitoring.py::test_console_alert_channel PASSED              [ 58%]
tests/test_monitoring.py::test_file_alert_channel PASSED                 [ 64%]
tests/test_monitoring.py::test_alert_manager PASSED                      [ 70%]
tests/test_monitoring.py::test_pipeline_initialization PASSED            [ 76%]
tests/test_monitoring.py::test_pipeline_run PASSED                       [ 82%]
tests/test_monitoring.py::test_pipeline_run_from_files PASSED            [ 88%]
tests/test_monitoring.py::test_drift_monitoring_report_summary PASSED    [ 94%]
tests/test_monitoring.py::test_drift_monitoring_report_recommendations PASSED [100%]

============================= 17 passed in 3.2s ===============================
```

### 9.2 Cobertura de Tests

**Detectores**:
- Inicialización con parámetros correctos
- Detección de drift con datos sintéticos
- No detección cuando datos son idénticos
- Manejo de casos edge (series cortas, valores faltantes)

**Sistema de Alertas**:
- Filtrado por severidad mínima
- Envío a múltiples canales
- Serialización y deserialización

**Pipeline**:
- Orquestación completa de detectores
- Generación de reportes
- Carga desde archivos
- Recomendaciones automáticas

### 9.3 Tests de Integración

```python
def test_integration_full_workflow():
    """
    Test de integración del flujo completo:
    1. Generar datos sintéticos
    2. Introducir drift
    3. Ejecutar pipeline
    4. Verificar alertas
    5. Validar recomendaciones
    """
    # Generar datos baseline
    baseline = generate_timeseries_data(n=500, seed=42)

    # Introducir drift
    drifted = introduce_drift(baseline,
                             temp_shift=5,
                             humidity_scale=0.8)

    # Crear pipeline
    pipeline = create_default_pipeline()

    # Ejecutar
    report = pipeline.run(baseline, drifted)

    # Validaciones
    assert len(report.alerts) > 0, "Debe detectar drift"
    assert report.get_summary()['requires_action'], "Debe requerir acción"
    assert len(report.get_recommendations()) > 0, "Debe generar recomendaciones"

    # Verificar tipos de drift detectados
    drift_types = {a.drift_type for a in report.alerts}
    assert DriftType.FEATURE_DRIFT in drift_types, "Debe detectar feature drift"
```

### 9.4 Tests de Performance

```python
def test_performance_large_dataset():
    """Verifica que el sistema maneja datasets grandes eficientemente."""
    import time

    # Generar dataset grande
    large_data = generate_timeseries_data(n=10000)

    # Medir tiempo de ejecución
    start = time.time()
    pipeline = create_default_pipeline()
    report = pipeline.run(large_data[:5000], large_data[5000:])
    elapsed = time.time() - start

    # Validar tiempo razonable (< 30 segundos)
    assert elapsed < 30, f"Ejecución demasiado lenta: {elapsed:.2f}s"

    # Validar resultados
    assert report is not None
    assert len(report.alerts) >= 0
```

---

## 10. Interpretación de Resultados

### 10.1 Niveles de Severidad

| Severidad | PSI | KS p-value | JS Div | Degradación | Acción Recomendada |
|-----------|-----|------------|--------|-------------|-------------------|
| NONE | < 0.1 | > 0.5 | < 0.05 | < 5% | Sin acción necesaria |
| LOW | 0.1-0.2 | 0.2-0.5 | 0.05-0.1 | 5-15% | Monitorear de cerca |
| MEDIUM | 0.2-0.25 | 0.05-0.2 | 0.1-0.2 | 15-25% | Investigar causa del drift |
| HIGH | 0.25-0.5 | 0.01-0.05 | 0.2-0.3 | 25-50% | Planear reentrenamiento (24-48h) |
| CRITICAL | > 0.5 | < 0.01 | > 0.3 | > 50% | Reentrenar inmediatamente |

### 10.2 Tipos de Drift

**FEATURE_DRIFT**: Cambio en la distribución de variables de entrada

*Ejemplo*:
- Métrica: temperature
- PSI: 0.28 (HIGH)
- Baseline: μ=22.5°C, σ=3.2
- Actual: μ=27.8°C, σ=4.1
- Interpretación: La temperatura promedio subió 5.3°C, lo cual puede afectar las predicciones de consumo

*Causas Posibles*:
- Cambio estacional (verano → invierno)
- Cambio climático de largo plazo
- Sensor descalibrado
- Cambio en fuente de datos

*Acción*:
- Verificar fuente de datos
- Validar calibración de sensores
- Considerar reentrenamiento con datos recientes

**TEMPORAL_DRIFT**: Cambio en patrones temporales (autocorrelación, estacionalidad)

*Ejemplo*:
- Métrica: autocorr_change
- Δ_ACF: 0.42 (HIGH)
- Interpretación: La estructura de dependencias temporales cambió significativamente

*Causas Posibles*:
- Cambio en patrones de consumo (horario laboral modificado)
- Eventos especiales (festividades, lockdowns)
- Cambio en comportamiento de usuarios

*Acción*:
- Investigar eventos externos
- Analizar ACF visual mente
- Considerar actualizar features temporales

**PERFORMANCE_DRIFT**: Degradación en métricas del modelo

*Ejemplo*:
- Métrica: RMSE
- Baseline: 850.2 kW
- Actual: 1105.3 kW
- Degradación: 30% (HIGH)
- Interpretación: El modelo está cometiendo errores significativamente mayores

*Causas Posibles*:
- Feature drift o temporal drift no detectados previamente
- Concept drift (relación features-target cambió)
- Datos de baja calidad

*Acción*:
- Analizar errores por segmento (hora del día, día de semana)
- Verificar otros tipos de drift
- Reentrenar modelo urgentemente

**CONCEPT_DRIFT**: Cambio en la relación features-target

*Ejemplo*:
- La correlación entre temperatura y consumo cambió de 0.75 a 0.42
- Variables que eran predictivas dejaron de serlo

*Causas Posibles*:
- Cambios en comportamiento del sistema
- Nuevos factores no capturados
- Instalación de sistemas de eficiencia energética

*Acción*:
- Análisis de correlaciones
- Feature engineering
- Reentrenamiento con re-selección de features

### 10.3 Recomendaciones Automáticas

El sistema genera recomendaciones basadas en el conjunto de alertas:

```python
def get_recommendations(self) -> List[str]:
    """Genera recomendaciones accionables basadas en alertas."""
    recommendations = []
    summary = self.get_summary()

    # Alertas críticas
    if summary['has_critical_alerts']:
        recommendations.append(
            "CRÍTICO: Reentrenamiento del modelo requerido inmediatamente. "
            "El drift detectado es severo y está afectando significativamente "
            "el rendimiento."
        )

    # Alertas altas
    elif summary['has_high_alerts']:
        recommendations.append(
            "URGENTE: Programar reentrenamiento del modelo dentro de 24-48 horas. "
            "El drift es significativo y continuará degradando el rendimiento."
        )

    # Feature drift
    feature_drift_alerts = [
        a for a in self.alerts
        if a.drift_type == DriftType.FEATURE_DRIFT
    ]
    if len(feature_drift_alerts) > 3:
        features = ", ".join(
            set(a.metric_name for a in feature_drift_alerts[:5])
        )
        recommendations.append(
            f"Feature drift detectado en múltiples variables ({features}). "
            "Verificar procesos de recolección de datos y calibración de sensores."
        )

    # Temporal drift
    temporal_alerts = [
        a for a in self.alerts
        if a.drift_type == DriftType.TEMPORAL_DRIFT
    ]
    if temporal_alerts:
        recommendations.append(
            "Los patrones temporales han cambiado (autocorrelación, estacionalidad). "
            "Considerar re-evaluar features temporales y lags utilizados."
        )

    # Performance drift
    performance_alerts = [
        a for a in self.alerts
        if a.drift_type == DriftType.PERFORMANCE_DRIFT
    ]
    if performance_alerts:
        recommendations.append(
            "Degradación en métricas del modelo detectada. "
            "Analizar errores por segmento para identificar patrones."
        )

    return recommendations
```

### 10.4 Análisis de Casos Específicos

**Caso 1: PSI Alto en Temperatura**

```
Alerta:
  Métrica: temperature_psi
  Valor PSI: 0.31
  Severidad: HIGH

Análisis:
1. Calcular estadísticas descriptivas:
   - Baseline: μ=22.5, σ=3.2, min=15.2, max=29.8
   - Actual: μ=27.8, σ=4.1, min=19.5, max=35.6

2. Visualizar distribuciones:
   - Histogramas superpuestos
   - Q-Q plot
   - CDF empírica

3. Investigar causas:
   - ¿Cambio estacional?
   - ¿Cambio en sensor?
   - ¿Cambio en fuente de datos?

4. Decisión:
   - Si es estacional y esperado: Monitorear
   - Si es inesperado: Investigar y posiblemente reentrenar
```

**Caso 2: Pérdida de Estacionariedad**

```
Alerta:
  Métrica: stationarity_change
  ADF p-value baseline: 0.01 (estacionaria)
  ADF p-value actual: 0.18 (no estacionaria)
  Severidad: HIGH

Análisis:
1. Visualizar serie temporal:
   - Gráfico de línea
   - Identificar tendencia
   - Identificar cambio en varianza

2. Descomponer serie:
   - STL decomposition
   - Analizar componente de tendencia
   - Verificar si es trend o drift

3. Investigar causa:
   - ¿Crecimiento sostenido del consumo?
   - ¿Cambio en comportamiento?

4. Acción:
   - Diferenciar serie si es tendencia real
   - Reentrenar con datos más recientes
   - Considerar modelos que manejen no-estacionariedad
```

**Caso 3: Degradación de Performance sin Drift de Features**

```
Situación:
  - RMSE degradó 25% (HIGH)
  - PSI de features < 0.1 (NONE)
  - No cambios en autocorrelación

Hipótesis:
  Concept drift - La relación features-target cambió

Análisis:
1. Correlaciones:
   - Comparar matriz de correlación baseline vs actual
   - Identificar correlaciones que cambiaron

2. Feature importance:
   - Re-calcular feature importance en datos actuales
   - Comparar con baseline

3. Análisis de residuos:
   - Graficar residuos vs features
   - Identificar patrones no capturados

Acción:
  - Feature re-selection
  - Feature engineering adicional
  - Reentrenamiento obligatorio
```

---

## 11. Casos de Uso y Ejemplos

### 11.1 Demo Completo

El sistema incluye un script de demostración interactivo:

```bash
python examples/drift_monitoring_demo.py
```

**Demos Incluidos** (detección + visualización integrada):

1. **Demo 1 - Detección Básica + Visualizaciones**: Pipeline por defecto con gráficos de distribución
2. **Demo 2 - Análisis de Series Temporales**: Detección temporal + visualizaciones de series de tiempo
3. **Demo 3 - Monitoreo de Performance**: Degradación del modelo + gráficos de métricas
4. **Demo 4 - Pipeline Completo con Reporte Visual**: Todos los detectores + reporte visual completo
5. **Demo 5 - Configuración Personalizada**: Detectores custom + visualizaciones selectivas
6. **Demo 6 - Workflow Basado en Archivos**: Pipeline completo desde archivos + visualizaciones

Cada demo combina detección de drift con generación automática de visualizaciones:
- Gráficos de distribución (histogramas, box plots, CDF, Q-Q plots)
- Series temporales (con media móvil y ACF)
- Métricas de performance (scatter plots, errores, comparación)
- Resumen de alertas (severidad, tipos, timeline)

### 11.2 Ejemplo: Monitoreo Batch (Offline)

Escenario: Comparar datos de training vs datos de producción recientes

```python
from src.monitoring import create_default_pipeline
import pandas as pd

# Cargar datos
train_data = pd.read_parquet("data/processed/train.parquet")
production_data = pd.read_parquet("data/production/last_week.parquet")

# Crear pipeline
pipeline = create_default_pipeline(
    output_dir="reports/weekly_drift_check"
)

# Ejecutar monitoreo
report = pipeline.run(train_data, production_data)

# Análisis
summary = report.get_summary()

if summary['requires_action']:
    print("ACCIÓN REQUERIDA: Drift detectado")

    # Detalles
    print(f"Total de alertas: {summary['total_alerts']}")

    # Severidad
    for severity, count in summary['severity_breakdown'].items():
        print(f"  {severity}: {count}")

    # Recomendaciones
    print("\nRecomendaciones:")
    for rec in report.get_recommendations():
        print(f"  - {rec}")

    # Alertas críticas
    critical = [a for a in report.alerts
               if a.severity == DriftSeverity.CRITICAL]

    if critical:
        print("\nALERTAS CRÍTICAS:")
        for alert in critical:
            print(f"  - {alert.metric_name}: {alert.message}")
else:
    print("Sin drift significativo detectado")
```

### 11.3 Ejemplo: Monitoreo en Tiempo Real con API

Escenario: Usar la API para monitorear drift durante producción

```python
import requests
import time

API_URL = "http://localhost:8000"

# 1. Realizar predicciones (normalmente desde aplicación cliente)
for i in range(100):
    response = requests.post(
        f"{API_URL}/predict",
        json={
            "features": {
                "temperature": 23.5 + np.random.randn(),
                "humidity": 65.0 + np.random.randn() * 5,
                "wind_speed": 5.2 + np.random.randn() * 1,
                # ... más features
            }
        }
    )
    prediction = response.json()

    # Simular observación de valor real (en producción,
    # esto vendría después cuando se observa el consumo real)
    time.sleep(600)  # Esperar 10 minutos

    actual_value = prediction['prediction'] + np.random.randn() * 100

    requests.post(
        f"{API_URL}/monitoring/actual",
        json={
            "zone": 1,
            "actual_value": actual_value,
            "timestamp": datetime.now().isoformat()
        }
    )

# 2. Verificar estado de drift (el modelo champion se usa automáticamente)
response = requests.get(
    f"{API_URL}/monitoring/drift/status",
    params={"zone": 1}
)

status = response.json()
print(f"Necesita chequeo: {status['needs_drift_check']}")
print(f"Próximo chequeo en: {status['next_check_in_hours']:.1f} horas")

if status['latest_report_summary']:
    summary = status['latest_report_summary']
    print(f"Última revisión - Alertas: {summary['total_alerts']}")

# 3. Ejecutar chequeo manual si es necesario
if status['needs_drift_check']:
    response = requests.post(
        f"{API_URL}/monitoring/drift/check",
        params={"zone": 1}
    )

    result = response.json()

    if result['status'] == 'success':
        print("Chequeo completado:")
        print(f"  Total alertas: {result['summary']['total_alerts']}")
        print(f"  Acción requerida: {result['summary']['requires_action']}")

        if result['recommendations']:
            print("Recomendaciones:")
            for rec in result['recommendations']:
                print(f"  - {rec}")
```

### 11.4 Ejemplo: Análisis de Autocorrelación

Escenario: Investigar cambios en la estructura temporal del consumo eléctrico

```python
from src.monitoring import TimeSeriesDriftDetector
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf

# Crear detector
detector = TimeSeriesDriftDetector(
    seasonal_period=144,
    autocorr_lags=144,  # Analizar 1 día completo
    acf_threshold=0.2
)

# Preparar datos
baseline = train_data[['zone_1_power_consumption']].values.flatten()
current = production_data[['zone_1_power_consumption']].values.flatten()

# Detectar drift temporal
alerts = detector.detect(
    pd.DataFrame({'zone_1_power_consumption': baseline}),
    pd.DataFrame({'zone_1_power_consumption': current})
)

# Filtrar alertas de autocorrelación
acf_alerts = [a for a in alerts if 'autocorr' in a.metric_name]

if acf_alerts:
    print(f"Cambio en autocorrelación detectado:")
    for alert in acf_alerts:
        print(f"  Δ_ACF = {alert.metadata.get('acf_change', 'N/A'):.3f}")
        print(f"  Severidad: {alert.severity.value}")

# Visualizar ACF
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

plot_acf(baseline, lags=144, ax=axes[0], title="ACF - Baseline")
plot_acf(current, lags=144, ax=axes[1], title="ACF - Producción")

plt.tight_layout()
plt.savefig("reports/acf_comparison.png")
print("Gráfico guardado en reports/acf_comparison.png")
```

### 11.5 Ejemplo: Detección de Drift Estacional

Escenario: Verificar si el patrón estacional ha cambiado

```python
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt

# Asegurar suficientes datos (mínimo 2 ciclos estacionales)
baseline_series = train_data['zone_1_power_consumption'][-288:]  # 2 días
current_series = production_data['zone_1_power_consumption'][:288]

# Descomponer series
decomp_baseline = seasonal_decompose(
    baseline_series,
    model='additive',
    period=144,
    extrapolate_trend='freq'
)

decomp_current = seasonal_decompose(
    current_series,
    model='additive',
    period=144,
    extrapolate_trend='freq'
)

# Calcular diferencia en componente estacional
seasonal_diff = np.sqrt(
    np.mean((decomp_baseline.seasonal - decomp_current.seasonal) ** 2)
)

print(f"Diferencia en estacionalidad: {seasonal_diff:.2f}")

# Visualizar descomposición
fig, axes = plt.subplots(4, 2, figsize=(14, 10))

# Baseline
axes[0, 0].plot(decomp_baseline.observed)
axes[0, 0].set_title("Baseline - Observado")

axes[1, 0].plot(decomp_baseline.trend)
axes[1, 0].set_title("Baseline - Tendencia")

axes[2, 0].plot(decomp_baseline.seasonal)
axes[2, 0].set_title("Baseline - Estacional")

axes[3, 0].plot(decomp_baseline.resid)
axes[3, 0].set_title("Baseline - Residual")

# Actual
axes[0, 1].plot(decomp_current.observed)
axes[0, 1].set_title("Producción - Observado")

axes[1, 1].plot(decomp_current.trend)
axes[1, 1].set_title("Producción - Tendencia")

axes[2, 1].plot(decomp_current.seasonal)
axes[2, 1].set_title("Producción - Estacional")

axes[3, 1].plot(decomp_current.resid)
axes[3, 1].set_title("Producción - Residual")

plt.tight_layout()
plt.savefig("reports/seasonal_decomposition.png")
print("Gráfico guardado en reports/seasonal_decomposition.png")
```

### 11.6 Ejemplo: Dashboard de Monitoreo

Escenario: Crear un dashboard para visualizar estado de drift

```python
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def create_drift_dashboard(report, output_file="reports/drift_dashboard.html"):
    """Crea dashboard interactivo con Plotly."""

    # Resumen
    summary = report.get_summary()

    # Subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Alertas por Severidad",
            "Alertas por Tipo de Drift",
            "Evolución de Métricas",
            "Top Features con Drift"
        ),
        specs=[
            [{"type": "bar"}, {"type": "pie"}],
            [{"type": "scatter"}, {"type": "bar"}]
        ]
    )

    # 1. Alertas por severidad
    severity_data = summary['severity_breakdown']
    fig.add_trace(
        go.Bar(
            x=list(severity_data.keys()),
            y=list(severity_data.values()),
            marker_color=['yellow', 'orange', 'red', 'purple'],
            name="Severidad"
        ),
        row=1, col=1
    )

    # 2. Alertas por tipo
    drift_type_data = summary['drift_type_breakdown']
    fig.add_trace(
        go.Pie(
            labels=list(drift_type_data.keys()),
            values=list(drift_type_data.values()),
            name="Tipo"
        ),
        row=1, col=2
    )

    # 3. Evolución temporal (ejemplo con PSI)
    # Aquí se necesitaría historial de múltiples chequeos
    # Para este ejemplo, usamos datos simulados
    timestamps = pd.date_range(end=datetime.now(), periods=10, freq='6H')
    psi_values = [a.metadata.get('psi', 0) for a in report.alerts[:10]]

    fig.add_trace(
        go.Scatter(
            x=timestamps,
            y=psi_values,
            mode='lines+markers',
            name="PSI",
            line=dict(color='blue')
        ),
        row=2, col=1
    )

    # Línea de threshold
    fig.add_hline(y=0.2, line_dash="dash", line_color="red", row=2, col=1)

    # 4. Top features con drift
    feature_alerts = [a for a in report.alerts
                     if a.drift_type == DriftType.FEATURE_DRIFT]

    if feature_alerts:
        # Ordenar por severidad
        feature_alerts_sorted = sorted(
            feature_alerts,
            key=lambda x: x.current_value,
            reverse=True
        )[:10]

        features = [a.metric_name for a in feature_alerts_sorted]
        values = [a.metadata.get('psi', 0) for a in feature_alerts_sorted]

        fig.add_trace(
            go.Bar(
                x=features,
                y=values,
                marker_color='indianred',
                name="PSI"
            ),
            row=2, col=2
        )

    # Configuración
    fig.update_layout(
        title_text="Dashboard de Monitoreo de Drift",
        showlegend=False,
        height=800
    )

    # Guardar
    fig.write_html(output_file)
    print(f"Dashboard guardado en {output_file}")

# Usar
report = pipeline.run(train_data, production_data)
create_drift_dashboard(report)
```

---

## 12. Referencias Bibliográficas

### 12.1 Papers Fundamentales

1. **Gama, J., Žliobaitė, I., Bifet, A., Pechenizkiy, M., & Bouchachia, A. (2014)**.
   "A survey on concept drift adaptation."
   *ACM computing surveys (CSUR)*, 46(4), 1-37.

   Revisión comprehensiva de métodos de detección y adaptación a concept drift.

2. **Žliobaitė, I. (2010)**.
   "Learning under concept drift: an overview."
   *arXiv preprint arXiv:1010.4784*.

   Overview de técnicas de aprendizaje bajo drift, con énfasis en series de tiempo.

3. **Bifet, A., & Gavaldà, R. (2007)**.
   "Learning from time-changing data with adaptive windowing."
   *Proceedings of the 2007 SIAM international conference on data mining* (pp. 443-448).

   Método ADWIN para detección de drift usando ventanas adaptativas.

### 12.2 Metodologías Estadísticas

4. **Siddiqi, N. (2006)**.
   *Credit risk scorecards: developing and implementing intelligent credit scoring*.
   John Wiley & Sons.

   Fuente original del Population Stability Index (PSI).

5. **Dickey, D. A., & Fuller, W. A. (1979)**.
   "Distribution of the estimators for autoregressive time series with a unit root."
   *Journal of the American statistical association*, 74(366a), 427-431.

   Test de raíz unitaria (ADF test) para estacionariedad.

6. **Cleveland, R. B., Cleveland, W. S., McRae, J. E., & Terpenning, I. (1990)**.
   "STL: A seasonal-trend decomposition."
   *J. Off. Stat*, 6(1), 3-73.

   Descomposición estacional usando LOESS.

### 12.3 Monitoreo de Modelos en Producción

7. **Breck, E., Polyzotis, N., Roy, S., Whang, S., & Zinkevich, M. (2019)**.
   "Data validation for machine learning."
   *Proceedings of SysML*.

   Framework de Google para validación de datos en ML (TensorFlow Data Validation).

8. **Rabanser, S., Günnemann, S., & Lipton, Z. (2019)**.
   "Failing loudly: An empirical study of methods for detecting dataset shift."
   *Advances in Neural Information Processing Systems*, 32.

   Comparación empírica de métodos de detección de drift.

9. **Lu, J., Liu, A., Dong, F., Gu, F., Gama, J., & Zhang, G. (2018)**.
   "Learning under concept drift: A review."
   *IEEE Transactions on Knowledge and Data Engineering*, 31(12), 2346-2363.

   Revisión reciente de métodos de aprendizaje bajo concept drift.

### 12.4 Series de Tiempo

10. **Box, G. E., Jenkins, G. M., Reinsel, G. C., & Ljung, G. M. (2015)**.
    *Time series analysis: forecasting and control*.
    John Wiley & Sons.

    Texto clásico sobre análisis de series de tiempo.

11. **Hyndman, R. J., & Athanasopoulos, G. (2018)**.
    *Forecasting: principles and practice*.
    OTexts.

    Texto moderno sobre forecasting, disponible online gratuitamente.

12. **Hamilton, J. D. (1994)**.
    *Time series analysis*.
    Princeton university press.

    Texto avanzado sobre econometría de series de tiempo.

### 12.5 Librerías y Herramientas

13. **Pedregosa, F., et al. (2011)**.
    "Scikit-learn: Machine learning in Python."
    *Journal of machine learning research*, 12, 2825-2830.

    Librería scikit-learn usada para métricas.

14. **Seabold, S., & Perktold, J. (2010)**.
    "statsmodels: Econometric and statistical modeling with python."
    *Proceedings of the 9th Python in Science Conference*.

    Librería statsmodels usada para análisis de series de tiempo.

15. **Virtanen, P., et al. (2020)**.
    "SciPy 1.0: fundamental algorithms for scientific computing in Python."
    *Nature methods*, 17(3), 261-272.

    Librería SciPy usada para tests estadísticos.

### 12.6 MLOps y Producción

16. **Sculley, D., et al. (2015)**.
    "Hidden technical debt in machine learning systems."
    *Advances in neural information processing systems*, 28.

    Análisis de deuda técnica en sistemas ML, incluyendo drift.

17. **Polyzotis, N., Roy, S., Whang, S. E., & Zinkevich, M. (2017)**.
    "Data management challenges in production machine learning."
    *Proceedings of the 2017 ACM International Conference on Management of Data* (pp. 1723-1726).

    Desafíos de gestión de datos en ML en producción.

### 12.7 Recursos Online

18. **Google Cloud**. "Monitoring models for training-serving skew and prediction drift."
    https://cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning

    Guía de Google sobre monitoreo de skew y drift.

19. **AWS**. "Amazon SageMaker Model Monitor."
    https://docs.aws.amazon.com/sagemaker/latest/dg/model-monitor.html

    Documentación de AWS sobre monitoreo de modelos.

20. **Evidently AI**. "Data and ML Model Monitoring."
    https://docs.evidentlyai.com/

    Herramienta open-source para monitoreo de drift.

---

## Apéndices

### Apéndice A: Glosario de Términos

**ACF (Autocorrelation Function)**: Función que mide la correlación de una serie de tiempo consigo misma en diferentes rezagos.

**ADF (Augmented Dickey-Fuller) Test**: Test estadístico para determinar si una serie de tiempo es estacionaria.

**Baseline**: Conjunto de datos de referencia, típicamente los datos de entrenamiento del modelo.

**Concept Drift**: Cambio en la relación entre features y target.

**Estacionariedad**: Propiedad de una serie de tiempo cuyas propiedades estadísticas no cambian con el tiempo.

**Feature Drift**: Cambio en la distribución de las variables de entrada.

**Jensen-Shannon Divergence**: Medida simétrica de similitud entre dos distribuciones de probabilidad.

**KS (Kolmogorov-Smirnov) Test**: Test no paramétrico que compara dos distribuciones empíricas.

**Label Drift**: Cambio en la distribución de la variable objetivo.

**Performance Drift**: Degradación en las métricas de performance del modelo.

**PSI (Population Stability Index)**: Métrica que mide el cambio en la distribución de una variable entre dos períodos.

**STL Decomposition**: Método para descomponer una serie de tiempo en componentes de tendencia, estacionalidad y residual.

**Temporal Drift**: Cambio en los patrones temporales de la serie (autocorrelación, estacionalidad).

### Apéndice B: Configuración Avanzada

**Ejemplo de Configuración YAML Completa**:

```yaml
drift_monitoring:
  # General
  enabled: true
  output_dir: reports/drift_monitoring
  log_level: INFO

  # Detectores
  detectors:
    statistical:
      enabled: true
      ks_threshold: 0.05
      psi_threshold: 0.2
      js_threshold: 0.1
      n_bins: 10

    timeseries:
      enabled: true
      seasonal_period: 144
      autocorr_lags: 24
      adf_threshold: 0.05
      acf_threshold: 0.3
      min_samples: 288  # 2 períodos estacionales

    performance:
      enabled: true
      window_size: 144
      performance_threshold: 0.15
      min_samples: 100
      metrics:
        - rmse
        - mae
        - mape

  # Sistema de alertas
  alerts:
    channels:
      console:
        enabled: true
        colored: true
        min_severity: medium

      file:
        enabled: true
        output_path: reports/alerts.json
        append: true
        min_severity: low

      # Extensible: agregar Slack, email, etc.
      # slack:
      #   enabled: false
      #   webhook_url: ${SLACK_WEBHOOK_URL}
      #   min_severity: high

    severity_thresholds:
      none: 0.0
      low: 0.5
      medium: 1.0
      high: 1.5
      critical: 2.0

  # Monitoreo en tiempo real
  realtime:
    enabled: true
    monitoring_window_hours: 24
    check_interval_hours: 6
    min_predictions: 100
    log_dir: logs/predictions

  # Features a monitorear
  monitored_features:
    - temperature
    - humidity
    - wind_speed
    - general_diffuse_flows
    - diffuse_flows
    - lag_zone_1_power_consumption_1_hora
    - lag_zone_1_power_consumption_24_horas
```

### Apéndice C: Troubleshooting

**Problema: "Insufficient data for drift check"**

*Causa*: Menos de 100 registros en ventana de monitoreo.

*Solución*:
1. Esperar más tiempo para acumular datos
2. Reducir `monitoring_window_hours`
3. Verificar que las predicciones se están loggeando correctamente

**Problema: "No seasonal decomposition possible"**

*Causa*: Datos insuficientes (< 2 períodos estacionales).

*Solución*:
1. Usar solo `StatisticalDriftDetector` temporalmente
2. Esperar más datos (mínimo 288 registros para período de 144)
3. Ajustar `seasonal_period` si el período real es diferente

**Problema: Demasiadas alertas (false positives)**

*Causa*: Thresholds demasiado estrictos o variabilidad natural alta.

*Solución*:
1. Relajar thresholds:
   ```python
   detector = StatisticalDriftDetector(
       ks_threshold=0.01,  # De 0.05 a 0.01
       psi_threshold=0.25  # De 0.2 a 0.25
   )
   ```
2. Aumentar `min_severity` en canales de alerta
3. Analizar variabilidad natural en datos baseline

**Problema: Performance lento con datasets grandes**

*Causa*: Cálculos computacionalmente intensivos en datos masivos.

*Solución*:
1. Reducir tamaño de ventanas
2. Usar sampling estratificado
3. Ejecutar monitoreo en background thread
4. Optimizar frecuencia de chequeos

---

**FIN DEL DOCUMENTO**

Este documento proporciona una guía completa del sistema de monitoreo de data drift implementado para el proyecto de predicción de consumo eléctrico en Tetouan City. Para preguntas o contribuciones, consultar el repositorio del proyecto.
