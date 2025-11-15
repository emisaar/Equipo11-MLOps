"""
Demo Completo: Monitoreo y Visualización de Data Drift para MLOps

Este script demuestra el sistema completo de monitoreo de drift incluyendo:
- Detección de drift (Statistical, TimeSeries, Performance)
- Visualizaciones automáticas (distribuciones, series temporales, métricas)
- Análisis de reportes y recomendaciones
- Pipeline completo de monitoreo

Proyecto: Pronóstico de consumo eléctrico de Tetouan

Uso
---
python examples/drift_monitoring_demo.py
"""

from pathlib import Path
import sys

# Agregar directorio padre al path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.monitoring import (
    create_default_pipeline,
    StatisticalDriftDetector,
    TimeSeriesDriftDetector,
    ModelPerformanceMonitor,
    DriftMonitoringPipeline,
    DriftVisualizer,
    AlertManager,
    ConsoleAlertChannel,
    FileAlertChannel,
    create_visualizations,
)


# ============================================================================
# UTILIDADES COMPARTIDAS
# ============================================================================


def create_sample_data(n: int = 1000, add_drift: bool = False, seed: int = None) -> pd.DataFrame:
    """
    Crea datos de series de tiempo de muestra para demostración.

    Parámetros
    ----------
    n : int
        Número de registros
    add_drift : bool
        Si agregar drift a los datos
    seed : int
        Semilla para reproducibilidad

    Retorna
    -------
    pd.DataFrame
        Datos de muestra con features de series de tiempo
    """
    if seed is None:
        seed = 42 if not add_drift else 123

    np.random.seed(seed)
    timestamps = pd.date_range("2025-01-01", periods=n, freq="10min")

    # Simular patrones realistas de consumo eléctrico
    hour_of_day = np.arange(n) % 144 / 144 * 2 * np.pi
    seasonal_pattern = 1000 + 500 * np.sin(hour_of_day)

    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "temperature": 25 + 5 * np.sin(hour_of_day) + np.random.normal(0, 2, n),
            "humidity": 60 + 10 * np.cos(hour_of_day) + np.random.normal(0, 5, n),
            "wind_speed": 5 + np.random.exponential(2, n),
            "general_diffuse_flows": 150 + 20 * np.sin(hour_of_day / 2) + np.random.normal(0, 5, n),
            "diffuse_flows": 80 + 15 * np.cos(hour_of_day / 2) + np.random.normal(0, 3, n),
            "zone_1_power_consumption": seasonal_pattern + np.random.normal(0, 50, n),
        }
    )

    # Agregar drift si se solicita
    if add_drift:
        df["temperature"] += 8  # Aumento de temperatura
        df["humidity"] *= 1.25  # Aumento de humedad
        df["wind_speed"] *= 0.7  # Disminución de velocidad del viento
        df["zone_1_power_consumption"] += 300  # Aumento de consumo

    return df


def add_predictions(df: pd.DataFrame, noise_level: float = 50) -> pd.DataFrame:
    """
    Agrega columnas de predicción al DataFrame.

    Parámetros
    ----------
    df : pd.DataFrame
        DataFrame con datos
    noise_level : float
        Nivel de ruido en las predicciones

    Retorna
    -------
    pd.DataFrame
        DataFrame con columnas y_true y y_pred
    """
    df = df.copy()
    df["y_true"] = df["zone_1_power_consumption"]
    df["y_pred"] = df["y_true"] + np.random.normal(0, noise_level, len(df))
    return df


# ============================================================================
# DEMOS PRINCIPALES - DETECCIÓN + VISUALIZACIÓN
# ============================================================================


def demo_1_basic_drift_detection_with_viz():
    """
    DEMO 1: Detección Básica de Drift + Visualizaciones

    Muestra:
    - Uso del pipeline por defecto
    - Detección de drift estadístico y temporal
    - Visualizaciones de distribuciones
    """
    print("\n" + "=" * 80)
    print("DEMO 1: Detección Básica de Drift + Visualizaciones")
    print("=" * 80)

    # Crear datos
    print("\n[1/4] Creando datos de muestra...")
    reference_data = create_sample_data(n=1000, add_drift=False)
    current_data = create_sample_data(n=1000, add_drift=True)

    print(f"  > Datos de referencia: {len(reference_data)} registros")
    print(f"  > Datos actuales: {len(current_data)} registros")

    # Crear pipeline
    print("\n[2/4] Configurando pipeline de detección...")
    output_dir = Path("reports/demo_drift/demo1_basic")
    pipeline = create_default_pipeline(output_dir=output_dir)

    print(f"  > Detectores configurados: {len(pipeline.detectors)}")
    for detector in pipeline.detectors:
        print(f"    - {detector.name}")

    # Ejecutar detección
    print("\n[3/4] Ejecutando detección de drift...")
    report = pipeline.run(
        reference_data=reference_data,
        current_data=current_data,
        reference_period="Baseline (sin drift)",
        current_period="Producción (con drift)",
    )

    print(f"  > Total de alertas detectadas: {len(report.alerts)}")

    # Mostrar primeras alertas
    if report.alerts:
        print("\n  Primeras alertas:")
        for alert in report.alerts[:3]:
            print(f"    - {alert.metric_name}: {alert.severity.value} ({alert.drift_type.value})")

    # Generar visualizaciones
    print("\n[4/4] Generando visualizaciones...")
    visualizer = DriftVisualizer(output_dir=output_dir / "plots")

    # Gráficos de distribución para features principales
    visualizer.plot_distribution_comparison(reference_data, current_data, "temperature")
    visualizer.plot_distribution_comparison(reference_data, current_data, "humidity")

    # Resumen de alertas
    visualizer.plot_drift_summary(report.alerts)

    print(f"  > Gráficos guardados en: {visualizer.output_dir}")
    print(f"  > Reporte JSON: {output_dir / 'drift_monitoring_report.json'}")


def demo_2_timeseries_drift_detection():
    """
    DEMO 2: Detección de Drift en Series Temporales

    Muestra:
    - Detección especializada en series temporales
    - Análisis de autocorrelación y estacionalidad
    - Visualizaciones temporales
    """
    print("\n" + "=" * 80)
    print("DEMO 2: Detección de Drift en Series Temporales")
    print("=" * 80)

    # Crear datos
    print("\n[1/4] Creando datos temporales...")
    reference_data = create_sample_data(n=1000, add_drift=False)
    current_data = create_sample_data(n=1000, add_drift=True)

    # Detector temporal específico
    print("\n[2/4] Configurando detector de series temporales...")
    ts_detector = TimeSeriesDriftDetector(
        seasonal_period=144,  # Período diario (10min * 144 = 24h)
        autocorr_lags=24,
        adf_threshold=0.05
    )

    # Ejecutar detección
    print("\n[3/4] Detectando drift temporal...")
    alerts = ts_detector.detect(reference_data, current_data)

    print(f"  > Alertas de series temporales: {len(alerts)}")
    for alert in alerts[:3]:
        print(f"    - {alert.metric_name}: {alert.severity.value}")

    # Visualizaciones temporales
    print("\n[4/4] Generando visualizaciones de series temporales...")
    output_dir = Path("reports/demo_drift/demo2_timeseries")
    visualizer = DriftVisualizer(output_dir=output_dir / "plots")

    # Análisis temporal de consumo
    visualizer.plot_timeseries_comparison(
        reference_data,
        current_data,
        "zone_1_power_consumption",
        timestamp_col="timestamp"
    )

    # Análisis temporal de temperatura
    visualizer.plot_timeseries_comparison(
        reference_data,
        current_data,
        "temperature",
        timestamp_col="timestamp"
    )

    print(f"  > Gráficos guardados en: {visualizer.output_dir}")


def demo_3_performance_monitoring_with_viz():
    """
    DEMO 3: Monitoreo de Performance del Modelo

    Muestra:
    - Detección de degradación de performance
    - Métricas de error (RMSE, MAE, MAPE)
    - Visualizaciones de performance
    """
    print("\n" + "=" * 80)
    print("DEMO 3: Monitoreo de Performance del Modelo")
    print("=" * 80)

    # Crear datos con predicciones
    print("\n[1/4] Creando datos con predicciones...")

    # Baseline: buen performance
    reference_data = create_sample_data(n=500, add_drift=False)
    reference_data = add_predictions(reference_data, noise_level=50)

    # Current: performance degradado
    current_data = create_sample_data(n=500, add_drift=True)
    current_data = add_predictions(current_data, noise_level=150)

    print(f"  > Datos de referencia: {len(reference_data)} registros con predicciones")
    print(f"  > Datos actuales: {len(current_data)} registros con predicciones")

    # Monitor de performance
    print("\n[2/4] Configurando monitor de performance...")
    perf_monitor = ModelPerformanceMonitor(
        window_size=144,
        performance_threshold=0.15
    )

    # Ejecutar detección
    print("\n[3/4] Detectando degradación de performance...")
    perf_alerts = perf_monitor.detect(reference_data, current_data)

    print(f"  > Alertas de performance: {len(perf_alerts)}")
    for alert in perf_alerts:
        change_pct = ((alert.current_value - alert.baseline_value) / alert.baseline_value * 100)
        print(
            f"    - {alert.metric_name}: {alert.severity.value} "
            f"(baseline: {alert.baseline_value:.2f}, "
            f"current: {alert.current_value:.2f}, "
            f"cambio: {change_pct:+.1f}%)"
        )

    # Visualizaciones de performance
    print("\n[4/4] Generando visualizaciones de performance...")
    output_dir = Path("reports/demo_drift/demo3_performance")
    visualizer = DriftVisualizer(output_dir=output_dir / "plots")

    visualizer.plot_performance_metrics(reference_data, current_data)

    print(f"  > Gráficos guardados en: {visualizer.output_dir}")


def demo_4_complete_pipeline_with_full_report():
    """
    DEMO 4: Pipeline Completo con Reporte Visual Completo

    Muestra:
    - Pipeline con todos los detectores
    - Generación automática de todas las visualizaciones
    - Reporte completo en JSON + imágenes
    """
    print("\n" + "=" * 80)
    print("DEMO 4: Pipeline Completo con Reporte Visual Completo")
    print("=" * 80)

    # Crear datos completos
    print("\n[1/4] Creando dataset completo...")
    reference_data = create_sample_data(n=1000, add_drift=False)
    reference_data = add_predictions(reference_data, noise_level=50)

    current_data = create_sample_data(n=1000, add_drift=True)
    current_data = add_predictions(current_data, noise_level=150)

    print(f"  > Features: {len(reference_data.columns)}")
    print(f"  > Registros: {len(reference_data)} (baseline) / {len(current_data)} (producción)")

    # Pipeline completo
    print("\n[2/4] Ejecutando pipeline completo de detección...")
    output_dir = Path("reports/demo_drift/demo4_complete")
    pipeline = create_default_pipeline(output_dir=output_dir)

    report = pipeline.run(
        reference_data=reference_data,
        current_data=current_data,
        reference_period="Training (Enero 2025)",
        current_period="Production (Febrero 2025)"
    )

    print(f"  > Total de alertas: {len(report.alerts)}")

    # Análisis del reporte
    print("\n[3/4] Analizando reporte...")
    summary = report.get_summary()

    print(f"  Resumen:")
    print(f"    - Total alertas: {summary['total_alerts']}")
    print(f"    - Requiere acción: {'Sí' if summary['requires_action'] else 'No'}")
    print(f"    - Alertas críticas: {'Sí' if summary['has_critical_alerts'] else 'No'}")

    if summary["severity_breakdown"]:
        print(f"\n  Severidad:")
        for severity, count in summary["severity_breakdown"].items():
            print(f"    - {severity}: {count}")

    if summary["drift_type_breakdown"]:
        print(f"\n  Tipo de drift:")
        for dtype, count in summary["drift_type_breakdown"].items():
            print(f"    - {dtype}: {count}")

    # Recomendaciones
    recommendations = report.get_recommendations()
    if recommendations:
        print(f"\n  Recomendaciones:")
        for i, rec in enumerate(recommendations[:3], 1):
            print(f"    {i}. {rec}")

    # Generar todas las visualizaciones
    print("\n[4/4] Generando reporte visual completo...")
    visualizer = DriftVisualizer(output_dir=output_dir / "plots")

    figures = visualizer.create_drift_report_plots(
        reference_data,
        current_data,
        report.alerts,
        top_n_features=5
    )

    print(f"  > Total de gráficos generados: {len(figures)}")
    print(f"  > Ubicación: {visualizer.output_dir}")
    print(f"  > Reporte JSON: {output_dir / 'drift_monitoring_report.json'}")


def demo_5_custom_pipeline_configuration():
    """
    DEMO 5: Configuración Personalizada del Pipeline

    Muestra:
    - Configuración custom de detectores
    - Canales de alerta personalizados
    - Umbrales ajustados
    """
    print("\n" + "=" * 80)
    print("DEMO 5: Configuración Personalizada del Pipeline")
    print("=" * 80)

    # Crear datos
    print("\n[1/4] Creando datos...")
    reference_data = create_sample_data(n=1000, add_drift=False)
    current_data = create_sample_data(n=1000, add_drift=True)

    # Configuración custom
    print("\n[2/4] Configurando pipeline personalizado...")

    # Alert manager con múltiples canales
    output_dir = Path("reports/demo_drift/demo5_custom")
    console_channel = ConsoleAlertChannel(colored=True)
    file_channel = FileAlertChannel(output_dir / "custom_alerts.json")
    alert_manager = AlertManager(channels=[console_channel, file_channel])

    print("  > Canales de alerta:")
    print("    - Console (con colores)")
    print("    - File (JSON)")

    # Detectores con umbrales personalizados
    detectors = [
        StatisticalDriftDetector(
            ks_threshold=0.01,      # Más estricto
            psi_threshold=0.15,     # Más estricto
            js_threshold=0.08       # Más estricto
        ),
        TimeSeriesDriftDetector(
            seasonal_period=144,
            autocorr_lags=48        # Más lags para análisis profundo
        ),
    ]

    print("  > Detectores configurados:")
    print("    - Statistical (umbrales estrictos)")
    print("    - TimeSeries (análisis profundo)")

    # Crear pipeline
    pipeline = DriftMonitoringPipeline(
        detectors=detectors,
        alert_manager=alert_manager,
        output_dir=output_dir,
    )

    # Ejecutar
    print("\n[3/4] Ejecutando pipeline personalizado...")
    report = pipeline.run(reference_data, current_data)

    print(f"  > Alertas detectadas: {len(report.alerts)}")

    # Visualizaciones
    print("\n[4/4] Generando visualizaciones...")
    visualizer = DriftVisualizer(output_dir=output_dir / "plots")

    # Solo features con drift crítico
    critical_features = set()
    for alert in report.alerts:
        if alert.severity.value in ['critical', 'high']:
            if hasattr(alert, 'feature_name') and alert.feature_name:
                critical_features.add(alert.feature_name)

    print(f"  > Features con drift crítico: {len(critical_features)}")
    for feature in list(critical_features)[:3]:
        visualizer.plot_distribution_comparison(reference_data, current_data, feature)

    visualizer.plot_drift_summary(report.alerts)

    print(f"  > Gráficos guardados en: {visualizer.output_dir}")


def demo_6_file_based_workflow():
    """
    DEMO 6: Workflow Completo Basado en Archivos

    Muestra:
    - Guardar datos en archivos Parquet
    - Ejecutar pipeline desde archivos
    - Generar reporte completo con visualizaciones
    - Workflow realista de producción
    """
    print("\n" + "=" * 80)
    print("DEMO 6: Workflow Completo Basado en Archivos")
    print("=" * 80)

    # Preparar directorios
    print("\n[1/5] Preparando estructura de archivos...")
    output_dir = Path("reports/demo_drift/demo6_file_based")
    data_dir = output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    # Crear y guardar datos
    print("\n[2/5] Generando y guardando datos...")
    reference_data = create_sample_data(n=1000, add_drift=False)
    reference_data = add_predictions(reference_data, noise_level=50)

    current_data = create_sample_data(n=1000, add_drift=True)
    current_data = add_predictions(current_data, noise_level=150)

    ref_path = data_dir / "baseline_jan2025.parquet"
    curr_path = data_dir / "production_feb2025.parquet"

    reference_data.to_parquet(ref_path)
    current_data.to_parquet(curr_path)

    print(f"  > Baseline guardado: {ref_path}")
    print(f"  > Producción guardado: {curr_path}")

    # Ejecutar pipeline desde archivos
    print("\n[3/5] Ejecutando pipeline desde archivos...")
    pipeline = create_default_pipeline(output_dir=output_dir / "results")

    report = pipeline.run_from_files(
        reference_path=ref_path,
        current_path=curr_path,
        reference_period="Baseline - Enero 2025",
        current_period="Producción - Febrero 2025",
    )

    print(f"  > Detección completada: {len(report.alerts)} alertas")

    # Generar reporte visual
    print("\n[4/5] Generando reporte visual completo...")
    visualizer = DriftVisualizer(output_dir=output_dir / "results" / "plots")

    figures = visualizer.create_drift_report_plots(
        reference_data,
        current_data,
        report.alerts,
        top_n_features=5
    )

    print(f"  > Gráficos generados: {len(figures)}")

    # Resumen de archivos generados
    print("\n[5/5] Resumen de archivos generados:")

    results_dir = output_dir / "results"
    print(f"\n  Directorio: {results_dir}")
    print(f"    - drift_monitoring_report.json (reporte completo)")
    print(f"    - drift_alerts.json (alertas)")

    plots_dir = results_dir / "plots"
    if plots_dir.exists():
        plot_files = list(plots_dir.glob("*.png"))
        print(f"\n  Directorio de gráficos: {plots_dir}")
        for plot_file in sorted(plot_files)[:5]:
            size_kb = plot_file.stat().st_size // 1024
            print(f"    - {plot_file.name} ({size_kb} KB)")

        if len(plot_files) > 5:
            print(f"    ... y {len(plot_files) - 5} más")


# ============================================================================
# MAIN
# ============================================================================


def main():
    """Ejecuta todas las demos del sistema de drift monitoring."""
    print("\n" + "=" * 80)
    print("SISTEMA COMPLETO DE MONITOREO Y VISUALIZACIÓN DE DRIFT")
    print("Time Series MLOps - Tetouan Power Consumption")
    print("=" * 80)
    print("\nEste demo muestra:")
    print("  - Detección de drift (Statistical, TimeSeries, Performance)")
    print("  - Visualizaciones automáticas (distribuciones, series, métricas)")
    print("  - Análisis de reportes y recomendaciones")
    print("  - Workflows completos de producción")

    demos = [
        ("Demo 1", "Detección básica + visualizaciones", demo_1_basic_drift_detection_with_viz),
        ("Demo 2", "Análisis de series temporales", demo_2_timeseries_drift_detection),
        ("Demo 3", "Monitoreo de performance", demo_3_performance_monitoring_with_viz),
        ("Demo 4", "Pipeline completo con reporte visual", demo_4_complete_pipeline_with_full_report),
        ("Demo 5", "Configuración personalizada", demo_5_custom_pipeline_configuration),
        ("Demo 6", "Workflow basado en archivos", demo_6_file_based_workflow),
    ]

    print("\n" + "-" * 80)
    print("Demos disponibles:")
    for num, desc, _ in demos:
        print(f"  {num}: {desc}")
    print("-" * 80)

    print("\nEjecutando todas las demos...\n")

    success_count = 0
    error_count = 0

    for name, desc, demo_func in demos:
        try:
            demo_func()
            success_count += 1
        except Exception as e:
            error_count += 1
            print(f"\n ERROR en {name}: {e}")
            import traceback
            traceback.print_exc()

    # Resumen final
    print("\n" + "=" * 80)
    print("RESUMEN DE EJECUCIÓN")
    print("=" * 80)
    print(f"\nDemos exitosos: {success_count}/{len(demos)}")
    if error_count > 0:
        print(f"Demos con errores: {error_count}/{len(demos)}")

    print("\n" + "=" * 80)
    print("ARCHIVOS GENERADOS")
    print("=" * 80)
    print("\nRevisar los siguientes directorios:")
    print("  - reports/demo_drift/demo1_basic/")
    print("  - reports/demo_drift/demo2_timeseries/")
    print("  - reports/demo_drift/demo3_performance/")
    print("  - reports/demo_drift/demo4_complete/")
    print("  - reports/demo_drift/demo5_custom/")
    print("  - reports/demo_drift/demo6_file_based/")

    print("\nTipos de archivos generados:")
    print("  - *.json - Reportes y alertas en formato JSON")
    print("  - *.png - Visualizaciones (300 DPI)")
    print("    > Distribuciones (histogramas, box plots, CDF, Q-Q)")
    print("    > Series temporales (con media móvil y ACF)")
    print("    > Performance (scatter plots, errores, métricas)")
    print("    > Resumen (severidad, tipos, timeline)")

    print("\n" + "=" * 80)
    print("DEMOS COMPLETADAS")
    print("=" * 80)
    print()


if __name__ == "__main__":
    main()
