#!/usr/bin/env python
"""
Script para generar visualizaciones de drift monitoring.

Uso:
    python generate_drift_plots.py

    O con parámetros personalizados:
    python generate_drift_plots.py --reference data/processed/train.parquet --current data/processed/test.parquet
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.monitoring import (
    DriftVisualizer,
    StatisticalDriftDetector,
    TimeSeriesDriftDetector,
    create_default_pipeline,
)


def generate_sample_data(n=500, seed=42):
    """Genera datos sintéticos para demo."""
    np.random.seed(seed)
    timestamps = pd.date_range("2025-01-01", periods=n, freq="10min")
    hour_of_day = np.arange(n) % 144 / 144 * 2 * np.pi
    seasonal = 1000 + 500 * np.sin(hour_of_day)

    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "temperature": 25 + 5 * np.sin(hour_of_day) + np.random.normal(0, 2, n),
            "humidity": 60 + 10 * np.cos(hour_of_day) + np.random.normal(0, 5, n),
            "wind_speed": 5 + np.random.exponential(2, n),
            "general_diffuse_flows": 150 + np.random.normal(0, 10, n),
            "diffuse_flows": 80 + np.random.normal(0, 5, n),
            "zone_1_power_consumption": seasonal + np.random.normal(0, 50, n),
        }
    )


def introduce_drift(df):
    """Introduce drift en los datos."""
    drifted = df.copy()
    drifted["temperature"] += 10
    drifted["humidity"] *= 1.3
    drifted["zone_1_power_consumption"] += 200
    return drifted


def main():
    parser = argparse.ArgumentParser(
        description="Generar visualizaciones de drift monitoring"
    )
    parser.add_argument(
        "--reference",
        type=str,
        help="Ruta al archivo de datos de referencia (baseline)",
    )
    parser.add_argument(
        "--current",
        type=str,
        help="Ruta al archivo de datos actuales (producción)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="reports/drift_monitoring",
        help="Directorio de salida para gráficos (default: reports/drift_monitoring)",
    )
    parser.add_argument(
        "--top-features",
        type=int,
        default=5,
        help="Número de features con mayor drift a graficar (default: 5)",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Ejecutar con datos sintéticos de demostración",
    )

    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("SISTEMA DE VISUALIZACIÓN DE DRIFT MONITORING")
    print("=" * 70)

    # Cargar o generar datos
    if args.demo or (not args.reference and not args.current):
        print("\nModo DEMO: Generando datos sintéticos...")
        baseline = generate_sample_data(n=500, seed=42)
        production = introduce_drift(baseline)
        print(f"  Baseline: {len(baseline)} registros")
        print(f"  Producción: {len(production)} registros")
    else:
        if not args.reference or not args.current:
            print("\nError: Se requieren ambos parámetros --reference y --current")
            print("       O use --demo para ejecutar con datos sintéticos")
            sys.exit(1)

        print(f"\nCargando datos...")
        print(f"  Referencia: {args.reference}")
        print(f"  Actual: {args.current}")

        ref_path = Path(args.reference)
        curr_path = Path(args.current)

        if not ref_path.exists():
            print(f"Error: Archivo no encontrado: {args.reference}")
            sys.exit(1)

        if not curr_path.exists():
            print(f"Error: Archivo no encontrado: {args.current}")
            sys.exit(1)

        # Cargar datos
        if ref_path.suffix == ".parquet":
            baseline = pd.read_parquet(ref_path)
            production = pd.read_parquet(curr_path)
        elif ref_path.suffix == ".csv":
            baseline = pd.read_csv(ref_path)
            production = pd.read_csv(curr_path)
        else:
            print(f"Error: Formato no soportado: {ref_path.suffix}")
            sys.exit(1)

        print(f"  Baseline: {len(baseline)} registros, {len(baseline.columns)} columnas")
        print(f"  Producción: {len(production)} registros, {len(production.columns)} columnas")

    # Ejecutar pipeline de drift detection
    print("\nEjecutando detección de drift...")
    pipeline = create_default_pipeline(output_dir=args.output)
    report = pipeline.run(baseline, production)

    # Generar visualizaciones
    print("\nGenerando visualizaciones...")
    visualizer = DriftVisualizer(output_dir=Path(args.output) / "plots")

    try:
        figures = visualizer.create_drift_report_plots(
            baseline, production, report.alerts, top_n_features=args.top_features
        )
    except Exception as e:
        print(f"\nError al generar visualizaciones: {str(e)}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

    # Resumen
    print("\n" + "=" * 70)
    print("VISUALIZACIONES GENERADAS")
    print("=" * 70)

    summary = report.get_summary()
    print(f"\nTotal de alertas detectadas: {summary['total_alerts']}")
    print(f"Acción requerida: {'Sí' if summary['requires_action'] else 'No'}")

    print(f"\nGráficos generados: {len(figures)}")
    print(f"Ubicación: {Path(args.output) / 'plots'}")

    # Listar archivos generados
    plots_dir = Path(args.output) / "plots"
    if plots_dir.exists():
        print("\nArchivos:")
        for file in sorted(plots_dir.glob("*.png")):
            size_kb = file.stat().st_size // 1024
            print(f"  - {file.name} ({size_kb} KB)")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
