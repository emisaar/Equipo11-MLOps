"""
Visualizaciones para Drift Monitoring.

Este módulo proporciona funciones para generar gráficos comparativos que ayudan
a visualizar y analizar el drift detectado entre datos de referencia y actuales.

Funciones
---------
plot_distribution_comparison
    Compara distribuciones de features entre baseline y producción
plot_timeseries_comparison
    Compara series temporales entre períodos
plot_performance_metrics
    Visualiza degradación de métricas del modelo
plot_drift_summary
    Resumen visual de alertas detectadas
create_drift_report_plots
    Genera conjunto completo de gráficos para un reporte
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure

from .drift_detectors import DriftAlert, DriftSeverity, DriftType


class DriftVisualizer:
    """
    Clase para crear visualizaciones de drift monitoring.

    Parameters
    ----------
    output_dir : Path, optional
        Directorio para guardar gráficos, por defecto "reports/drift_monitoring/plots"
    style : str, optional
        Estilo de matplotlib, por defecto "seaborn-v0_8-darkgrid"
    figsize : tuple, optional
        Tamaño por defecto de figuras, por defecto (12, 6)
    """

    def __init__(
        self,
        output_dir: Optional[Path] = None,
        style: str = "seaborn-v0_8-darkgrid",
        figsize: tuple = (12, 6),
    ):
        """Inicializa visualizador de drift."""
        self.output_dir = Path(output_dir) if output_dir else Path("reports/drift_monitoring/plots")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.style = style
        self.figsize = figsize

        # Configurar estilo
        plt.style.use(style)
        sns.set_palette("husl")

    def plot_distribution_comparison(
        self,
        reference_data: pd.DataFrame,
        current_data: pd.DataFrame,
        feature_name: str,
        save: bool = True,
    ) -> Figure:
        """
        Compara distribución de una feature entre baseline y producción.

        Parameters
        ----------
        reference_data : pd.DataFrame
            Datos de referencia (baseline)
        current_data : pd.DataFrame
            Datos actuales (producción)
        feature_name : str
            Nombre de la feature a comparar
        save : bool, optional
            Si guardar el gráfico, por defecto True

        Returns
        -------
        Figure
            Figura de matplotlib
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Obtener datos
        ref_values = reference_data[feature_name].dropna()
        curr_values = current_data[feature_name].dropna()

        # 1. Histogramas superpuestos
        axes[0, 0].hist(ref_values, bins=30, alpha=0.5, label="Baseline", color="blue", density=True)
        axes[0, 0].hist(
            curr_values, bins=30, alpha=0.5, label="Producción", color="red", density=True
        )
        axes[0, 0].set_xlabel(feature_name)
        axes[0, 0].set_ylabel("Densidad")
        axes[0, 0].set_title(f"Distribución de {feature_name}")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # 2. Box plots comparativos
        data_for_box = pd.DataFrame(
            {
                "Valor": np.concatenate([ref_values, curr_values]),
                "Tipo": (
                    ["Baseline"] * len(ref_values) + ["Producción"] * len(curr_values)
                ),
            }
        )
        sns.boxplot(data=data_for_box, x="Tipo", y="Valor", ax=axes[0, 1])
        axes[0, 1].set_title(f"Comparación Box Plot - {feature_name}")
        axes[0, 1].grid(True, alpha=0.3)

        # 3. Distribución acumulada (CDF)
        axes[1, 0].hist(
            ref_values,
            bins=50,
            cumulative=True,
            density=True,
            alpha=0.5,
            label="Baseline",
            color="blue",
        )
        axes[1, 0].hist(
            curr_values,
            bins=50,
            cumulative=True,
            density=True,
            alpha=0.5,
            label="Producción",
            color="red",
        )
        axes[1, 0].set_xlabel(feature_name)
        axes[1, 0].set_ylabel("Probabilidad Acumulada")
        axes[1, 0].set_title("Función de Distribución Acumulada (CDF)")
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # 4. Q-Q Plot
        from scipy import stats

        (osm, osr), (slope, intercept, r) = stats.probplot(curr_values, dist=stats.norm, fit=True)
        axes[1, 1].scatter(osm, osr, alpha=0.5, s=10)
        axes[1, 1].plot(osm, slope * osm + intercept, "r-", label=f"Fit (R²={r**2:.3f})")
        axes[1, 1].set_xlabel("Cuantiles Teóricos")
        axes[1, 1].set_ylabel("Cuantiles Observados")
        axes[1, 1].set_title(f"Q-Q Plot - {feature_name} (Producción)")
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save:
            filename = self.output_dir / f"distribution_{feature_name}.png"
            fig.savefig(filename, dpi=300, bbox_inches="tight")
            print(f"  Gráfico guardado: {filename}")

        return fig

    def plot_timeseries_comparison(
        self,
        reference_data: pd.DataFrame,
        current_data: pd.DataFrame,
        feature_name: str,
        timestamp_col: Optional[str] = None,
        save: bool = True,
    ) -> Figure:
        """
        Compara series temporales entre baseline y producción.

        Parameters
        ----------
        reference_data : pd.DataFrame
            Datos de referencia
        current_data : pd.DataFrame
            Datos actuales
        feature_name : str
            Nombre de la feature a graficar
        timestamp_col : str, optional
            Columna de timestamps, si None usa el índice
        save : bool, optional
            Si guardar el gráfico

        Returns
        -------
        Figure
            Figura de matplotlib
        """
        fig, axes = plt.subplots(3, 1, figsize=(14, 10))

        # Preparar datos
        if timestamp_col and timestamp_col in reference_data.columns:
            ref_x = pd.to_datetime(reference_data[timestamp_col])
            curr_x = pd.to_datetime(current_data[timestamp_col])
        else:
            ref_x = np.arange(len(reference_data))
            curr_x = np.arange(len(current_data))

        ref_y = reference_data[feature_name].values
        curr_y = current_data[feature_name].values

        # 1. Series temporales
        axes[0].plot(ref_x[: min(500, len(ref_x))], ref_y[: min(500, len(ref_y))],
                     label="Baseline", color="blue", alpha=0.7, linewidth=1)
        axes[0].plot(curr_x[: min(500, len(curr_x))], curr_y[: min(500, len(curr_y))],
                     label="Producción", color="red", alpha=0.7, linewidth=1)
        axes[0].set_ylabel(feature_name)
        axes[0].set_title(f"Comparación Temporal - {feature_name}")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # 2. Rolling mean
        window = min(24, len(ref_y) // 10)
        ref_rolling = pd.Series(ref_y).rolling(window=window, center=True).mean()
        curr_rolling = pd.Series(curr_y).rolling(window=window, center=True).mean()

        axes[1].plot(ref_x[: len(ref_rolling)], ref_rolling,
                     label=f"Baseline (MA {window})", color="blue", linewidth=2)
        axes[1].plot(curr_x[: len(curr_rolling)], curr_rolling,
                     label=f"Producción (MA {window})", color="red", linewidth=2)
        axes[1].set_ylabel(f"{feature_name} (Media Móvil)")
        axes[1].set_title("Tendencia (Media Móvil)")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        # 3. Autocorrelación comparativa
        from statsmodels.tsa.stattools import acf

        max_lags = min(48, len(ref_y) // 4, len(curr_y) // 4)

        try:
            ref_acf = acf(ref_y, nlags=max_lags, fft=True)
            curr_acf = acf(curr_y, nlags=max_lags, fft=True)

            lags = np.arange(len(ref_acf))
            axes[2].bar(
                lags - 0.2, ref_acf, width=0.4, alpha=0.7, label="Baseline", color="blue"
            )
            axes[2].bar(
                lags + 0.2, curr_acf, width=0.4, alpha=0.7, label="Producción", color="red"
            )
            axes[2].axhline(y=0, color="k", linestyle="-", linewidth=0.5)
            axes[2].axhline(y=1.96 / np.sqrt(len(ref_y)), color="b", linestyle="--", alpha=0.5)
            axes[2].axhline(y=-1.96 / np.sqrt(len(ref_y)), color="b", linestyle="--", alpha=0.5)
            axes[2].set_xlabel("Lag")
            axes[2].set_ylabel("Autocorrelación")
            axes[2].set_title("Función de Autocorrelación (ACF)")
            axes[2].legend()
            axes[2].grid(True, alpha=0.3)
        except Exception as e:
            axes[2].text(
                0.5,
                0.5,
                f"No se pudo calcular ACF:\n{str(e)}",
                ha="center",
                va="center",
                transform=axes[2].transAxes,
            )

        plt.tight_layout()

        if save:
            filename = self.output_dir / f"timeseries_{feature_name}.png"
            fig.savefig(filename, dpi=300, bbox_inches="tight")
            print(f"  Gráfico guardado: {filename}")

        return fig

    def plot_performance_metrics(
        self,
        reference_data: pd.DataFrame,
        current_data: pd.DataFrame,
        save: bool = True,
    ) -> Figure:
        """
        Visualiza métricas de performance del modelo.

        Parameters
        ----------
        reference_data : pd.DataFrame
            Datos de referencia con y_true y y_pred
        current_data : pd.DataFrame
            Datos actuales con y_true y y_pred
        save : bool, optional
            Si guardar el gráfico

        Returns
        -------
        Figure
            Figura de matplotlib
        """
        if "y_true" not in current_data.columns or "y_pred" not in current_data.columns:
            print("  Advertencia: y_true/y_pred no encontrados, saltando gráfico de performance")
            return None

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 1. Scatter plot: Predicciones vs Reales (Baseline)
        if "y_true" in reference_data.columns and "y_pred" in reference_data.columns:
            axes[0, 0].scatter(
                reference_data["y_true"],
                reference_data["y_pred"],
                alpha=0.3,
                s=10,
                label="Baseline",
            )
            min_val = min(reference_data["y_true"].min(), reference_data["y_pred"].min())
            max_val = max(reference_data["y_true"].max(), reference_data["y_pred"].max())
            axes[0, 0].plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2)
            axes[0, 0].set_xlabel("Valores Reales")
            axes[0, 0].set_ylabel("Predicciones")
            axes[0, 0].set_title("Baseline: Predicciones vs Reales")
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)

        # 2. Scatter plot: Predicciones vs Reales (Producción)
        axes[0, 1].scatter(
            current_data["y_true"], current_data["y_pred"], alpha=0.3, s=10, label="Producción"
        )
        min_val = min(current_data["y_true"].min(), current_data["y_pred"].min())
        max_val = max(current_data["y_true"].max(), current_data["y_pred"].max())
        axes[0, 1].plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2)
        axes[0, 1].set_xlabel("Valores Reales")
        axes[0, 1].set_ylabel("Predicciones")
        axes[0, 1].set_title("Producción: Predicciones vs Reales")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # 3. Distribución de errores
        ref_errors = reference_data["y_true"] - reference_data["y_pred"] if "y_true" in reference_data.columns else []
        curr_errors = current_data["y_true"] - current_data["y_pred"]

        if len(ref_errors) > 0:
            axes[1, 0].hist(ref_errors, bins=30, alpha=0.5, label="Baseline", color="blue", density=True)
        axes[1, 0].hist(curr_errors, bins=30, alpha=0.5, label="Producción", color="red", density=True)
        axes[1, 0].axvline(x=0, color="k", linestyle="--", linewidth=1)
        axes[1, 0].set_xlabel("Error (Real - Predicho)")
        axes[1, 0].set_ylabel("Densidad")
        axes[1, 0].set_title("Distribución de Errores")
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # 4. Métricas comparativas
        from sklearn.metrics import mean_absolute_error, mean_squared_error

        metrics_data = []

        if len(ref_errors) > 0:
            ref_rmse = np.sqrt(mean_squared_error(reference_data["y_true"], reference_data["y_pred"]))
            ref_mae = mean_absolute_error(reference_data["y_true"], reference_data["y_pred"])
            ref_mape = np.mean(
                np.abs((reference_data["y_true"] - reference_data["y_pred"]) / reference_data["y_true"])
            ) * 100
            metrics_data.append(["Baseline", ref_rmse, ref_mae, ref_mape])

        curr_rmse = np.sqrt(mean_squared_error(current_data["y_true"], current_data["y_pred"]))
        curr_mae = mean_absolute_error(current_data["y_true"], current_data["y_pred"])
        curr_mape = np.mean(
            np.abs((current_data["y_true"] - current_data["y_pred"]) / current_data["y_true"])
        ) * 100
        metrics_data.append(["Producción", curr_rmse, curr_mae, curr_mape])

        metrics_df = pd.DataFrame(metrics_data, columns=["Período", "RMSE", "MAE", "MAPE"])

        x = np.arange(len(metrics_df))
        width = 0.25

        axes[1, 1].bar(x - width, metrics_df["RMSE"], width, label="RMSE", alpha=0.8)
        axes[1, 1].bar(x, metrics_df["MAE"], width, label="MAE", alpha=0.8)
        axes[1, 1].bar(x + width, metrics_df["MAPE"], width, label="MAPE", alpha=0.8)
        axes[1, 1].set_xlabel("Período")
        axes[1, 1].set_ylabel("Valor de Métrica")
        axes[1, 1].set_title("Comparación de Métricas")
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(metrics_df["Período"])
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save:
            filename = self.output_dir / "performance_comparison.png"
            fig.savefig(filename, dpi=300, bbox_inches="tight")
            print(f"  Gráfico guardado: {filename}")

        return fig

    def plot_drift_summary(
        self,
        alerts: List[DriftAlert],
        save: bool = True,
    ) -> Figure:
        """
        Crea resumen visual de alertas de drift.

        Parameters
        ----------
        alerts : List[DriftAlert]
            Lista de alertas detectadas
        save : bool, optional
            Si guardar el gráfico

        Returns
        -------
        Figure
            Figura de matplotlib
        """
        if not alerts:
            print("  No hay alertas para visualizar")
            return None

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 1. Alertas por severidad
        severity_counts = {}
        for alert in alerts:
            sev = alert.severity.value
            severity_counts[sev] = severity_counts.get(sev, 0) + 1

        colors_severity = {
            "none": "green",
            "low": "yellow",
            "medium": "orange",
            "high": "red",
            "critical": "darkred",
        }

        severities = list(severity_counts.keys())
        counts = list(severity_counts.values())
        colors = [colors_severity.get(s, "gray") for s in severities]

        axes[0, 0].bar(severities, counts, color=colors, alpha=0.7)
        axes[0, 0].set_xlabel("Severidad")
        axes[0, 0].set_ylabel("Cantidad de Alertas")
        axes[0, 0].set_title("Alertas por Severidad")
        axes[0, 0].grid(True, alpha=0.3, axis="y")

        # 2. Alertas por tipo de drift
        drift_type_counts = {}
        for alert in alerts:
            dtype = alert.drift_type.value
            drift_type_counts[dtype] = drift_type_counts.get(dtype, 0) + 1

        types = list(drift_type_counts.keys())
        type_counts = list(drift_type_counts.values())

        axes[0, 1].pie(type_counts, labels=types, autopct="%1.1f%%", startangle=90)
        axes[0, 1].set_title("Distribución por Tipo de Drift")

        # 3. Top 10 features con drift
        feature_alerts = [a for a in alerts if a.drift_type == DriftType.FEATURE_DRIFT]

        if feature_alerts:
            feature_scores = {}
            for alert in feature_alerts:
                metric = alert.metric_name.replace("_psi", "").replace("_ks", "").replace("_js", "")
                score = alert.metadata.get("psi", 0) + alert.metadata.get("js_divergence", 0)
                feature_scores[metric] = max(feature_scores.get(metric, 0), score)

            # Ordenar y tomar top 10
            sorted_features = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)[
                :10
            ]
            features = [f[0] for f in sorted_features]
            scores = [f[1] for f in sorted_features]

            axes[1, 0].barh(features, scores, color="steelblue", alpha=0.7)
            axes[1, 0].set_xlabel("Score de Drift (PSI + JS)")
            axes[1, 0].set_ylabel("Feature")
            axes[1, 0].set_title("Top 10 Features con Mayor Drift")
            axes[1, 0].grid(True, alpha=0.3, axis="x")
        else:
            axes[1, 0].text(
                0.5, 0.5, "No hay alertas de feature drift", ha="center", va="center"
            )

        # 4. Timeline de severidad
        severity_order = {"none": 0, "low": 1, "medium": 2, "high": 3, "critical": 4}

        severity_numeric = [severity_order.get(a.severity.value, 0) for a in alerts]
        severity_labels = [a.severity.value for a in alerts]

        axes[1, 1].scatter(range(len(alerts)), severity_numeric, c=severity_numeric,
                          cmap="YlOrRd", s=100, alpha=0.6)
        axes[1, 1].set_xlabel("Índice de Alerta")
        axes[1, 1].set_ylabel("Nivel de Severidad")
        axes[1, 1].set_yticks(range(5))
        axes[1, 1].set_yticklabels(["none", "low", "medium", "high", "critical"])
        axes[1, 1].set_title("Severidad de Alertas Detectadas")
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save:
            filename = self.output_dir / "drift_summary.png"
            fig.savefig(filename, dpi=300, bbox_inches="tight")
            print(f"  Gráfico guardado: {filename}")

        return fig

    def create_drift_report_plots(
        self,
        reference_data: pd.DataFrame,
        current_data: pd.DataFrame,
        alerts: List[DriftAlert],
        top_n_features: int = 5,
    ) -> Dict[str, Figure]:
        """
        Genera conjunto completo de gráficos para reporte de drift.

        Parameters
        ----------
        reference_data : pd.DataFrame
            Datos de referencia
        current_data : pd.DataFrame
            Datos actuales
        alerts : List[DriftAlert]
            Lista de alertas detectadas
        top_n_features : int, optional
            Número de features con mayor drift a graficar, por defecto 5

        Returns
        -------
        Dict[str, Figure]
            Diccionario de figuras generadas
        """
        print("\n Generando gráficos de drift...")

        figures = {}

        # 1. Resumen de drift
        print("  1. Resumen de alertas...")
        fig_summary = self.plot_drift_summary(alerts, save=True)
        if fig_summary:
            figures["summary"] = fig_summary

        # 2. Performance (si hay datos)
        if "y_true" in current_data.columns and "y_pred" in current_data.columns:
            print("  2. Métricas de performance...")
            fig_perf = self.plot_performance_metrics(reference_data, current_data, save=True)
            if fig_perf:
                figures["performance"] = fig_perf

        # 3. Top N features con mayor drift
        feature_alerts = [a for a in alerts if a.drift_type == DriftType.FEATURE_DRIFT]

        if feature_alerts:
            # Calcular scores y ordenar
            feature_scores = {}
            for alert in feature_alerts:
                metric = alert.metric_name.replace("_psi", "").replace("_ks", "").replace("_js", "")
                score = alert.metadata.get("psi", 0) + alert.metadata.get("js_divergence", 0)
                feature_scores[metric] = max(feature_scores.get(metric, 0), score)

            top_features = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)[
                :top_n_features
            ]

            print(f"  3. Distribuciones de top {len(top_features)} features...")
            for i, (feature, score) in enumerate(top_features, 1):
                if feature in reference_data.columns and feature in current_data.columns:
                    print(f"     {i}. {feature} (score: {score:.3f})...")
                    fig = self.plot_distribution_comparison(
                        reference_data, current_data, feature, save=True
                    )
                    figures[f"distribution_{feature}"] = fig
                    plt.close(fig)  # Cerrar para liberar memoria

        # 4. Series temporales (si hay columnas temporales)
        temporal_features = [
            col
            for col in current_data.columns
            if "power_consumption" in col.lower() or "consumption" in col.lower()
        ]

        if temporal_features:
            print(f"  4. Series temporales...")
            for feature in temporal_features[:2]:  # Solo primeras 2 para no saturar
                print(f"     - {feature}...")
                fig = self.plot_timeseries_comparison(
                    reference_data, current_data, feature, save=True
                )
                figures[f"timeseries_{feature}"] = fig
                plt.close(fig)

        print(f"\n Total de gráficos generados: {len(figures)}")
        print(f" Gráficos guardados en: {self.output_dir}")

        return figures


def create_visualizations(
    reference_data: pd.DataFrame,
    current_data: pd.DataFrame,
    alerts: List[DriftAlert],
    output_dir: Optional[Path] = None,
) -> Dict[str, Figure]:
    """
    Función de conveniencia para crear visualizaciones de drift.

    Parameters
    ----------
    reference_data : pd.DataFrame
        Datos de referencia
    current_data : pd.DataFrame
        Datos actuales
    alerts : List[DriftAlert]
        Lista de alertas
    output_dir : Path, optional
        Directorio de salida

    Returns
    -------
    Dict[str, Figure]
        Diccionario de figuras generadas
    """
    visualizer = DriftVisualizer(output_dir=output_dir)
    return visualizer.create_drift_report_plots(reference_data, current_data, alerts)
