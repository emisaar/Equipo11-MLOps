"""
Clases de Detección de Data Drift para Monitoreo MLOps de Series de Tiempo.

Este módulo implementa una arquitectura OOP comprehensiva para detectar data drift
en pipelines de pronóstico de series de tiempo. Proporciona detectores especializados
para cambios en distribuciones estadísticas, degradación del rendimiento del modelo,
y patrones específicos de series de tiempo.

Clases
------
DriftDetector
    Clase base abstracta para todos los detectores de drift
StatisticalDriftDetector
    Detecta cambios en distribuciones usando test KS, PSI y divergencia JS
TimeSeriesDriftDetector
    Detecta drift en patrones temporales, estacionalidad y autocorrelación
ModelPerformanceMonitor
    Monitorea métricas de rendimiento del modelo con ventanas deslizantes
DriftAlert
    Encapsula alertas de detección de drift con niveles de severidad
DriftMonitoringReport
    Agrega resultados de detección de drift en reportes comprehensivos
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import jensenshannon
from scipy.stats import ks_2samp
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf, adfuller


class DriftSeverity(Enum):
    """Enumeración de niveles de severidad de drift."""

    NONE = "none"          # Sin drift
    LOW = "low"            # Drift bajo
    MEDIUM = "medium"      # Drift medio
    HIGH = "high"          # Drift alto
    CRITICAL = "critical"  # Drift crítico


class DriftType(Enum):
    """Enumeración de tipos de drift."""

    FEATURE_DRIFT = "feature_drift"           # Drift en features/variables
    LABEL_DRIFT = "label_drift"               # Drift en etiquetas/targets
    CONCEPT_DRIFT = "concept_drift"           # Drift conceptual (relación feature-target)
    PERFORMANCE_DRIFT = "performance_drift"   # Drift en rendimiento del modelo
    TEMPORAL_DRIFT = "temporal_drift"         # Drift en patrones temporales


@dataclass
class DriftAlert:
    """
    Encapsula una alerta de detección de drift.

    Atributos
    ---------
    drift_type : DriftType
        Tipo de drift detectado
    severity : DriftSeverity
        Nivel de severidad del drift
    metric_name : str
        Nombre de la métrica que activó la alerta
    baseline_value : float
        Valor de la métrica en los datos de referencia/baseline
    current_value : float
        Valor de la métrica en los datos actuales/producción
    threshold : float
        Umbral que fue excedido
    timestamp : datetime
        Momento en que se generó la alerta
    message : str
        Descripción legible de la alerta
    metadata : Dict[str, Any]
        Metadatos adicionales sobre la alerta
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
        """Convierte la alerta a formato diccionario."""
        return {
            "drift_type": self.drift_type.value,
            "severity": self.severity.value,
            "metric_name": self.metric_name,
            "baseline_value": float(self.baseline_value),
            "current_value": float(self.current_value),
            "threshold": float(self.threshold),
            "timestamp": self.timestamp.isoformat(),
            "message": self.message,
            "metadata": self.metadata,
        }


class DriftDetector(ABC):
    """
    Clase base abstracta para detectores de drift.

    Todos los detectores de drift deben implementar el método detect() que compara
    datos de referencia (baseline) contra datos actuales (producción).

    Parámetros
    ----------
    name : str
        Nombre del detector de drift
    thresholds : Dict[str, float]
        Diccionario de valores de umbral para diferentes métricas
    """

    def __init__(self, name: str, thresholds: Optional[Dict[str, float]] = None):
        """Inicializa el detector de drift."""
        self.name = name
        self.thresholds = thresholds or {}
        self._alerts: List[DriftAlert] = []

    @abstractmethod
    def detect(
        self, reference_data: pd.DataFrame, current_data: pd.DataFrame
    ) -> List[DriftAlert]:
        """
        Detect drift by comparing reference and current data.

        Parameters
        ----------
        reference_data : pd.DataFrame
            Baseline/training data
        current_data : pd.DataFrame
            Current/production data

        Returns
        -------
        List[DriftAlert]
            List of drift alerts detected
        """
        pass

    def get_alerts(self) -> List[DriftAlert]:
        """Return all alerts generated by this detector."""
        return self._alerts

    def clear_alerts(self) -> None:
        """Clear all stored alerts."""
        self._alerts = []

    def _determine_severity(self, score: float, thresholds: Dict[str, float]) -> DriftSeverity:
        """
        Determine severity based on drift score and thresholds.

        Parameters
        ----------
        score : float
            Drift score
        thresholds : Dict[str, float]
            Dictionary with keys 'low', 'medium', 'high', 'critical'

        Returns
        -------
        DriftSeverity
            Severity level
        """
        if score >= thresholds.get("critical", 0.5):
            return DriftSeverity.CRITICAL
        elif score >= thresholds.get("high", 0.3):
            return DriftSeverity.HIGH
        elif score >= thresholds.get("medium", 0.2):
            return DriftSeverity.MEDIUM
        elif score >= thresholds.get("low", 0.1):
            return DriftSeverity.LOW
        else:
            return DriftSeverity.NONE


class StatisticalDriftDetector(DriftDetector):
    """
    Detects drift using statistical tests.

    Implements multiple statistical methods:
    - Kolmogorov-Smirnov test for continuous distributions
    - Population Stability Index (PSI) for binned distributions
    - Jensen-Shannon divergence for distribution similarity

    Parameters
    ----------
    name : str, optional
        Name of the detector, by default "StatisticalDriftDetector"
    ks_threshold : float, optional
        Threshold for KS test p-value, by default 0.05
    psi_threshold : float, optional
        Threshold for PSI score, by default 0.2
    js_threshold : float, optional
        Threshold for JS divergence, by default 0.1
    n_bins : int, optional
        Number of bins for PSI calculation, by default 10
    """

    def __init__(
        self,
        name: str = "StatisticalDriftDetector",
        ks_threshold: float = 0.05,
        psi_threshold: float = 0.2,
        js_threshold: float = 0.1,
        n_bins: int = 10,
    ):
        """Initialize statistical drift detector."""
        thresholds = {
            "ks_pvalue": ks_threshold,
            "psi": psi_threshold,
            "js_divergence": js_threshold,
        }
        super().__init__(name, thresholds)
        self.n_bins = n_bins

    def detect(
        self, reference_data: pd.DataFrame, current_data: pd.DataFrame
    ) -> List[DriftAlert]:
        """
        Detect statistical drift in features.

        Parameters
        ----------
        reference_data : pd.DataFrame
            Baseline data
        current_data : pd.DataFrame
            Current data

        Returns
        -------
        List[DriftAlert]
            List of drift alerts
        """
        self.clear_alerts()

        # Get numeric columns only
        numeric_cols = reference_data.select_dtypes(include=[np.number]).columns
        common_cols = set(numeric_cols) & set(current_data.columns)

        for col in common_cols:
            ref_values = reference_data[col].dropna()
            curr_values = current_data[col].dropna()

            if len(ref_values) == 0 or len(curr_values) == 0:
                continue

            # Kolmogorov-Smirnov test
            ks_stat, ks_pvalue = self._ks_test(ref_values, curr_values)

            # Population Stability Index
            psi_score = self._calculate_psi(ref_values, curr_values)

            # Jensen-Shannon divergence
            js_div = self._calculate_js_divergence(ref_values, curr_values)

            # Check thresholds and create alerts
            if ks_pvalue < self.thresholds["ks_pvalue"]:
                alert = DriftAlert(
                    drift_type=DriftType.FEATURE_DRIFT,
                    severity=self._determine_severity(
                        1 - ks_pvalue, {"low": 0.7, "medium": 0.85, "high": 0.95, "critical": 0.99}
                    ),
                    metric_name=f"{col}_ks_test",
                    baseline_value=ks_stat,
                    current_value=ks_pvalue,
                    threshold=self.thresholds["ks_pvalue"],
                    message=f"KS test detected significant distribution change in {col} (p-value: {ks_pvalue:.4f})",
                    metadata={"test": "kolmogorov_smirnov", "statistic": ks_stat},
                )
                self._alerts.append(alert)

            if psi_score > self.thresholds["psi"]:
                alert = DriftAlert(
                    drift_type=DriftType.FEATURE_DRIFT,
                    severity=self._determine_severity(
                        psi_score, {"low": 0.1, "medium": 0.2, "high": 0.3, "critical": 0.5}
                    ),
                    metric_name=f"{col}_psi",
                    baseline_value=0.0,
                    current_value=psi_score,
                    threshold=self.thresholds["psi"],
                    message=f"PSI detected significant shift in {col} (PSI: {psi_score:.4f})",
                    metadata={"test": "population_stability_index"},
                )
                self._alerts.append(alert)

            if js_div > self.thresholds["js_divergence"]:
                alert = DriftAlert(
                    drift_type=DriftType.FEATURE_DRIFT,
                    severity=self._determine_severity(
                        js_div, {"low": 0.1, "medium": 0.2, "high": 0.3, "critical": 0.5}
                    ),
                    metric_name=f"{col}_js_divergence",
                    baseline_value=0.0,
                    current_value=js_div,
                    threshold=self.thresholds["js_divergence"],
                    message=f"JS divergence detected distribution difference in {col} (JS: {js_div:.4f})",
                    metadata={"test": "jensen_shannon_divergence"},
                )
                self._alerts.append(alert)

        return self._alerts

    def _ks_test(self, ref_values: pd.Series, curr_values: pd.Series) -> Tuple[float, float]:
        """
        Perform Kolmogorov-Smirnov test.

        Returns
        -------
        Tuple[float, float]
            KS statistic and p-value
        """
        statistic, pvalue = ks_2samp(ref_values, curr_values)
        return float(statistic), float(pvalue)

    def _calculate_psi(self, ref_values: pd.Series, curr_values: pd.Series) -> float:
        """
        Calculate Population Stability Index (PSI).

        PSI = sum((actual% - expected%) * ln(actual% / expected%))

        Returns
        -------
        float
            PSI score
        """
        # Create bins based on reference data quantiles
        try:
            bins = np.histogram_bin_edges(ref_values, bins=self.n_bins)
            ref_hist, _ = np.histogram(ref_values, bins=bins)
            curr_hist, _ = np.histogram(curr_values, bins=bins)

            # Convert to proportions
            ref_prop = ref_hist / len(ref_values)
            curr_prop = curr_hist / len(curr_values)

            # Avoid division by zero
            ref_prop = np.where(ref_prop == 0, 0.0001, ref_prop)
            curr_prop = np.where(curr_prop == 0, 0.0001, curr_prop)

            # Calculate PSI
            psi = np.sum((curr_prop - ref_prop) * np.log(curr_prop / ref_prop))
            return float(psi)
        except Exception:
            return 0.0

    def _calculate_js_divergence(self, ref_values: pd.Series, curr_values: pd.Series) -> float:
        """
        Calculate Jensen-Shannon divergence.

        Returns
        -------
        float
            JS divergence (0 to 1, where 0 is identical)
        """
        try:
            bins = np.histogram_bin_edges(ref_values, bins=self.n_bins)
            ref_hist, _ = np.histogram(ref_values, bins=bins)
            curr_hist, _ = np.histogram(curr_values, bins=bins)

            # Convert to probability distributions
            ref_prob = ref_hist / ref_hist.sum()
            curr_prob = curr_hist / curr_hist.sum()

            # Avoid zeros
            ref_prob = np.where(ref_prob == 0, 1e-10, ref_prob)
            curr_prob = np.where(curr_prob == 0, 1e-10, curr_prob)

            # Calculate JS divergence
            js_div = jensenshannon(ref_prob, curr_prob)
            return float(js_div)
        except Exception:
            return 0.0


class TimeSeriesDriftDetector(DriftDetector):
    """
    Detects drift specific to time series data.

    Monitors changes in:
    - Temporal patterns and trends
    - Seasonality components
    - Autocorrelation structure
    - Stationarity properties

    Parameters
    ----------
    name : str, optional
        Name of the detector, by default "TimeSeriesDriftDetector"
    seasonal_period : int, optional
        Period for seasonal decomposition, by default 144 (1 day for 10-min data)
    autocorr_lags : int, optional
        Number of lags for autocorrelation analysis, by default 24
    adf_threshold : float, optional
        Threshold for ADF test p-value, by default 0.05
    """

    def __init__(
        self,
        name: str = "TimeSeriesDriftDetector",
        seasonal_period: int = 144,
        autocorr_lags: int = 24,
        adf_threshold: float = 0.05,
    ):
        """Initialize time series drift detector."""
        thresholds = {
            "autocorr_change": 0.3,
            "trend_change": 0.2,
            "seasonality_change": 0.25,
            "adf_pvalue": adf_threshold,
        }
        super().__init__(name, thresholds)
        self.seasonal_period = seasonal_period
        self.autocorr_lags = autocorr_lags

    def detect(
        self, reference_data: pd.DataFrame, current_data: pd.DataFrame
    ) -> List[DriftAlert]:
        """
        Detect time series-specific drift.

        Parameters
        ----------
        reference_data : pd.DataFrame
            Baseline time series data
        current_data : pd.DataFrame
            Current time series data

        Returns
        -------
        List[DriftAlert]
            List of drift alerts
        """
        self.clear_alerts()

        # Get numeric columns
        numeric_cols = reference_data.select_dtypes(include=[np.number]).columns
        common_cols = set(numeric_cols) & set(current_data.columns)

        for col in common_cols:
            ref_series = reference_data[col].dropna()
            curr_series = current_data[col].dropna()

            if len(ref_series) < self.seasonal_period * 2 or len(curr_series) < self.seasonal_period * 2:
                continue

            # Autocorrelation drift
            self._detect_autocorr_drift(col, ref_series, curr_series)

            # Stationarity drift
            self._detect_stationarity_drift(col, ref_series, curr_series)

            # Seasonal pattern drift
            self._detect_seasonal_drift(col, ref_series, curr_series)

        return self._alerts

    def _detect_autocorr_drift(
        self, col: str, ref_series: pd.Series, curr_series: pd.Series
    ) -> None:
        """Detect changes in autocorrelation structure."""
        try:
            ref_acf = acf(ref_series, nlags=self.autocorr_lags, fft=True)
            curr_acf = acf(curr_series, nlags=self.autocorr_lags, fft=True)

            # Calculate mean absolute difference
            acf_diff = np.mean(np.abs(ref_acf - curr_acf))

            if acf_diff > self.thresholds["autocorr_change"]:
                alert = DriftAlert(
                    drift_type=DriftType.TEMPORAL_DRIFT,
                    severity=self._determine_severity(
                        acf_diff, {"low": 0.2, "medium": 0.3, "high": 0.4, "critical": 0.5}
                    ),
                    metric_name=f"{col}_autocorr_drift",
                    baseline_value=0.0,
                    current_value=acf_diff,
                    threshold=self.thresholds["autocorr_change"],
                    message=f"Autocorrelation structure changed in {col} (diff: {acf_diff:.4f})",
                    metadata={"ref_acf": ref_acf.tolist(), "curr_acf": curr_acf.tolist()},
                )
                self._alerts.append(alert)
        except Exception as e:
            # Skip if autocorrelation calculation fails
            pass

    def _detect_stationarity_drift(
        self, col: str, ref_series: pd.Series, curr_series: pd.Series
    ) -> None:
        """Detect changes in stationarity using Augmented Dickey-Fuller test."""
        try:
            ref_adf = adfuller(ref_series, autolag="AIC")
            curr_adf = adfuller(curr_series, autolag="AIC")

            ref_stationary = ref_adf[1] < self.thresholds["adf_pvalue"]
            curr_stationary = curr_adf[1] < self.thresholds["adf_pvalue"]

            # Alert if stationarity property changed
            if ref_stationary != curr_stationary:
                alert = DriftAlert(
                    drift_type=DriftType.TEMPORAL_DRIFT,
                    severity=DriftSeverity.MEDIUM,
                    metric_name=f"{col}_stationarity_drift",
                    baseline_value=ref_adf[1],
                    current_value=curr_adf[1],
                    threshold=self.thresholds["adf_pvalue"],
                    message=f"Stationarity changed in {col} (ref stationary: {ref_stationary}, curr: {curr_stationary})",
                    metadata={
                        "ref_adf_statistic": ref_adf[0],
                        "curr_adf_statistic": curr_adf[0],
                    },
                )
                self._alerts.append(alert)
        except Exception:
            pass

    def _detect_seasonal_drift(
        self, col: str, ref_series: pd.Series, curr_series: pd.Series
    ) -> None:
        """Detect changes in seasonal patterns."""
        try:
            # Decompose both series
            ref_decomp = seasonal_decompose(
                ref_series, model="additive", period=self.seasonal_period, extrapolate_trend="freq"
            )
            curr_decomp = seasonal_decompose(
                curr_series, model="additive", period=self.seasonal_period, extrapolate_trend="freq"
            )

            # Compare seasonal components
            min_len = min(len(ref_decomp.seasonal), len(curr_decomp.seasonal))
            ref_seasonal = ref_decomp.seasonal[:min_len]
            curr_seasonal = curr_decomp.seasonal[:min_len]

            seasonal_diff = np.mean(np.abs(ref_seasonal - curr_seasonal))
            seasonal_scale = np.std(ref_seasonal)

            if seasonal_scale > 0:
                normalized_diff = seasonal_diff / seasonal_scale

                if normalized_diff > self.thresholds["seasonality_change"]:
                    alert = DriftAlert(
                        drift_type=DriftType.TEMPORAL_DRIFT,
                        severity=self._determine_severity(
                            normalized_diff,
                            {"low": 0.2, "medium": 0.25, "high": 0.35, "critical": 0.5},
                        ),
                        metric_name=f"{col}_seasonality_drift",
                        baseline_value=0.0,
                        current_value=normalized_diff,
                        threshold=self.thresholds["seasonality_change"],
                        message=f"Seasonal pattern changed in {col} (normalized diff: {normalized_diff:.4f})",
                        metadata={"seasonal_diff": float(seasonal_diff)},
                    )
                    self._alerts.append(alert)
        except Exception:
            pass


class ModelPerformanceMonitor(DriftDetector):
    """
    Monitors model performance metrics over time.

    Tracks degradation in predictive performance using sliding windows
    and compares against baseline metrics.

    Parameters
    ----------
    name : str, optional
        Name of the monitor, by default "ModelPerformanceMonitor"
    window_size : int, optional
        Size of sliding window for metrics calculation, by default 144
    performance_threshold : float, optional
        Relative threshold for performance degradation, by default 0.15
    """

    def __init__(
        self,
        name: str = "ModelPerformanceMonitor",
        window_size: int = 144,
        performance_threshold: float = 0.15,
    ):
        """Initialize model performance monitor."""
        thresholds = {"performance_degradation": performance_threshold}
        super().__init__(name, thresholds)
        self.window_size = window_size

    def detect(
        self,
        reference_data: pd.DataFrame,
        current_data: pd.DataFrame,
        y_true_col: str = "y_true",
        y_pred_col: str = "y_pred",
    ) -> List[DriftAlert]:
        """
        Detect performance drift.

        Parameters
        ----------
        reference_data : pd.DataFrame
            Baseline data with true and predicted values
        current_data : pd.DataFrame
            Current data with true and predicted values
        y_true_col : str, optional
            Column name for true values, by default "y_true"
        y_pred_col : str, optional
            Column name for predicted values, by default "y_pred"

        Returns
        -------
        List[DriftAlert]
            List of performance drift alerts
        """
        self.clear_alerts()

        # Calculate baseline metrics
        ref_metrics = self._calculate_metrics(
            reference_data[y_true_col], reference_data[y_pred_col]
        )

        # Calculate current metrics
        curr_metrics = self._calculate_metrics(current_data[y_true_col], current_data[y_pred_col])

        # Check for degradation
        for metric_name in ["rmse", "mae", "mape"]:
            baseline_value = ref_metrics[metric_name]
            current_value = curr_metrics[metric_name]

            if baseline_value > 0:
                relative_change = (current_value - baseline_value) / baseline_value

                if relative_change > self.thresholds["performance_degradation"]:
                    alert = DriftAlert(
                        drift_type=DriftType.PERFORMANCE_DRIFT,
                        severity=self._determine_severity(
                            relative_change,
                            {"low": 0.1, "medium": 0.2, "high": 0.3, "critical": 0.5},
                        ),
                        metric_name=metric_name.upper(),
                        baseline_value=baseline_value,
                        current_value=current_value,
                        threshold=self.thresholds["performance_degradation"],
                        message=f"{metric_name.upper()} degraded by {relative_change*100:.1f}% "
                        f"(baseline: {baseline_value:.2f}, current: {current_value:.2f})",
                        metadata={"relative_change": relative_change},
                    )
                    self._alerts.append(alert)

        return self._alerts

    def _calculate_metrics(self, y_true: pd.Series, y_pred: pd.Series) -> Dict[str, float]:
        """Calculate performance metrics."""
        y_true_clean = y_true.dropna()
        y_pred_clean = y_pred[y_true_clean.index]

        rmse = np.sqrt(mean_squared_error(y_true_clean, y_pred_clean))
        mae = mean_absolute_error(y_true_clean, y_pred_clean)

        # MAPE calculation
        mask = y_true_clean != 0
        if mask.any():
            mape = np.mean(np.abs((y_true_clean[mask] - y_pred_clean[mask]) / y_true_clean[mask])) * 100
        else:
            mape = 0.0

        return {"rmse": float(rmse), "mae": float(mae), "mape": float(mape)}
