"""
Pipeline de Monitoreo de Drift para MLOps.

Este módulo proporciona el pipeline principal de orquestación para monitoreo de
data drift, integrando múltiples detectores de drift y canales de alerta.

Clases
------
DriftMonitoringPipeline
    Pipeline principal para orquestar detección de drift y alertas
MonitoringDataLoader
    Maneja carga y preprocesamiento de datos de monitoreo
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from .alert_system import (
    AlertManager,
    ConsoleAlertChannel,
    DriftMonitoringReport,
    FileAlertChannel,
)
from .drift_detectors import (
    DriftAlert,
    DriftDetector,
    ModelPerformanceMonitor,
    StatisticalDriftDetector,
    TimeSeriesDriftDetector,
)


class MonitoringDataLoader:
    """
    Handles loading and preprocessing of monitoring data.

    Parameters
    ----------
    reference_path : Path, optional
        Path to reference/baseline data
    current_path : Path, optional
        Path to current/production data
    """

    def __init__(
        self, reference_path: Optional[Path] = None, current_path: Optional[Path] = None
    ):
        """Initialize monitoring data loader."""
        self.reference_path = reference_path
        self.current_path = current_path

    def load_reference_data(self, path: Optional[Path] = None) -> pd.DataFrame:
        """
        Load reference/baseline data.

        Parameters
        ----------
        path : Path, optional
            Path to reference data file (CSV or Parquet)

        Returns
        -------
        pd.DataFrame
            Reference data
        """
        file_path = path or self.reference_path
        if file_path is None:
            raise ValueError("No reference data path provided")

        return self._load_data(file_path)

    def load_current_data(self, path: Optional[Path] = None) -> pd.DataFrame:
        """
        Load current/production data.

        Parameters
        ----------
        path : Path, optional
            Path to current data file (CSV or Parquet)

        Returns
        -------
        pd.DataFrame
            Current data
        """
        file_path = path or self.current_path
        if file_path is None:
            raise ValueError("No current data path provided")

        return self._load_data(file_path)

    def _load_data(self, path: Path) -> pd.DataFrame:
        """Load data from CSV or Parquet file."""
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Data file not found: {path}")

        if path.suffix == ".csv":
            return pd.read_csv(path, parse_dates=True)
        elif path.suffix == ".parquet":
            return pd.read_parquet(path)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")

    def create_windowed_data(
        self, data: pd.DataFrame, window_size: int, step_size: Optional[int] = None
    ) -> List[pd.DataFrame]:
        """
        Create sliding windows from time series data.

        Parameters
        ----------
        data : pd.DataFrame
            Input time series data
        window_size : int
            Size of each window
        step_size : int, optional
            Step size between windows, by default window_size (non-overlapping)

        Returns
        -------
        List[pd.DataFrame]
            List of windowed dataframes
        """
        if step_size is None:
            step_size = window_size

        windows = []
        for i in range(0, len(data) - window_size + 1, step_size):
            window = data.iloc[i : i + window_size]
            windows.append(window)

        return windows


class DriftMonitoringPipeline:
    """
    Main pipeline for orchestrating drift detection and alerting.

    This pipeline integrates multiple drift detectors (statistical, time series,
    performance) and manages alert routing through configured channels.

    Parameters
    ----------
    detectors : List[DriftDetector], optional
        List of drift detectors to use
    alert_manager : AlertManager, optional
        Alert manager for routing alerts
    output_dir : Path, optional
        Directory for output reports and artifacts
    enable_statistical : bool, optional
        Enable statistical drift detection, by default True
    enable_timeseries : bool, optional
        Enable time series drift detection, by default True
    enable_performance : bool, optional
        Enable performance monitoring, by default True
    """

    def __init__(
        self,
        detectors: Optional[List[DriftDetector]] = None,
        alert_manager: Optional[AlertManager] = None,
        output_dir: Optional[Path] = None,
        enable_statistical: bool = True,
        enable_timeseries: bool = True,
        enable_performance: bool = True,
    ):
        """Initialize drift monitoring pipeline."""
        self.output_dir = Path(output_dir) if output_dir else Path("reports/drift_monitoring")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize detectors
        self.detectors: List[DriftDetector] = detectors or []
        if not self.detectors:
            if enable_statistical:
                self.detectors.append(
                    StatisticalDriftDetector(
                        ks_threshold=0.05, psi_threshold=0.2, js_threshold=0.1
                    )
                )
            if enable_timeseries:
                self.detectors.append(
                    TimeSeriesDriftDetector(seasonal_period=144, autocorr_lags=24)
                )
            if enable_performance:
                self.detectors.append(
                    ModelPerformanceMonitor(window_size=144, performance_threshold=0.15)
                )

        # Initialize alert manager
        self.alert_manager = alert_manager or self._create_default_alert_manager()

        # Data loader
        self.data_loader = MonitoringDataLoader()

    def _create_default_alert_manager(self) -> AlertManager:
        """Create default alert manager with console and file channels."""
        console_channel = ConsoleAlertChannel(colored=True)
        file_channel = FileAlertChannel(self.output_dir / "drift_alerts.json")

        return AlertManager(channels=[console_channel, file_channel])

    def run(
        self,
        reference_data: pd.DataFrame,
        current_data: pd.DataFrame,
        reference_period: str = "baseline",
        current_period: str = "current",
    ) -> DriftMonitoringReport:
        """
        Run the complete drift monitoring pipeline.

        Parameters
        ----------
        reference_data : pd.DataFrame
            Baseline/training data
        current_data : pd.DataFrame
            Current/production data
        reference_period : str, optional
            Description of reference period, by default "baseline"
        current_period : str, optional
            Description of current period, by default "current"

        Returns
        -------
        DriftMonitoringReport
            Comprehensive drift monitoring report
        """
        all_alerts: List[DriftAlert] = []

        # Run each detector
        for detector in self.detectors:
            print(f"\n Running {detector.name}...")

            if isinstance(detector, ModelPerformanceMonitor):
                # Performance monitoring requires y_true and y_pred columns
                if "y_true" in current_data.columns and "y_pred" in current_data.columns:
                    alerts = detector.detect(reference_data, current_data)
                    all_alerts.extend(alerts)
                else:
                    print(f"    Skipping {detector.name}: y_true/y_pred columns not found")
            else:
                alerts = detector.detect(reference_data, current_data)
                all_alerts.extend(alerts)

            print(f"  > {detector.name} detected {len(alerts)} alerts")

        # Send alerts through channels
        print(f"\n Sending {len(all_alerts)} alerts through channels...")
        send_results = self.alert_manager.send_alerts(all_alerts)

        for channel_name, count in send_results.items():
            print(f"  > {channel_name}: {count} alerts sent")

        # Create report
        report = DriftMonitoringReport(
            alerts=all_alerts,
            reference_period=reference_period,
            current_period=current_period,
        )

        # Save report
        report_path = self.output_dir / "drift_monitoring_report.json"
        report.save(report_path)
        print(f"\n Report saved to: {report_path}")

        # Print summary
        report.print_summary()

        return report

    def run_from_files(
        self,
        reference_path: Path,
        current_path: Path,
        reference_period: str = "baseline",
        current_period: str = "current",
    ) -> DriftMonitoringReport:
        """
        Run pipeline loading data from files.

        Parameters
        ----------
        reference_path : Path
            Path to reference data file
        current_path : Path
            Path to current data file
        reference_period : str, optional
            Description of reference period
        current_period : str, optional
            Description of current period

        Returns
        -------
        DriftMonitoringReport
            Comprehensive drift monitoring report
        """
        print(" Loading data...")
        reference_data = self.data_loader.load_reference_data(reference_path)
        current_data = self.data_loader.load_current_data(current_path)

        print(f"  Reference data: {len(reference_data)} records")
        print(f"  Current data: {len(current_data)} records")

        return self.run(reference_data, current_data, reference_period, current_period)

    def run_with_performance_data(
        self,
        reference_features: pd.DataFrame,
        current_features: pd.DataFrame,
        reference_predictions: pd.DataFrame,
        current_predictions: pd.DataFrame,
    ) -> DriftMonitoringReport:
        """
        Run pipeline with separate feature and prediction data.

        Parameters
        ----------
        reference_features : pd.DataFrame
            Reference feature data
        current_features : pd.DataFrame
            Current feature data
        reference_predictions : pd.DataFrame
            Reference predictions with y_true and y_pred columns
        current_predictions : pd.DataFrame
            Current predictions with y_true and y_pred columns

        Returns
        -------
        DriftMonitoringReport
            Comprehensive drift monitoring report
        """
        # Merge features with predictions
        reference_data = reference_features.copy()
        if "y_true" in reference_predictions.columns:
            reference_data["y_true"] = reference_predictions["y_true"]
        if "y_pred" in reference_predictions.columns:
            reference_data["y_pred"] = reference_predictions["y_pred"]

        current_data = current_features.copy()
        if "y_true" in current_predictions.columns:
            current_data["y_true"] = current_predictions["y_true"]
        if "y_pred" in current_predictions.columns:
            current_data["y_pred"] = current_predictions["y_pred"]

        return self.run(reference_data, current_data)

    def add_detector(self, detector: DriftDetector) -> None:
        """Add a drift detector to the pipeline."""
        self.detectors.append(detector)

    def remove_detector(self, detector_name: str) -> None:
        """Remove a drift detector by name."""
        self.detectors = [d for d in self.detectors if d.name != detector_name]

    def get_detector_names(self) -> List[str]:
        """Get names of all configured detectors."""
        return [d.name for d in self.detectors]

    def get_alert_summary(self) -> Dict[str, int]:
        """
        Get summary of alerts from all detectors.

        Returns
        -------
        Dict[str, int]
            Dictionary mapping detector names to alert counts
        """
        return {detector.name: len(detector.get_alerts()) for detector in self.detectors}


def create_default_pipeline(output_dir: Optional[Path] = None) -> DriftMonitoringPipeline:
    """
    Create a drift monitoring pipeline with default configuration.

    This is a convenience function for quickly setting up a pipeline with
    sensible defaults for the Tetouan power consumption project.

    Parameters
    ----------
    output_dir : Path, optional
        Output directory for reports, by default "reports/drift_monitoring"

    Returns
    -------
    DriftMonitoringPipeline
        Configured drift monitoring pipeline
    """
    return DriftMonitoringPipeline(
        output_dir=output_dir,
        enable_statistical=True,
        enable_timeseries=True,
        enable_performance=True,
    )
