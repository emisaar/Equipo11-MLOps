"""
Integración de Monitoreo de Drift para FastAPI.

Este módulo integra el sistema de monitoreo de drift con la API de predicción FastAPI,
permitiendo detección de drift en tiempo real y registro de predicciones para monitoreo.

Clases
------
PredictionLogger
    Registra predicciones y valores reales para monitoreo de drift
RealTimeDriftMonitor
    Monitorea drift en tiempo real basado en predicciones registradas
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from src.monitoring import (
    DriftMonitoringPipeline,
    DriftMonitoringReport,
    StatisticalDriftDetector,
    ModelPerformanceMonitor,
    AlertManager,
    ConsoleAlertChannel,
    FileAlertChannel,
)

logger = logging.getLogger(__name__)


class PredictionLogger:
    """
    Logs predictions and features for drift monitoring.

    Parameters
    ----------
    log_dir : Path
        Directory to store prediction logs
    max_log_size : int, optional
        Maximum number of records to keep in memory, by default 10000
    """

    def __init__(self, log_dir: Path = Path("logs/predictions"), max_log_size: int = 10000):
        """Initialize prediction logger."""
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.max_log_size = max_log_size
        self._predictions_buffer: List[Dict] = []

    def log_prediction(
        self,
        zone: int,
        model_type: str,
        features: Dict[str, float],
        prediction: float,
        timestamp: Optional[datetime] = None,
    ) -> None:
        """
        Log a prediction with its features.

        Parameters
        ----------
        zone : int
            Zone number (1, 2, or 3)
        model_type : str
            Model type used
        features : Dict[str, float]
            Feature values used for prediction
        prediction : float
            Predicted value
        timestamp : datetime, optional
            Timestamp of prediction, by default current time
        """
        if timestamp is None:
            timestamp = datetime.now()

        log_entry = {
            "timestamp": timestamp.isoformat(),
            "zone": zone,
            "model_type": model_type,
            "prediction": prediction,
            **features,
        }

        self._predictions_buffer.append(log_entry)

        # Flush if buffer is full
        if len(self._predictions_buffer) >= self.max_log_size:
            self.flush()

    def log_actual_value(
        self, timestamp: datetime, zone: int, actual_value: float
    ) -> None:
        """
        Log actual value to match with predictions.

        Parameters
        ----------
        timestamp : datetime
            Timestamp matching the prediction
        zone : int
            Zone number
        actual_value : float
            Actual observed value
        """
        # Load existing actuals
        actuals_file = self.log_dir / f"actuals_zone_{zone}.jsonl"

        actual_entry = {
            "timestamp": timestamp.isoformat(),
            "zone": zone,
            "actual_value": actual_value,
        }

        # Append to JSONL file
        with open(actuals_file, "a") as f:
            f.write(json.dumps(actual_entry) + "\n")

    def flush(self) -> None:
        """Flush prediction buffer to disk."""
        if not self._predictions_buffer:
            return

        # Group by zone and model_type
        df = pd.DataFrame(self._predictions_buffer)

        for (zone, model_type), group in df.groupby(["zone", "model_type"]):
            log_file = self.log_dir / f"predictions_zone_{zone}_{model_type}.jsonl"

            # Append to JSONL file
            with open(log_file, "a") as f:
                for record in group.to_dict("records"):
                    f.write(json.dumps(record) + "\n")

        self._predictions_buffer = []

    def load_predictions(
        self,
        zone: int,
        model_type: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """
        Load logged predictions for a specific zone and model.

        Parameters
        ----------
        zone : int
            Zone number
        model_type : str
            Model type
        start_time : datetime, optional
            Start time filter
        end_time : datetime, optional
            End time filter

        Returns
        -------
        pd.DataFrame
            Logged predictions
        """
        log_file = self.log_dir / f"predictions_zone_{zone}_{model_type}.jsonl"

        if not log_file.exists():
            return pd.DataFrame()

        # Read JSONL file
        records = []
        with open(log_file, "r") as f:
            for line in f:
                records.append(json.loads(line))

        if not records:
            return pd.DataFrame()

        df = pd.DataFrame(records)
        df["timestamp"] = pd.to_datetime(df["timestamp"])

        # Apply time filters
        if start_time:
            df = df[df["timestamp"] >= start_time]
        if end_time:
            df = df[df["timestamp"] <= end_time]

        return df

    def merge_predictions_with_actuals(
        self, zone: int, model_type: str, tolerance_minutes: int = 10
    ) -> pd.DataFrame:
        """
        Merge predictions with actual values based on timestamp proximity.

        Parameters
        ----------
        zone : int
            Zone number
        model_type : str
            Model type
        tolerance_minutes : int, optional
            Maximum time difference for matching, by default 10 minutes

        Returns
        -------
        pd.DataFrame
            Merged predictions and actuals with y_pred and y_true columns
        """
        # Load predictions
        predictions = self.load_predictions(zone, model_type)

        if predictions.empty:
            return pd.DataFrame()

        # Load actuals
        actuals_file = self.log_dir / f"actuals_zone_{zone}.jsonl"

        if not actuals_file.exists():
            return pd.DataFrame()

        # Read actuals
        actuals_records = []
        with open(actuals_file, "r") as f:
            for line in f:
                actuals_records.append(json.loads(line))

        if not actuals_records:
            return pd.DataFrame()

        actuals = pd.DataFrame(actuals_records)
        actuals["timestamp"] = pd.to_datetime(actuals["timestamp"])

        # Merge on approximate timestamp
        merged = pd.merge_asof(
            predictions.sort_values("timestamp"),
            actuals.sort_values("timestamp"),
            on="timestamp",
            direction="nearest",
            tolerance=pd.Timedelta(minutes=tolerance_minutes),
            suffixes=("_pred", "_actual"),
        )

        # Rename columns for drift monitoring
        merged["y_pred"] = merged["prediction"]
        merged["y_true"] = merged["actual_value"]

        return merged.dropna(subset=["y_pred", "y_true"])


class RealTimeDriftMonitor:
    """
    Real-time drift monitoring for production predictions.

    Parameters
    ----------
    prediction_logger : PredictionLogger
        Logger for accessing prediction history
    reference_data_path : Path, optional
        Path to reference/baseline data
    monitoring_window_hours : int, optional
        Window size for monitoring in hours, by default 24
    check_interval_hours : int, optional
        How often to check for drift in hours, by default 6
    """

    def __init__(
        self,
        prediction_logger: PredictionLogger,
        reference_data_path: Optional[Path] = None,
        monitoring_window_hours: int = 24,
        check_interval_hours: int = 6,
    ):
        """Initialize real-time drift monitor."""
        self.prediction_logger = prediction_logger
        self.reference_data_path = reference_data_path
        self.monitoring_window_hours = monitoring_window_hours
        self.check_interval_hours = check_interval_hours
        self.last_check_time: Optional[datetime] = None

        # Create output directory if it doesn't exist
        self.output_dir = Path("reports/realtime_drift_monitoring")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize monitoring pipeline
        self.pipeline = DriftMonitoringPipeline(
            output_dir=self.output_dir,
            enable_statistical=True,
            enable_timeseries=True,  # Enable for comprehensive drift detection
            enable_performance=True,
        )

    def should_check_drift(self) -> bool:
        """
        Determine if it's time to check for drift.

        Returns
        -------
        bool
            True if drift check should be performed
        """
        if self.last_check_time is None:
            return True

        time_since_last_check = datetime.now() - self.last_check_time
        return time_since_last_check >= timedelta(hours=self.check_interval_hours)

    def check_drift(
        self, zone: int, model_type: str
    ) -> Optional[DriftMonitoringReport]:
        """
        Check for drift in recent predictions.

        Parameters
        ----------
        zone : int
            Zone number
        model_type : str
            Model type

        Returns
        -------
        Optional[DriftMonitoringReport]
            Drift monitoring report, or None if insufficient data
        """
        # Load reference data
        if self.reference_data_path and self.reference_data_path.exists():
            reference_data = pd.read_parquet(self.reference_data_path)
        else:
            # Use older predictions as reference
            lookback_time = datetime.now() - timedelta(days=7)
            reference_start = lookback_time - timedelta(hours=self.monitoring_window_hours)
            reference_data = self.prediction_logger.load_predictions(
                zone, model_type, reference_start, lookback_time
            )

        # Load recent predictions
        current_end = datetime.now()
        current_start = current_end - timedelta(hours=self.monitoring_window_hours)
        current_data = self.prediction_logger.load_predictions(
            zone, model_type, current_start, current_end
        )

        # Check if we have enough data
        if len(reference_data) < 50 or len(current_data) < 50:
            return None

        # Merge with actuals for performance monitoring
        current_with_actuals = self.prediction_logger.merge_predictions_with_actuals(
            zone, model_type
        )

        if not current_with_actuals.empty and len(current_with_actuals) >= 50:
            # Use performance monitoring
            reference_with_actuals = reference_data
            if "y_true" not in reference_data.columns:
                # Skip performance monitoring if no actuals in reference
                current_data_for_drift = current_data
            else:
                current_data_for_drift = current_with_actuals
                reference_with_actuals = reference_data
        else:
            current_data_for_drift = current_data
            reference_with_actuals = reference_data

        # Run drift detection
        report = self.pipeline.run(
            reference_with_actuals,
            current_data_for_drift,
            reference_period=f"7 days ago ({self.monitoring_window_hours}h window)",
            current_period=f"Last {self.monitoring_window_hours} hours",
        )

        self.last_check_time = datetime.now()

        return report

    def get_drift_status(self, zone: int, model_type: str) -> Dict[str, any]:
        """
        Get current drift monitoring status.

        Parameters
        ----------
        zone : int
            Zone number
        model_type : str
            Model type

        Returns
        -------
        Dict[str, any]
            Status information
        """
        # Check if drift check is needed
        needs_check = self.should_check_drift()

        # Get latest report if exists
        report_file = self.output_dir / "drift_monitoring_report.json"
        latest_report = None

        if report_file.exists():
            try:
                report_data = json.loads(report_file.read_text())
                latest_report = report_data.get("summary")
            except Exception as e:
                logger.warning(f"Error loading drift report: {e}")

        return {
            "zone": zone,
            "model_type": model_type,
            "needs_drift_check": needs_check,
            "last_check_time": self.last_check_time.isoformat() if self.last_check_time else None,
            "next_check_in_hours": (
                self.check_interval_hours
                - (datetime.now() - self.last_check_time).total_seconds() / 3600
                if self.last_check_time
                else 0
            ),
            "latest_report_summary": latest_report,
        }


# Global instances (initialized in FastAPI lifespan)
prediction_logger: Optional[PredictionLogger] = None
drift_monitor: Optional[RealTimeDriftMonitor] = None


def initialize_monitoring(reference_data_path: Optional[Path] = None) -> None:
    """
    Initialize monitoring components.

    Parameters
    ----------
    reference_data_path : Path, optional
        Path to reference data for drift comparison
    """
    global prediction_logger, drift_monitor

    prediction_logger = PredictionLogger(max_log_size=50)  # Vaciar buffer cada 50 predicciones
    drift_monitor = RealTimeDriftMonitor(
        prediction_logger=prediction_logger,
        reference_data_path=reference_data_path,
        monitoring_window_hours=24,
        check_interval_hours=6,
    )


def get_prediction_logger() -> PredictionLogger:
    """Get the global prediction logger instance."""
    if prediction_logger is None:
        raise RuntimeError("Monitoring not initialized. Call initialize_monitoring() first.")
    return prediction_logger


def get_drift_monitor() -> RealTimeDriftMonitor:
    """Get the global drift monitor instance."""
    if drift_monitor is None:
        raise RuntimeError("Monitoring not initialized. Call initialize_monitoring() first.")
    return drift_monitor
