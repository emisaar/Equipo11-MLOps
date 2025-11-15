"""
Data Drift Monitoring Module for MLOps Pipeline.

This module provides comprehensive data drift detection and monitoring
capabilities for time series forecasting models.

Main Components
---------------
drift_detectors
    Core drift detection classes (Statistical, TimeSeries, Performance)
alert_system
    Alert management and notification system
drift_pipeline
    Main orchestration pipeline

Quick Start
-----------
>>> from src.monitoring import create_default_pipeline
>>>
>>> # Create pipeline with default configuration
>>> pipeline = create_default_pipeline()
>>>
>>> # Run monitoring
>>> report = pipeline.run_from_files(
...     reference_path="data/processed/train.parquet",
...     current_path="data/processed/test.parquet"
... )

Examples
--------
Using individual detectors:

>>> from src.monitoring.drift_detectors import StatisticalDriftDetector
>>>
>>> detector = StatisticalDriftDetector(ks_threshold=0.05, psi_threshold=0.2)
>>> alerts = detector.detect(reference_data, current_data)

Custom pipeline configuration:

>>> from src.monitoring import DriftMonitoringPipeline
>>> from src.monitoring.alert_system import AlertManager, ConsoleAlertChannel
>>>
>>> alert_manager = AlertManager(channels=[ConsoleAlertChannel()])
>>> pipeline = DriftMonitoringPipeline(alert_manager=alert_manager)
>>> report = pipeline.run(reference_data, current_data)
"""

from .alert_system import (
    AlertChannel,
    AlertManager,
    ConsoleAlertChannel,
    DriftMonitoringReport,
    FileAlertChannel,
)
from .drift_detectors import (
    DriftAlert,
    DriftDetector,
    DriftSeverity,
    DriftType,
    ModelPerformanceMonitor,
    StatisticalDriftDetector,
    TimeSeriesDriftDetector,
)
from .drift_pipeline import (
    DriftMonitoringPipeline,
    MonitoringDataLoader,
    create_default_pipeline,
)
from .drift_visualization import (
    DriftVisualizer,
    create_visualizations,
)

__all__ = [
    # Core detectors
    "DriftDetector",
    "StatisticalDriftDetector",
    "TimeSeriesDriftDetector",
    "ModelPerformanceMonitor",
    # Alert types
    "DriftAlert",
    "DriftSeverity",
    "DriftType",
    # Alert system
    "AlertChannel",
    "AlertManager",
    "ConsoleAlertChannel",
    "FileAlertChannel",
    "DriftMonitoringReport",
    # Pipeline
    "DriftMonitoringPipeline",
    "MonitoringDataLoader",
    "create_default_pipeline",
    # Visualization
    "DriftVisualizer",
    "create_visualizations",
]

__version__ = "2.0.0"
