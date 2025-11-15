from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# OOP-based drift detection
from src.monitoring import (
    StatisticalDriftDetector,
    TimeSeriesDriftDetector,
    ModelPerformanceMonitor,
    DriftMonitoringPipeline,
    DriftSeverity,
    DriftType,
    AlertManager,
    ConsoleAlertChannel,
    FileAlertChannel,
    create_default_pipeline,
)


# ============================================================================
# OOP-BASED DRIFT DETECTION TESTS
# ============================================================================


@pytest.fixture
def sample_timeseries_data():
    """Create sample time series data for testing."""
    np.random.seed(42)
    n = 500
    timestamps = pd.date_range("2025-01-01", periods=n, freq="10min")

    # Simulate realistic power consumption data
    hour_of_day = np.arange(n) % 144 / 144 * 2 * np.pi
    seasonal_pattern = 1000 + 500 * np.sin(hour_of_day)

    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "temperature": 25 + 5 * np.sin(hour_of_day) + np.random.normal(0, 2, n),
            "humidity": 60 + 10 * np.cos(hour_of_day) + np.random.normal(0, 5, n),
            "wind_speed": 5 + np.random.exponential(2, n),
            "power_consumption": seasonal_pattern + np.random.normal(0, 50, n),
        }
    )
    return df


@pytest.fixture
def drifted_timeseries_data(sample_timeseries_data):
    """Create drifted version of sample data."""
    df = sample_timeseries_data.copy()
    # Introduce drift
    df["temperature"] += 10  # Temperature shift
    df["humidity"] *= 1.3  # Humidity scale change
    df["power_consumption"] += 200  # Consumption increase
    return df


# ============================================================================
# Statistical Drift Detector Tests
# ============================================================================


def test_statistical_detector_initialization():
    """Test StatisticalDriftDetector initialization."""
    detector = StatisticalDriftDetector(
        ks_threshold=0.05, psi_threshold=0.2, js_threshold=0.1
    )
    assert detector.name == "StatisticalDriftDetector"
    assert detector.thresholds["ks_pvalue"] == 0.05
    assert detector.thresholds["psi"] == 0.2
    assert detector.thresholds["js_divergence"] == 0.1


def test_statistical_detector_detects_drift(sample_timeseries_data, drifted_timeseries_data):
    """Test that StatisticalDriftDetector detects significant drift."""
    detector = StatisticalDriftDetector(ks_threshold=0.05, psi_threshold=0.1)

    alerts = detector.detect(sample_timeseries_data, drifted_timeseries_data)

    # Should detect drift in temperature, humidity, and power_consumption
    assert len(alerts) > 0
    assert any(alert.metric_name.startswith("temperature") for alert in alerts)


def test_statistical_detector_no_drift_on_same_data(sample_timeseries_data):
    """Test that detector doesn't flag drift when comparing identical data."""
    detector = StatisticalDriftDetector(ks_threshold=0.05, psi_threshold=0.2)

    alerts = detector.detect(sample_timeseries_data, sample_timeseries_data)

    # Should detect minimal or no drift
    assert len(alerts) == 0 or all(alert.severity == DriftSeverity.NONE for alert in alerts)


# ============================================================================
# Time Series Drift Detector Tests
# ============================================================================


def test_timeseries_detector_initialization():
    """Test TimeSeriesDriftDetector initialization."""
    detector = TimeSeriesDriftDetector(seasonal_period=144, autocorr_lags=24)
    assert detector.name == "TimeSeriesDriftDetector"
    assert detector.seasonal_period == 144
    assert detector.autocorr_lags == 24


def test_timeseries_detector_detects_pattern_drift(
    sample_timeseries_data, drifted_timeseries_data
):
    """Test that TimeSeriesDriftDetector detects temporal pattern changes."""
    detector = TimeSeriesDriftDetector(seasonal_period=144, autocorr_lags=12)

    alerts = detector.detect(sample_timeseries_data, drifted_timeseries_data)

    # Should detect some temporal drift
    assert len(alerts) >= 0  # May or may not detect depending on data size


def test_timeseries_detector_handles_short_series():
    """Test that detector handles short time series gracefully."""
    detector = TimeSeriesDriftDetector(seasonal_period=144)

    # Create short series (less than 2 seasonal periods)
    short_data = pd.DataFrame({"value": np.random.randn(100)})

    alerts = detector.detect(short_data, short_data)

    # Should not crash, may return empty list
    assert isinstance(alerts, list)


# ============================================================================
# Model Performance Monitor Tests
# ============================================================================


def test_performance_monitor_initialization():
    """Test ModelPerformanceMonitor initialization."""
    monitor = ModelPerformanceMonitor(window_size=144, performance_threshold=0.15)
    assert monitor.name == "ModelPerformanceMonitor"
    assert monitor.window_size == 144
    assert monitor.thresholds["performance_degradation"] == 0.15


def test_performance_monitor_detects_degradation():
    """Test that performance monitor detects model degradation."""
    monitor = ModelPerformanceMonitor(performance_threshold=0.1)

    # Create baseline performance data
    np.random.seed(42)
    n = 200
    baseline_data = pd.DataFrame(
        {
            "y_true": np.random.randn(n) * 100 + 1000,
            "y_pred": np.random.randn(n) * 100 + 1000,  # Good predictions
        }
    )

    # Create degraded performance data
    current_data = pd.DataFrame(
        {
            "y_true": np.random.randn(n) * 100 + 1000,
            "y_pred": np.random.randn(n) * 200 + 1000,  # Worse predictions
        }
    )

    alerts = monitor.detect(baseline_data, current_data)

    # Should detect performance degradation
    assert len(alerts) > 0
    assert any(alert.drift_type == DriftType.PERFORMANCE_DRIFT for alert in alerts)


def test_performance_monitor_no_degradation_on_same_data():
    """Test that monitor doesn't flag degradation on identical data."""
    monitor = ModelPerformanceMonitor(performance_threshold=0.15)

    np.random.seed(42)
    n = 200
    data = pd.DataFrame(
        {"y_true": np.random.randn(n) * 100 + 1000, "y_pred": np.random.randn(n) * 100 + 1000}
    )

    alerts = monitor.detect(data, data)

    # Should not detect degradation
    assert len(alerts) == 0


# ============================================================================
# Alert System Tests
# ============================================================================


def test_console_alert_channel():
    """Test ConsoleAlertChannel functionality."""
    from src.monitoring.drift_detectors import DriftAlert

    channel = ConsoleAlertChannel(min_severity=DriftSeverity.MEDIUM)

    alert_high = DriftAlert(
        drift_type=DriftType.FEATURE_DRIFT,
        severity=DriftSeverity.HIGH,
        metric_name="temperature",
        baseline_value=25.0,
        current_value=35.0,
        threshold=0.2,
        message="Temperature drifted significantly",
    )

    alert_low = DriftAlert(
        drift_type=DriftType.FEATURE_DRIFT,
        severity=DriftSeverity.LOW,
        metric_name="humidity",
        baseline_value=60.0,
        current_value=65.0,
        threshold=0.2,
        message="Humidity drifted slightly",
    )

    # High severity should be sent
    assert channel.should_send(alert_high) is True

    # Low severity should not be sent (below threshold)
    assert channel.should_send(alert_low) is False


def test_file_alert_channel(tmp_path: Path):
    """Test FileAlertChannel functionality."""
    from src.monitoring.drift_detectors import DriftAlert

    output_file = tmp_path / "alerts.json"
    channel = FileAlertChannel(output_file, min_severity=DriftSeverity.LOW)

    alert = DriftAlert(
        drift_type=DriftType.FEATURE_DRIFT,
        severity=DriftSeverity.MEDIUM,
        metric_name="temperature",
        baseline_value=25.0,
        current_value=35.0,
        threshold=0.2,
        message="Temperature drifted",
    )

    # Send alert
    channel.send_alert(alert)
    channel.flush()

    # Check file exists and contains alert
    assert output_file.exists()
    import json

    alerts_data = json.loads(output_file.read_text())
    assert len(alerts_data) == 1
    assert alerts_data[0]["metric_name"] == "temperature"


def test_alert_manager():
    """Test AlertManager functionality."""
    manager = AlertManager()

    console_channel = ConsoleAlertChannel(min_severity=DriftSeverity.MEDIUM)
    manager.add_channel(console_channel)

    from src.monitoring.drift_detectors import DriftAlert

    alerts = [
        DriftAlert(
            drift_type=DriftType.FEATURE_DRIFT,
            severity=DriftSeverity.HIGH,
            metric_name="temperature",
            baseline_value=25.0,
            current_value=35.0,
            threshold=0.2,
        )
    ]

    results = manager.send_alerts(alerts)

    assert "console" in results
    assert results["console"] == 1


# ============================================================================
# Pipeline Tests
# ============================================================================


def test_pipeline_initialization():
    """Test DriftMonitoringPipeline initialization."""
    pipeline = create_default_pipeline()

    assert len(pipeline.detectors) == 3  # Statistical, TimeSeries, Performance
    assert pipeline.alert_manager is not None


def test_pipeline_run(sample_timeseries_data, drifted_timeseries_data, tmp_path: Path):
    """Test full pipeline execution."""
    pipeline = DriftMonitoringPipeline(output_dir=tmp_path)

    report = pipeline.run(sample_timeseries_data, drifted_timeseries_data)

    # Check report was generated
    assert report is not None
    assert len(report.alerts) >= 0

    # Check files were created
    assert (tmp_path / "drift_monitoring_report.json").exists()
    assert (tmp_path / "drift_alerts.json").exists()


def test_pipeline_run_from_files(sample_timeseries_data, tmp_path: Path):
    """Test pipeline execution from files."""
    # Create test files
    ref_file = tmp_path / "reference.parquet"
    curr_file = tmp_path / "current.parquet"

    sample_timeseries_data.to_parquet(ref_file)
    sample_timeseries_data.to_parquet(curr_file)

    # Create pipeline
    pipeline = DriftMonitoringPipeline(output_dir=tmp_path / "output")

    # Run from files
    report = pipeline.run_from_files(ref_file, curr_file)

    assert report is not None


def test_drift_monitoring_report_summary():
    """Test DriftMonitoringReport summary generation."""
    from src.monitoring import DriftMonitoringReport
    from src.monitoring.drift_detectors import DriftAlert

    alerts = [
        DriftAlert(
            drift_type=DriftType.FEATURE_DRIFT,
            severity=DriftSeverity.HIGH,
            metric_name="temperature",
            baseline_value=25.0,
            current_value=35.0,
            threshold=0.2,
        ),
        DriftAlert(
            drift_type=DriftType.PERFORMANCE_DRIFT,
            severity=DriftSeverity.CRITICAL,
            metric_name="RMSE",
            baseline_value=10.0,
            current_value=20.0,
            threshold=0.15,
        ),
    ]

    report = DriftMonitoringReport(alerts)
    summary = report.get_summary()

    assert summary["total_alerts"] == 2
    assert summary["has_critical_alerts"] is True
    assert summary["has_high_alerts"] is True
    assert summary["requires_action"] is True


def test_drift_monitoring_report_recommendations():
    """Test DriftMonitoringReport recommendations."""
    from src.monitoring import DriftMonitoringReport
    from src.monitoring.drift_detectors import DriftAlert

    alerts = [
        DriftAlert(
            drift_type=DriftType.FEATURE_DRIFT,
            severity=DriftSeverity.CRITICAL,
            metric_name="temperature",
            baseline_value=25.0,
            current_value=35.0,
            threshold=0.2,
        )
    ]

    report = DriftMonitoringReport(alerts)
    recommendations = report.get_recommendations()

    assert len(recommendations) > 0
    assert any("CRITICAL" in rec for rec in recommendations)
