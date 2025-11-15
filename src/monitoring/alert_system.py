"""
Sistema de Alertas para Monitoreo de Data Drift.

Este módulo proporciona un sistema de alertas configurable para manejar alertas
de detección de drift, incluyendo canales de notificación, agregación de alertas
y generación de reportes.

Clases
------
AlertChannel
    Clase base abstracta para canales de notificación de alertas
ConsoleAlertChannel
    Imprime alertas en consola
FileAlertChannel
    Escribe alertas en archivo JSON
AlertManager
    Gestiona múltiples canales de alerta y enruta alertas
DriftMonitoringReport
    Agrega alertas de drift en reportes comprehensivos
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from .drift_detectors import DriftAlert, DriftSeverity, DriftType


class AlertChannel(ABC):
    """
    Abstract base class for alert notification channels.

    Alert channels define how drift alerts are communicated to users
    or external systems (console, file, email, Slack, etc.).

    Parameters
    ----------
    name : str
        Name of the alert channel
    min_severity : DriftSeverity, optional
        Minimum severity level to trigger alerts, by default MEDIUM
    """

    def __init__(self, name: str, min_severity: DriftSeverity = DriftSeverity.MEDIUM):
        """Initialize alert channel."""
        self.name = name
        self.min_severity = min_severity
        self.enabled = True

    @abstractmethod
    def send_alert(self, alert: DriftAlert) -> bool:
        """
        Send a single alert through this channel.

        Parameters
        ----------
        alert : DriftAlert
            Alert to send

        Returns
        -------
        bool
            True if alert was sent successfully
        """
        pass

    def should_send(self, alert: DriftAlert) -> bool:
        """
        Determine if alert should be sent based on severity.

        Parameters
        ----------
        alert : DriftAlert
            Alert to check

        Returns
        -------
        bool
            True if alert meets minimum severity threshold
        """
        if not self.enabled:
            return False

        severity_order = {
            DriftSeverity.NONE: 0,
            DriftSeverity.LOW: 1,
            DriftSeverity.MEDIUM: 2,
            DriftSeverity.HIGH: 3,
            DriftSeverity.CRITICAL: 4,
        }

        return severity_order[alert.severity] >= severity_order[self.min_severity]

    def enable(self) -> None:
        """Enable this alert channel."""
        self.enabled = True

    def disable(self) -> None:
        """Disable this alert channel."""
        self.enabled = False


class ConsoleAlertChannel(AlertChannel):
    """
    Alert channel that prints to console.

    Parameters
    ----------
    name : str, optional
        Channel name, by default "console"
    min_severity : DriftSeverity, optional
        Minimum severity to print, by default LOW
    colored : bool, optional
        Use ANSI color codes for output, by default True
    """

    def __init__(
        self,
        name: str = "console",
        min_severity: DriftSeverity = DriftSeverity.LOW,
        colored: bool = True,
    ):
        """Initialize console alert channel."""
        super().__init__(name, min_severity)
        self.colored = colored
        self.colors = {
            DriftSeverity.NONE: "",
            DriftSeverity.LOW: "\033[94m",  # Blue
            DriftSeverity.MEDIUM: "\033[93m",  # Yellow
            DriftSeverity.HIGH: "\033[91m",  # Red
            DriftSeverity.CRITICAL: "\033[95m",  # Magenta
        }
        self.reset_color = "\033[0m"

    def send_alert(self, alert: DriftAlert) -> bool:
        """Print alert to console."""
        if not self.should_send(alert):
            return False

        color = self.colors[alert.severity] if self.colored else ""
        reset = self.reset_color if self.colored else ""

        print(f"\n{color}[{alert.severity.value.upper()} DRIFT ALERT]{reset}")
        print(f"  Type: {alert.drift_type.value}")
        print(f"  Metric: {alert.metric_name}")
        print(f"  Baseline: {alert.baseline_value:.4f}")
        print(f"  Current: {alert.current_value:.4f}")
        print(f"  Threshold: {alert.threshold:.4f}")
        print(f"  Time: {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  Message: {alert.message}")

        return True


class FileAlertChannel(AlertChannel):
    """
    Alert channel that writes to JSON file.

    Parameters
    ----------
    output_path : Path
        Path to output JSON file
    name : str, optional
        Channel name, by default "file"
    min_severity : DriftSeverity, optional
        Minimum severity to write, by default LOW
    append : bool, optional
        Append to existing file, by default True
    """

    def __init__(
        self,
        output_path: Path,
        name: str = "file",
        min_severity: DriftSeverity = DriftSeverity.LOW,
        append: bool = True,
    ):
        """Initialize file alert channel."""
        super().__init__(name, min_severity)
        self.output_path = Path(output_path)
        self.append = append
        self._alerts_buffer: List[Dict] = []

    def send_alert(self, alert: DriftAlert) -> bool:
        """Write alert to file."""
        if not self.should_send(alert):
            return False

        self._alerts_buffer.append(alert.to_dict())
        return True

    def flush(self) -> None:
        """Write buffered alerts to file."""
        if not self._alerts_buffer:
            return

        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        existing_data = []
        if self.append and self.output_path.exists():
            try:
                existing_data = json.loads(self.output_path.read_text())
                if not isinstance(existing_data, list):
                    existing_data = []
            except json.JSONDecodeError:
                existing_data = []

        all_alerts = existing_data + self._alerts_buffer

        self.output_path.write_text(json.dumps(all_alerts, indent=2))
        self._alerts_buffer = []


class AlertManager:
    """
    Manages multiple alert channels and routes drift alerts.

    Parameters
    ----------
    channels : List[AlertChannel], optional
        List of alert channels to use
    """

    def __init__(self, channels: Optional[List[AlertChannel]] = None):
        """Initialize alert manager."""
        self.channels: List[AlertChannel] = channels or []
        self._alert_history: List[DriftAlert] = []

    def add_channel(self, channel: AlertChannel) -> None:
        """Add an alert channel."""
        self.channels.append(channel)

    def remove_channel(self, channel_name: str) -> None:
        """Remove an alert channel by name."""
        self.channels = [ch for ch in self.channels if ch.name != channel_name]

    def send_alerts(self, alerts: List[DriftAlert]) -> Dict[str, int]:
        """
        Send alerts through all configured channels.

        Parameters
        ----------
        alerts : List[DriftAlert]
            List of alerts to send

        Returns
        -------
        Dict[str, int]
            Dictionary mapping channel names to number of alerts sent
        """
        results = {}

        for channel in self.channels:
            sent_count = 0
            for alert in alerts:
                if channel.send_alert(alert):
                    sent_count += 1
            results[channel.name] = sent_count

        # Store in history
        self._alert_history.extend(alerts)

        # Flush file channels
        for channel in self.channels:
            if isinstance(channel, FileAlertChannel):
                channel.flush()

        return results

    def get_alert_history(
        self,
        drift_type: Optional[DriftType] = None,
        min_severity: Optional[DriftSeverity] = None,
        limit: Optional[int] = None,
    ) -> List[DriftAlert]:
        """
        Get alert history with optional filtering.

        Parameters
        ----------
        drift_type : DriftType, optional
            Filter by drift type
        min_severity : DriftSeverity, optional
            Filter by minimum severity
        limit : int, optional
            Maximum number of alerts to return

        Returns
        -------
        List[DriftAlert]
            Filtered alert history
        """
        filtered = self._alert_history

        if drift_type:
            filtered = [a for a in filtered if a.drift_type == drift_type]

        if min_severity:
            severity_order = {
                DriftSeverity.NONE: 0,
                DriftSeverity.LOW: 1,
                DriftSeverity.MEDIUM: 2,
                DriftSeverity.HIGH: 3,
                DriftSeverity.CRITICAL: 4,
            }
            min_level = severity_order[min_severity]
            filtered = [a for a in filtered if severity_order[a.severity] >= min_level]

        if limit:
            filtered = filtered[-limit:]

        return filtered

    def clear_history(self) -> None:
        """Clear alert history."""
        self._alert_history = []


class DriftMonitoringReport:
    """
    Aggregates drift detection results into comprehensive reports.

    Parameters
    ----------
    alerts : List[DriftAlert]
        List of drift alerts
    reference_period : str, optional
        Description of reference period
    current_period : str, optional
        Description of current period
    """

    def __init__(
        self,
        alerts: List[DriftAlert],
        reference_period: str = "baseline",
        current_period: str = "current",
    ):
        """Initialize drift monitoring report."""
        self.alerts = alerts
        self.reference_period = reference_period
        self.current_period = current_period
        self.timestamp = datetime.now()

    def get_summary(self) -> Dict[str, any]:
        """
        Get summary statistics of the report.

        Returns
        -------
        Dict[str, any]
            Summary statistics
        """
        total_alerts = len(self.alerts)

        severity_counts = {severity: 0 for severity in DriftSeverity}
        drift_type_counts = {dtype: 0 for dtype in DriftType}

        for alert in self.alerts:
            severity_counts[alert.severity] += 1
            drift_type_counts[alert.drift_type] += 1

        return {
            "timestamp": self.timestamp.isoformat(),
            "reference_period": self.reference_period,
            "current_period": self.current_period,
            "total_alerts": total_alerts,
            "severity_breakdown": {k.value: v for k, v in severity_counts.items() if v > 0},
            "drift_type_breakdown": {k.value: v for k, v in drift_type_counts.items() if v > 0},
            "has_critical_alerts": severity_counts[DriftSeverity.CRITICAL] > 0,
            "has_high_alerts": severity_counts[DriftSeverity.HIGH] > 0,
            "requires_action": severity_counts[DriftSeverity.CRITICAL] > 0
            or severity_counts[DriftSeverity.HIGH] > 0,
        }

    def get_recommendations(self) -> List[str]:
        """
        Get recommended actions based on alerts.

        Returns
        -------
        List[str]
            List of recommended actions
        """
        recommendations = []
        summary = self.get_summary()

        if summary["has_critical_alerts"]:
            recommendations.append(
                " CRITICAL: Immediate model retraining required due to severe drift"
            )
            recommendations.append(
                " Investigate root cause of critical drift before deployment"
            )

        if summary["has_high_alerts"]:
            recommendations.append(
                " Schedule model retraining within 24-48 hours"
            )
            recommendations.append(
                " Review feature engineering pipeline for potential issues"
            )

        drift_types = summary.get("drift_type_breakdown", {})

        if "feature_drift" in drift_types:
            recommendations.append(
                " Feature drift detected - verify data collection processes"
            )

        if "temporal_drift" in drift_types:
            recommendations.append(
                " Temporal patterns changed - consider seasonal adjustments"
            )

        if "performance_drift" in drift_types:
            recommendations.append(
                " Model performance degraded - retrain with recent data"
            )

        if not recommendations:
            recommendations.append(" No significant drift detected - model performing as expected")

        return recommendations

    def to_dict(self) -> Dict[str, any]:
        """
        Convert report to dictionary format.

        Returns
        -------
        Dict[str, any]
            Report as dictionary
        """
        return {
            "summary": self.get_summary(),
            "recommendations": self.get_recommendations(),
            "alerts": [alert.to_dict() for alert in self.alerts],
        }

    def save(self, output_path: Path) -> None:
        """
        Save report to JSON file.

        Parameters
        ----------
        output_path : Path
            Path to output file
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(self.to_dict(), indent=2))

    def print_summary(self) -> None:
        """Print human-readable summary to console."""
        summary = self.get_summary()

        print("\n" + "=" * 70)
        print("DRIFT MONITORING REPORT")
        print("=" * 70)
        print(f"Timestamp: {summary['timestamp']}")
        print(f"Reference Period: {self.reference_period}")
        print(f"Current Period: {self.current_period}")
        print(f"\nTotal Alerts: {summary['total_alerts']}")

        if summary['severity_breakdown']:
            print("\nSeverity Breakdown:")
            for severity, count in summary['severity_breakdown'].items():
                print(f"  {severity.upper()}: {count}")

        if summary['drift_type_breakdown']:
            print("\nDrift Type Breakdown:")
            for dtype, count in summary['drift_type_breakdown'].items():
                print(f"  {dtype}: {count}")

        print("\n" + "-" * 70)
        print("RECOMMENDATIONS:")
        print("-" * 70)
        for i, rec in enumerate(self.get_recommendations(), 1):
            print(f"{i}. {rec}")

        print("=" * 70 + "\n")
