"""Streaming Telemetry Anomaly Detection & Non-Parametric Hypothesis Testing Engine."""

import sys
from pathlib import Path

repo_root = str(Path(__file__).resolve().parent.parent.parent.parent)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from experiments.domain_discovery.statistics_and_information.benchmark import (
    StreamingTelemetryAnomalyDetector,
    run_streaming_telemetry_benchmark,
)

__all__ = [
    "StreamingTelemetryAnomalyDetector",
    "run_streaming_telemetry_benchmark",
]
