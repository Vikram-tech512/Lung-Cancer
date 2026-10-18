"""
backend/monitoring/grafana.py
================================
Grafana dashboard provisioning helper.

Generates the Grafana dashboard JSON for the Lung Cancer AI Platform
and writes it to the standard Grafana provisioning directory.

Dashboard panels
----------------
1. Inference Latency Histogram   (p50, p95, p99 per model)
2. Inference Request Rate        (req/s per model × outcome)
3. Active Cases Gauge
4. Model Accuracy Gauges         (AUC / Sensitivity / Specificity)
5. Training Loss Curves          (train vs val per model)
6. Training AUC Curves
7. GPU Memory Utilisation
8. DICOM QA Gate Pass Rate
9. API Request Rate & Error Rate
10. Request Latency (HTTP p95)

Usage
-----
>>> from backend.monitoring.grafana import generate_dashboard
>>> json_path = generate_dashboard(output_dir="deploy/grafana/dashboards")
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

DASHBOARD_UID  = "lung-cancer-ai-platform"
DASHBOARD_TITLE = "Lung Cancer AI Platform"


def generate_dashboard(output_dir: str = "deploy/grafana/dashboards") -> Path:
    """
    Generate Grafana dashboard JSON and write to *output_dir*.
    Returns the path of the written file.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dashboard = _build_dashboard()
    out_path  = out_dir / "lung_cancer_ai.json"
    out_path.write_text(json.dumps(dashboard, indent=2))
    print(f"[grafana] Dashboard written -> {out_path}")
    return out_path


def _build_dashboard() -> Dict[str, Any]:
    panels = []
    panel_id = 1

    # ---------------------------------------------------------------
    # Row 1: Inference metrics
    # ---------------------------------------------------------------
    panels.append(_row(panel_id, "🧠 Inference Metrics", 0))
    panel_id += 1

    panels.append(_histogram_panel(
        panel_id, "Inference Latency (ms)",
        'histogram_quantile(0.95, sum(rate(lung_cancer_inference_latency_ms_bucket[5m])) by (le, model))',
        x=0, y=1, w=12, h=8,
    ))
    panel_id += 1

    panels.append(_stat_panel(
        panel_id, "Total Inferences",
        'sum(increase(lung_cancer_inference_total[1h]))',
        x=12, y=1, w=6, h=4,
    ))
    panel_id += 1

    panels.append(_gauge_panel(
        panel_id, "Active Cases",
        'lung_cancer_active_cases',
        x=18, y=1, w=6, h=4,
    ))
    panel_id += 1

    # ---------------------------------------------------------------
    # Row 2: Model Performance
    # ---------------------------------------------------------------
    panels.append(_row(panel_id, "📊 Model Performance", 9))
    panel_id += 1

    for metric in ("auc", "sensitivity", "specificity"):
        panels.append(_time_series_panel(
            panel_id,
            f"Model {metric.upper()} ({metric})",
            f'lung_cancer_model_accuracy{{metric="{metric}"}}',
            x=0 + (panels.__len__() % 3) * 8, y=10, w=8, h=6,
        ))
        panel_id += 1

    # ---------------------------------------------------------------
    # Row 3: Training curves
    # ---------------------------------------------------------------
    panels.append(_row(panel_id, "🏋️ Training Pipeline", 17))
    panel_id += 1

    panels.append(_time_series_panel(
        panel_id, "Training Loss (train vs val)",
        'lung_cancer_training_epoch_loss',
        x=0, y=18, w=12, h=8,
    ))
    panel_id += 1

    panels.append(_time_series_panel(
        panel_id, "Training AUC (train vs val)",
        'lung_cancer_training_epoch_auc',
        x=12, y=18, w=12, h=8,
    ))
    panel_id += 1

    # ---------------------------------------------------------------
    # Row 4: Infrastructure
    # ---------------------------------------------------------------
    panels.append(_row(panel_id, "🖥️ Infrastructure", 27))
    panel_id += 1

    panels.append(_time_series_panel(
        panel_id, "GPU Memory (bytes)",
        'lung_cancer_gpu_memory_bytes',
        x=0, y=28, w=8, h=6,
    ))
    panel_id += 1

    panels.append(_time_series_panel(
        panel_id, "API Request Rate (req/s)",
        'rate(lung_cancer_inference_total{model="http"}[1m])',
        x=8, y=28, w=8, h=6,
    ))
    panel_id += 1

    panels.append(_pie_panel(
        panel_id, "DICOM QA Outcomes",
        'lung_cancer_dicom_validation_total',
        x=16, y=28, w=8, h=6,
    ))
    panel_id += 1

    return {
        "uid":         DASHBOARD_UID,
        "title":       DASHBOARD_TITLE,
        "description": "Real-time monitoring for the Lung Cancer AI Platform",
        "tags":        ["lung-cancer", "ai", "medical-imaging"],
        "timezone":    "browser",
        "refresh":     "10s",
        "time":        {"from": "now-1h", "to": "now"},
        "panels":      panels,
        "schemaVersion": 38,
        "version":     1,
        "__requires": [{"type": "datasource", "name": "Prometheus", "version": "1.0.0"}],
    }


# ---------------------------------------------------------------------------
# Panel builders
# ---------------------------------------------------------------------------
def _row(uid: int, title: str, y: int) -> Dict:
    return {
        "id": uid, "type": "row", "title": title,
        "gridPos": {"x": 0, "y": y, "w": 24, "h": 1}, "collapsed": False,
    }


def _time_series_panel(uid: int, title: str, expr: str, x=0, y=0, w=12, h=8) -> Dict:
    return {
        "id": uid, "type": "timeseries", "title": title,
        "gridPos": {"x": x, "y": y, "w": w, "h": h},
        "targets": [{"expr": expr, "legendFormat": "{{model}} {{split}}", "refId": "A"}],
        "fieldConfig": {"defaults": {"color": {"mode": "palette-classic"}}},
    }


def _histogram_panel(uid: int, title: str, expr: str, x=0, y=0, w=12, h=8) -> Dict:
    return {
        "id": uid, "type": "timeseries", "title": title,
        "gridPos": {"x": x, "y": y, "w": w, "h": h},
        "targets": [{"expr": expr, "legendFormat": "p95 {{model}}", "refId": "A"}],
        "fieldConfig": {"defaults": {"unit": "ms"}},
    }


def _stat_panel(uid: int, title: str, expr: str, x=0, y=0, w=6, h=4) -> Dict:
    return {
        "id": uid, "type": "stat", "title": title,
        "gridPos": {"x": x, "y": y, "w": w, "h": h},
        "targets": [{"expr": expr, "refId": "A"}],
        "options": {"reduceOptions": {"calcs": ["lastNotNull"]}},
    }


def _gauge_panel(uid: int, title: str, expr: str, x=0, y=0, w=6, h=4) -> Dict:
    return {
        "id": uid, "type": "gauge", "title": title,
        "gridPos": {"x": x, "y": y, "w": w, "h": h},
        "targets": [{"expr": expr, "refId": "A"}],
        "fieldConfig": {"defaults": {"min": 0, "max": 50, "thresholds": {
            "mode": "absolute",
            "steps": [
                {"color": "green", "value": None},
                {"color": "yellow", "value": 20},
                {"color": "red", "value": 40},
            ],
        }}},
    }


def _pie_panel(uid: int, title: str, expr: str, x=0, y=0, w=8, h=6) -> Dict:
    return {
        "id": uid, "type": "piechart", "title": title,
        "gridPos": {"x": x, "y": y, "w": w, "h": h},
        "targets": [{"expr": expr, "legendFormat": "{{outcome}}", "refId": "A"}],
    }
