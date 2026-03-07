from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class MetricTarget:
    """Define qué se medirá para evaluar una acción."""
    key: str
    label: str
    direction: str  # "up", "down", "stable"
    min_relative_change: Optional[float] = None
    min_absolute_change: Optional[float] = None


@dataclass
class ExperimentPlan:
    """Unidad operativa mínima de una recomendación."""
    rec_id: str
    title: str
    hypothesis: str
    action: str
    where_to_apply: str
    duration_days: int
    primary_metric: MetricTarget
    secondary_metrics: List[MetricTarget] = field(default_factory=list)
    success_rule_text: str = ""
    confidence_label: str = "media"