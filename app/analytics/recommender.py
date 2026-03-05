# app/analytics/recommender.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List

from .insights import Insight
from .history_store import HistoryStore
from .ranking import ScoredInsight, rank_insights as _rank_insights, select_plan as _select_plan


@dataclass(frozen=True)
class RankedInsight:
    """
    Compat legacy: mismo concepto que ScoredInsight, pero con nombre antiguo.
    """
    insight: Insight
    score: float
    reason: str


def rank_insights(insights: List[Insight], history: HistoryStore, *, period_key: str) -> List[RankedInsight]:
    """
    Wrapper legacy sobre app/analytics/ranking.py.
    Mantiene API simple y garantiza una sola fuente de verdad.
    """
    scored: List[ScoredInsight] = _rank_insights(insights, history, period_key=period_key)
    return [RankedInsight(insight=s.insight, score=s.score, reason=s.reason) for s in scored]


def pick_plan(ranked: List[RankedInsight], k: int = 3) -> List[RankedInsight]:
    """
    Wrapper legacy: aplica la misma lógica de diversidad usando select_plan del ranking real.
    """
    # Convertimos a ScoredInsight para reaprovechar select_plan (diversidad por kpi_target, etc.)
    scored = [ScoredInsight(insight=r.insight, score=r.score, reason=r.reason) for r in ranked]
    picked = _select_plan(scored, k=k, require_revenue_for_owner=True)
    return [RankedInsight(insight=s.insight, score=s.score, reason=s.reason) for s in picked]