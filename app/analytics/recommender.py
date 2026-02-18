from __future__ import annotations

from dataclasses import dataclass

from .insights import Insight
from .history import HistoryStore


@dataclass(frozen=True)
class RankedInsight:
    insight: Insight
    score: float
    reason: str


def _impact_bonus(impact_range: tuple[float, float]) -> float:
    """
    Convierte impacto (€) a bonus pequeño y estable (0..0.25 aprox).
    No queremos que el impacto domine la confianza.
    """
    low, high = impact_range
    mid = (low + high) / 2.0
    # Escala suave: 0€ -> 0, 500€ -> ~0.15, 1500€ -> ~0.25
    if mid <= 0:
        return 0.0
    if mid >= 1500:
        return 0.25
    return 0.25 * (mid / 1500.0)


def rank_insights(insights: list[Insight], history: HistoryStore) -> list[RankedInsight]:
    """
    Score = confianza + bonus ingresos + bonus impacto - repetición + outcome_bonus.
    """
    ranked: list[RankedInsight] = []

    for ins in insights:
        base = float(ins.confidence)

        bonus_revenue = 0.12 if ins.impact_type == "revenue" else 0.0
        bonus_impact = _impact_bonus(ins.estimated_impact_eur)

        penalty_repeat = history.repetition_penalty(ins.id, last_n=2)
        bonus_outcome = history.outcome_bonus(ins.id, lookback=6)

        score = base + bonus_revenue + bonus_impact - penalty_repeat + bonus_outcome

        reason = (
            f"conf={base:.2f}"
            f"{'+rev' if bonus_revenue else ''}"
            f"+imp({bonus_impact:.2f})"
        )
        if penalty_repeat:
            reason += f"-rep({penalty_repeat:.2f})"
        if bonus_outcome:
            reason += f"+hist({bonus_outcome:.2f})"

        ranked.append(RankedInsight(insight=ins, score=float(score), reason=reason))

    ranked.sort(key=lambda x: x.score, reverse=True)
    return ranked


def pick_plan(ranked: list[RankedInsight], k: int = 3) -> list[RankedInsight]:
    """
    Selecciona k insights evitando duplicados.
    """
    chosen: list[RankedInsight] = []
    seen = set()

    for r in ranked:
        if r.insight.id in seen:
            continue
        chosen.append(r)
        seen.add(r.insight.id)
        if len(chosen) >= k:
            break

    return chosen