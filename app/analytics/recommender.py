from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from .history_store import HistoryStore
from .ranking import ScoredRecommendation, score_recommendations, select_plan
from .recommendation_library import (
    RecommendationCard,
    apply_history_penalty,
    build_recommendation_cards,
)


# ============================================================
# Modelo principal del motor de recomendaciones
# ============================================================
@dataclass(frozen=True)
class RecommendationResult:
    """
    Resultado final listo para UI Owner/Admin.
    """

    card: RecommendationCard
    score: float
    reason: str


@dataclass(frozen=True)
class RecommendationPlan:
    """
    Plan final preparado para la UI.
    """

    top_cards: List[RecommendationResult]
    all_cards: List[RecommendationResult]


# ============================================================
# Helpers internos
# ============================================================
def _safe_history_summary(history: Optional[HistoryStore]) -> dict:
    """
    Obtiene un resumen compacto del histórico sin romper el flujo
    si no hay store o si el método no existe.
    """
    if history is None:
        return {}

    method = getattr(history, "get_history_summary", None)
    if callable(method):
        try:
            return method() or {}
        except Exception:
            return {}

    return {}


def _to_results(scored: List[ScoredRecommendation]) -> List[RecommendationResult]:
    """
    Convierte ScoredRecommendation -> RecommendationResult.
    """
    return [
        RecommendationResult(
            card=s.card,
            score=s.score,
            reason=s.reason,
        )
        for s in scored
    ]


# ============================================================
# API principal
# ============================================================
def build_recommendation_plan(
    ctx: dict,
    business_type: str,
    subtype: str,
    history: Optional[HistoryStore] = None,
    *,
    client_id: Optional[str] = None,
    period_key: Optional[str] = None,
    top_k: int = 3,
) -> RecommendationPlan:
    """
    Construye el plan completo de recomendaciones para el dashboard.

    Flujo:
    1) genera cards candidatas según contexto + vertical
    2) aplica ajuste ligero por histórico resumido (rotación / ya lo hacen / etc.)
    3) pasa por scoring real del ranking
    4) selecciona top cards para Owner con diversidad pragmática

    Notas:
    - client_id y period_key se mantienen por compatibilidad de interfaz,
      aunque el store ya suele venir instanciado por cliente.
    - la capa de histórico existe dos veces a propósito:
      a) apply_history_penalty() = rotación ligera
      b) score_recommendations() = ajuste más serio por histórico real
    """
    _ = client_id, period_key  # compatibilidad explícita; no se usan de forma directa aquí

    history_summary = _safe_history_summary(history)

    base_cards = build_recommendation_cards(
        ctx=ctx,
        business_type=business_type,
        subtype=subtype,
    )

    cards_after_history_penalty = apply_history_penalty(
        base_cards,
        history_summary=history_summary,
    )

    scored_cards = score_recommendations(
        cards_after_history_penalty,
        history_store=history,
    )

    top_scored = select_plan(
        scored_cards,
        k=top_k,
    )

    return RecommendationPlan(
        top_cards=_to_results(top_scored),
        all_cards=_to_results(scored_cards),
    )


# ============================================================
# Compatibilidad temporal con API antigua
# ============================================================
@dataclass(frozen=True)
class RankedInsight:
    """
    Compatibilidad temporal con la interfaz antigua.
    Ya no representa un Insight clásico, sino una recomendación operativa.
    """

    insight: RecommendationCard
    score: float
    reason: str


def rank_insights(
    ctx: dict,
    business_type: str,
    subtype: str,
    history: Optional[HistoryStore],
    *,
    client_id: Optional[str] = None,
    period_key: Optional[str] = None,
) -> List[RankedInsight]:
    """
    Compat temporal.
    Devuelve la lista completa ya rankeada.
    """
    plan = build_recommendation_plan(
        ctx=ctx,
        business_type=business_type,
        subtype=subtype,
        history=history,
        client_id=client_id,
        period_key=period_key,
        top_k=999,
    )

    return [
        RankedInsight(
            insight=r.card,
            score=r.score,
            reason=r.reason,
        )
        for r in plan.all_cards
    ]


def pick_plan(ranked: List[RankedInsight], k: int = 3) -> List[RankedInsight]:
    """
    Compat temporal.
    Selecciona las primeras k por score.
    """
    ranked_sorted = sorted(ranked, key=lambda x: x.score, reverse=True)
    return ranked_sorted[:k]