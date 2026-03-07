from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional

from .history_store import HistoryStore
from .recommendation_library import (
    RecommendationCard,
    apply_history_penalty,
    build_recommendation_cards,
    select_top_cards,
)


# ============================================================
# Modelo principal del motor de recomendaciones
# ============================================================
@dataclass(frozen=True)
class RecommendationResult:
    """
    Resultado final del motor para una recomendación concreta.
    Ya viene listo para:
    - Owner
    - Admin
    - seguimiento posterior
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
def _safe_history_summary(
    history: Optional[HistoryStore],
    *,
    client_id: Optional[str] = None,
    period_key: Optional[str] = None,
) -> dict:
    """
    Intenta obtener un resumen de histórico sin romper si la API real
    del HistoryStore aún no está completamente cerrada.

    Esto permite avanzar el MVP sin bloquearte por la capa de persistencia.
    """
    if history is None:
        return {}

    # Caso ideal: el store ya expone un resumen útil
    for method_name in (
        "get_history_summary",
        "build_history_summary",
        "summary_for_period",
        "get_period_summary",
    ):
        method = getattr(history, method_name, None)
        if callable(method):
            try:
                if client_id is not None and period_key is not None:
                    return method(client_id=client_id, period_key=period_key) or {}
                if period_key is not None:
                    return method(period_key=period_key) or {}
                return method() or {}
            except TypeError:
                try:
                    return method(period_key=period_key) or {}
                except Exception:
                    pass
            except Exception:
                pass

    return {}


def _score_reason(card: RecommendationCard, history_summary: dict | None = None) -> str:
    """
    Explicación corta y comercial del porqué queda priorizada.
    """
    history_summary = history_summary or {}

    reason_parts: list[str] = []

    if card.priority_label == "alta":
        reason_parts.append("prioridad alta por contexto actual")
    elif card.priority_label == "media":
        reason_parts.append("prioridad media por potencial y facilidad")
    else:
        reason_parts.append("prioridad baja pero útil como prueba complementaria")

    if card.confidence_label:
        reason_parts.append(f"confianza {card.confidence_label}")

    if card.effort == "bajo":
        reason_parts.append("esfuerzo bajo")

    if card.insight_id in set(history_summary.get("recent_positive", [])):
        reason_parts.append("alineada con aprendizaje positivo reciente")

    if card.insight_id in set(history_summary.get("already_doing", [])):
        reason_parts.append("ya existe contexto previo en el negocio")

    return " · ".join(reason_parts)


def _materialize_results(cards: Iterable[RecommendationCard], history_summary: dict | None = None) -> list[RecommendationResult]:
    """
    Convierte cards en resultados finales con score explícito.
    """
    results: list[RecommendationResult] = []

    for card in cards:
        results.append(
            RecommendationResult(
                card=card,
                score=card.priority_weight,
                reason=_score_reason(card, history_summary=history_summary),
            )
        )

    return sorted(results, key=lambda x: x.score, reverse=True)


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
    2) aplica histórico
    3) ordena
    4) selecciona top cards para Owner
    """
    history_summary = _safe_history_summary(
        history,
        client_id=client_id,
        period_key=period_key,
    )

    base_cards = build_recommendation_cards(
        ctx=ctx,
        business_type=business_type,
        subtype=subtype,
    )

    rescored_cards = apply_history_penalty(
        base_cards,
        history_summary=history_summary,
    )

    all_results = _materialize_results(
        rescored_cards,
        history_summary=history_summary,
    )

    top_cards_raw = select_top_cards(
        [r.card for r in all_results],
        k=top_k,
    )

    top_ids = {c.insight_id for c in top_cards_raw}
    top_results = [r for r in all_results if r.card.insight_id in top_ids]

    # Reordenar top por score descendente
    top_results = sorted(top_results, key=lambda x: x.score, reverse=True)

    return RecommendationPlan(
        top_cards=top_results,
        all_cards=all_results,
    )


# ============================================================
# Compatibilidad temporal con API antigua
# ============================================================
@dataclass(frozen=True)
class RankedInsight:
    """
    Compatibilidad temporal con la interfaz antigua.
    Ojo: ya no representa un Insight clásico, sino una recomendación operativa.
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