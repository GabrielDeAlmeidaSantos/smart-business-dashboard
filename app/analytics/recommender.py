from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from .experiments import (
    build_experiment_plan,
    build_metric_target,
    experiment_to_history_payload,
    ExperimentPlan,
)
from .history_store import HistoryStore
from .ranking import ScoredRecommendation, score_recommendations, select_plan
from .recommendation_library import (
    RecommendationCard,
    apply_history_penalty,
    build_recommendation_cards,
)


# =============================================================================
# OBJETIVO DE ESTA VERSIÓN
# - Ensamblar mejor profile + library + ranking + history.
# - Hacer que el top final sea más coherente con el vertical detectado.
# - Traducir recomendaciones a planes/experimentos más estandarizados.
# - Preparar mejor el output para Owner, Admin, learning e histórico.
#
# PROBLEMAS DE LA VERSIÓN ORIGINAL
# 1) Era correcta pero demasiado fina: hacía de pegamento mínimo, no de motor real.
# 2) No exponía señales útiles del plan (cobertura vertical, calidad del top, etc.).
# 3) No aprovechaba experiments.py.
# 4) No preparaba payloads ricos para history_store.
# 5) La compatibilidad con business_type/subtype unknown era demasiado pasiva.
# =============================================================================


# ============================================================
# Modelos principales del motor de recomendaciones
# ============================================================
@dataclass(frozen=True)
class RecommendationResult:
    """Resultado final listo para UI Owner/Admin."""

    card: RecommendationCard
    score: float
    reason: str
    experiment: ExperimentPlan


@dataclass(frozen=True)
class RecommendationPlan:
    """
    Plan final preparado para la UI y para histórico.

    Además del top y del ranking completo, ahora incluye metadata útil para:
    - onboarding / debug
    - UI owner/admin
    - persistencia de planes con más contexto
    """

    top_cards: List[RecommendationResult]
    all_cards: List[RecommendationResult]
    meta: dict

    def top_history_payloads(self) -> list[dict]:
        """Payloads listos para history_store a partir del top actual."""
        return [experiment_to_history_payload(x.experiment) for x in self.top_cards]


# ============================================================
# Helpers internos
# ============================================================
def _safe_history_summary(history: Optional[HistoryStore]) -> dict:
    """
    Obtiene un resumen compacto del histórico sin romper el flujo.
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



def _safe_learning_signal(history: Optional[HistoryStore]) -> dict:
    """
    Señal opcional de aprendizaje. No rompe si aún no existe todo.
    """
    if history is None:
        return {}

    out: dict = {}

    method_story = getattr(history, "get_action_story", None)
    if callable(method_story):
        try:
            out["recent_story"] = method_story(limit=5) or []
        except Exception:
            out["recent_story"] = []

    method_focus = getattr(history, "get_next_focus_items", None)
    if callable(method_focus):
        try:
            out["next_focus"] = method_focus(limit=3) or []
        except Exception:
            out["next_focus"] = []

    return out



def _vertical_strength(business_type: str, subtype: str) -> str:
    """
    Señal simple para metadata del plan.
    No pretende ser probabilística; solo explica el contexto de selección.
    """
    if business_type == "unknown":
        return "débil"
    if subtype == "unknown":
        return "media"
    return "alta"



def _primary_metric_direction_from_card(card: RecommendationCard) -> str:
    """
    Heurística simple para direction del experimento.
    Casi todo en este producto quiere mejorar algo o mantener que no empeore.
    """
    goal = str(card.goal or "").strip().lower()
    metric = str(card.primary_metric_key or "").strip().lower()

    if goal in {"riesgo_mix", "margen"} and "share" in metric:
        return "down"
    if "share" in metric and goal in {"riesgo_mix"}:
        return "down"
    return "up"



def _build_experiment_from_card(card: RecommendationCard) -> ExperimentPlan:
    """
    Convierte una RecommendationCard en un experimento estandarizado.
    Aquí está una de las mejoras clave: el motor deja de devolver solo consejos.
    """
    primary_metric = build_metric_target(
        key=card.primary_metric_key,
        label=card.primary_metric_label,
        direction=_primary_metric_direction_from_card(card),
        notes="Métrica principal sugerida por el motor actual.",
    )

    secondary_metrics = [
        build_metric_target(
            key=k,
            label=l,
            direction="up",
            notes="Métrica secundaria de apoyo.",
        )
        for k, l in zip(card.secondary_metric_keys, card.secondary_metric_labels)
    ]

    implementation_steps = (
        f"Aplicar la prueba en {card.where_to_apply}.",
        "Mantener una ejecución consistente durante toda la ventana de prueba.",
        f"Revisar principalmente {card.primary_metric_label.lower()} antes de cambiar otra cosa.",
    )

    guardrail = "No empeorar claramente la operativa ni la conversión general mientras se prueba."

    return build_experiment_plan(
        recommendation_id=card.insight_id,
        title=card.title,
        hypothesis=card.hypothesis,
        action=card.action,
        why_now=card.why_now,
        where_to_apply=card.where_to_apply,
        duration_days=card.duration_days,
        review_window_label=card.review_window_label,
        primary_metric=primary_metric,
        secondary_metrics=secondary_metrics,
        success_rule_text=card.success_rule_text,
        guardrail_rule_text=guardrail,
        implementation_steps=implementation_steps,
        confidence_label=card.confidence_label,
        priority_label=card.priority_label,
        effort=card.effort,
        estimated_minutes_to_apply=card.time_to_apply_min,
        owner_goal=card.goal,
        tags=card.tags,
    )



def _to_results(scored: List[ScoredRecommendation]) -> List[RecommendationResult]:
    """Convierte ScoredRecommendation -> RecommendationResult con experimento."""
    results: list[RecommendationResult] = []
    for s in scored:
        results.append(
            RecommendationResult(
                card=s.card,
                score=s.score,
                reason=s.reason,
                experiment=_build_experiment_from_card(s.card),
            )
        )
    return results



def _select_top_from_results(all_results: List[RecommendationResult], k: int) -> List[RecommendationResult]:
    """
    select_plan opera con ScoredRecommendation. Esta función usa el ranking ya decidido
    indirectamente en build_recommendation_plan y mantiene el orden resultante.
    """
    return all_results[: max(0, int(k))]


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
    2) aplica ajuste ligero por histórico resumido
    3) pasa por scoring real del ranking
    4) selecciona top cards con diversidad pragmática
    5) traduce el resultado a experimentos estandarizados
    6) expone metadata útil del plan
    """
    history_summary = _safe_history_summary(history)
    learning_signal = _safe_learning_signal(history)

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

    # Convertimos ambas vistas a resultados ricos.
    all_results = _to_results(scored_cards)

    # Para respetar exactamente el top decidido por select_plan, reconstruimos por insight_id.
    top_ids = [x.card.insight_id for x in top_scored]
    top_results = [r for r in all_results if r.card.insight_id in top_ids]
    top_results.sort(key=lambda r: top_ids.index(r.card.insight_id))

    meta = {
        "client_id": client_id,
        "period_key": period_key,
        "business_type": business_type,
        "subtype": subtype,
        "vertical_strength": _vertical_strength(business_type, subtype),
        "cards_base_count": len(base_cards),
        "cards_after_history_penalty_count": len(cards_after_history_penalty),
        "cards_scored_count": len(scored_cards),
        "top_k_requested": int(top_k),
        "top_k_returned": len(top_results),
        "history_summary": history_summary,
        "learning_signal": learning_signal,
        "ctx_keys": sorted(list(ctx.keys())),
        "top_ids": top_ids,
    }

    return RecommendationPlan(
        top_cards=top_results,
        all_cards=all_results,
        meta=meta,
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

    Se mantiene simple solo por compatibilidad.
    Para flujo moderno, usar build_recommendation_plan().
    """
    ranked_sorted = sorted(ranked, key=lambda x: x.score, reverse=True)
    return ranked_sorted[:k]