from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

from app.analytics.history_store import HistoryStore


# =============================================================================
# OBJETIVO DEL MÓDULO
# - Convertir el histórico en una narrativa de mejora continua.
# - Resumir aprendizajes realmente útiles.
# - Sugerir el siguiente foco a partir de resultados previos.
# - Reforzar el valor del mantenimiento mensual:
#     * no solo mostramos datos,
#     * acumulamos aprendizaje,
#     * priorizamos mejor con el tiempo.
#
# IDEA CLAVE
# Este módulo no reemplaza history_store.py.
# Usa history_store.py como fuente y lo traduce a lenguaje de producto:
# - qué hemos probado,
# - qué funcionó,
# - qué no funcionó,
# - qué conviene hacer ahora.
# =============================================================================


# =============================================================================
# Modelos
# =============================================================================
@dataclass(frozen=True)
class LearningInsight:
    """Aprendizaje útil extraído del histórico."""

    insight_id: str
    category: str  # win | loss | pending | inconclusive | habit
    title: str
    summary: str
    evidence: str
    recommended_next_step: str
    confidence_label: str
    metric_label: str | None = None
    outcome: str | None = None
    period: str | None = None


@dataclass(frozen=True)
class LearningSummary:
    """
    Resumen global de aprendizaje del negocio.

    commercial_value_message:
        Texto que ayuda a defender el valor del seguimiento mensual.
    """

    maturity_label: str
    learning_score: int
    commercial_value_message: str
    owner_summary: str
    admin_summary: str
    insights: list[LearningInsight]
    stats: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["insights"] = [asdict(x) for x in self.insights]
        return out


# =============================================================================
# Helpers internos
# =============================================================================
def _safe_str(x: Any) -> str:
    return "" if x is None else str(x).strip()



def _outcome_label(outcome: str) -> str:
    mapping = {
        "improved": "mejoró",
        "not_improved": "no mejoró",
        "inconclusive": "quedó inconcluso",
        "neutral": "fue neutro",
        "unknown": "sin resultado claro",
    }
    return mapping.get(str(outcome or "unknown").strip().lower(), "sin resultado claro")



def _confidence_from_outcome(outcome: str, has_metric_delta: bool, has_note: bool) -> str:
    outcome = str(outcome or "unknown").strip().lower()
    if outcome == "improved" and (has_metric_delta or has_note):
        return "alta"
    if outcome in {"not_improved", "inconclusive", "neutral"} and (has_metric_delta or has_note):
        return "media"
    if has_note:
        return "media"
    return "baja"



def _build_evidence(row: dict) -> str:
    parts: list[str] = []

    metric_label = _safe_str(row.get("primary_metric_label"))
    baseline = row.get("baseline_value")
    observed = row.get("observed_value")
    delta_abs = row.get("delta_absolute")
    delta_rel = row.get("delta_relative")

    if metric_label:
        parts.append(f"Métrica principal: {metric_label}")

    if baseline is not None and observed is not None:
        parts.append(f"Base {baseline} → observado {observed}")

    if delta_abs is not None:
        parts.append(f"Δ abs: {delta_abs}")

    if delta_rel is not None:
        parts.append(f"Δ rel: {delta_rel}")

    note = _safe_str(row.get("learning_note") or row.get("note"))
    if note:
        parts.append(f"Nota: {note}")

    return " | ".join(parts) if parts else "Sin evidencia detallada guardada."



def _next_step_from_outcome(row: dict) -> str:
    outcome = _safe_str(row.get("outcome")).lower()
    note = _safe_str(row.get("learning_note") or row.get("note"))

    if outcome == "improved":
        if note:
            return "Repetir la acción donde funcionó mejor y estandarizarla si mantiene consistencia."
        return "Escalar la acción en una franja, producto o contexto parecido antes de ampliarla más."

    if outcome == "not_improved":
        return "No insistir igual. Cambiar el enfoque, la oferta o el contexto antes de volver a probarla."

    if outcome == "inconclusive":
        return "Repetir con una ejecución más controlada o con una medición más clara para sacar una conclusión útil."

    if outcome == "neutral":
        return "Dejarla en segundo plano y priorizar acciones con más potencial o mejor señal histórica."

    status = _safe_str(row.get("status")).lower()
    if status in {"active", "planned", "already_doing"}:
        return "Mantener seguimiento y revisar el resultado antes de cambiar varias cosas a la vez."

    return "Seguir acumulando aprendizaje antes de tomar una decisión más fuerte."



def _title_from_row(row: dict) -> str:
    insight_id = _safe_str(row.get("insight_id")) or "accion_sin_nombre"
    outcome = _safe_str(row.get("outcome")).lower()
    if outcome == "improved":
        return f"La acción {insight_id} mostró señal positiva"
    if outcome == "not_improved":
        return f"La acción {insight_id} no funcionó como se esperaba"
    if outcome == "inconclusive":
        return f"La acción {insight_id} sigue sin una conclusión clara"
    if outcome == "neutral":
        return f"La acción {insight_id} tuvo impacto neutro"
    return f"La acción {insight_id} sigue en seguimiento"



def _category_from_row(row: dict) -> str:
    outcome = _safe_str(row.get("outcome")).lower()
    status = _safe_str(row.get("status")).lower()

    if outcome == "improved":
        return "win"
    if outcome == "not_improved":
        return "loss"
    if outcome == "inconclusive":
        return "inconclusive"
    if status == "already_doing":
        return "habit"
    if status in {"active", "planned"}:
        return "pending"
    return "pending"


# =============================================================================
# Construcción principal
# =============================================================================
def build_learning_summary(
    history_store: HistoryStore,
    *,
    recent_limit: int = 8,
    next_focus_limit: int = 3,
) -> LearningSummary:
    """
    Construye una lectura de aprendizaje continuo a partir del histórico.
    """
    learnings = history_store.get_recent_learnings(limit=max(3, int(recent_limit)))
    next_focus = history_store.get_next_focus_items(limit=max(1, int(next_focus_limit)))
    hist_summary = history_store.get_history_summary()

    insights: list[LearningInsight] = []

    # -------------------------------------------------------------------------
    # Insights de aprendizaje reciente
    # -------------------------------------------------------------------------
    for row in learnings:
        outcome = _safe_str(row.get("outcome")).lower() or "unknown"
        note = _safe_str(row.get("learning_note") or row.get("note"))
        metric_label = _safe_str(row.get("primary_metric_label")) or None
        has_metric_delta = row.get("delta_absolute") is not None or row.get("delta_relative") is not None
        confidence = _confidence_from_outcome(outcome, has_metric_delta=has_metric_delta, has_note=bool(note))

        summary = note or f"La acción {_safe_str(row.get('insight_id'))} {_outcome_label(outcome)}."

        insights.append(
            LearningInsight(
                insight_id=_safe_str(row.get("insight_id")),
                category=_category_from_row(row),
                title=_title_from_row(row),
                summary=summary,
                evidence=_build_evidence(row),
                recommended_next_step=_next_step_from_outcome(row),
                confidence_label=confidence,
                metric_label=metric_label,
                outcome=outcome,
                period=_safe_str(row.get("period")) or None,
            )
        )

    # -------------------------------------------------------------------------
    # Insights de siguiente foco si aún no están representados
    # -------------------------------------------------------------------------
    existing_ids = {x.insight_id for x in insights}
    for row in next_focus:
        iid = _safe_str(row.get("insight_id"))
        if not iid or iid in existing_ids:
            continue

        experiment = row.get("experiment") or {}
        where_to_apply = _safe_str(experiment.get("where_to_apply"))
        metric_label = _safe_str(row.get("primary_metric_label")) or None

        summary = "Acción pendiente o en seguimiento que todavía no tiene resultado consolidado."
        if where_to_apply:
            summary += f" Próximo contexto sugerido: {where_to_apply}."

        insights.append(
            LearningInsight(
                insight_id=iid,
                category="pending",
                title=f"Siguiente foco sugerido: {iid}",
                summary=summary,
                evidence="Pendiente de revisión o todavía sin resultado final guardado.",
                recommended_next_step="Mantener la prueba viva hasta revisar el resultado y registrar qué pasó.",
                confidence_label="media",
                metric_label=metric_label,
                outcome=_safe_str(row.get("status")) or None,
                period=None,
            )
        )

    # -------------------------------------------------------------------------
    # Score y madurez
    # -------------------------------------------------------------------------
    counts = hist_summary.get("counts", {}) or {}
    reviewed_items = int(counts.get("reviewed_items", 0) or 0)
    improved_items = int(counts.get("improved_items", 0) or 0)
    not_improved_items = int(counts.get("not_improved_items", 0) or 0)
    inconclusive_items = int(counts.get("inconclusive_items", 0) or 0)
    total_items = int(counts.get("items", 0) or 0)

    score = 25
    score += min(30, reviewed_items * 4)
    score += min(20, improved_items * 5)
    score += min(10, not_improved_items * 2)  # también aprender qué no funciona tiene valor
    score += min(8, inconclusive_items * 1)
    if total_items > 0 and reviewed_items == 0:
        score -= 8
    score = max(0, min(100, score))

    if score >= 80:
        maturity_label = "alta"
    elif score >= 55:
        maturity_label = "media"
    elif score >= 30:
        maturity_label = "inicial"
    else:
        maturity_label = "muy inicial"

    # -------------------------------------------------------------------------
    # Mensajes de valor comercial
    # -------------------------------------------------------------------------
    if reviewed_items == 0:
        commercial_value_message = (
            "El valor del seguimiento mensual aquí está en empezar a convertir acciones sueltas en aprendizaje acumulado y decisiones mejores."
        )
        owner_summary = (
            "Todavía no hay suficiente aprendizaje revisado. El siguiente paso es probar menos cosas a la vez y registrar mejor qué pasó."
        )
    elif improved_items > 0:
        commercial_value_message = (
            "El seguimiento mensual ya aporta valor porque no solo enseña datos: conserva qué acciones funcionaron y ayuda a repetirlas con más criterio."
        )
        owner_summary = (
            "Ya hay señales útiles de aprendizaje. El sistema puede ayudarte a repetir lo que funcionó mejor y evitar volver a insistir en lo que no dio resultado."
        )
    else:
        commercial_value_message = (
            "El seguimiento mensual aporta valor aunque no todo mejore, porque evita repetir acciones pobres y acelera el aprendizaje útil del negocio."
        )
        owner_summary = (
            "Ya hay suficiente histórico para distinguir mejor entre ideas prometedoras, ideas flojas y acciones que aún necesitan una prueba más limpia."
        )

    admin_summary = (
        f"Madurez de aprendizaje {maturity_label}. "
        f"Items revisados={reviewed_items}, improved={improved_items}, "
        f"not_improved={not_improved_items}, inconclusive={inconclusive_items}, total_items={total_items}."
    )

    stats = {
        "reviewed_items": reviewed_items,
        "improved_items": improved_items,
        "not_improved_items": not_improved_items,
        "inconclusive_items": inconclusive_items,
        "total_items": total_items,
        "next_focus_count": len(next_focus),
        "insights_generated": len(insights),
    }

    return LearningSummary(
        maturity_label=maturity_label,
        learning_score=int(score),
        commercial_value_message=commercial_value_message,
        owner_summary=owner_summary,
        admin_summary=admin_summary,
        insights=insights,
        stats=stats,
    )


# =============================================================================
# Helpers de UI
# =============================================================================
def learning_owner_caption(summary: LearningSummary) -> str:
    return (
        f"Aprendizaje acumulado: {summary.maturity_label}. "
        f"{summary.owner_summary} "
        f"Madurez estimada: {summary.learning_score}/100."
    )



def learning_admin_lines(summary: LearningSummary) -> list[str]:
    lines = [summary.admin_summary, summary.commercial_value_message]
    for ins in summary.insights:
        lines.append(
            f"- {ins.category.upper()}: {ins.title} -> {ins.summary} | siguiente: {ins.recommended_next_step}"
        )
    return lines



def pick_next_best_learning_action(summary: LearningSummary) -> LearningInsight | None:
    """Devuelve el mejor siguiente foco según el resumen generado."""
    if not summary.insights:
        return None

    # Priorizamos pending, luego wins que convenga escalar, luego inconclusive.
    category_order = {"pending": 0, "win": 1, "inconclusive": 2, "habit": 3, "loss": 4}
    ordered = sorted(
        summary.insights,
        key=lambda x: (category_order.get(x.category, 9), -len(x.summary or "")),
    )
    return ordered[0] if ordered else None