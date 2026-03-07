from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Iterable, Optional


# =============================================================================
# OBJETIVO DEL MÓDULO
# - Estandarizar las pruebas recomendadas.
# - Dar una estructura común a hipótesis, métrica principal, ventana de revisión
#   y criterio de éxito.
# - Hacer que el producto se perciba más como copiloto de decisiones y menos como
#   una lista de consejos sueltos.
#
# CRÍTICA A LA VERSIÓN ORIGINAL
# - La base era correcta, pero demasiado mínima.
# - Faltaban campos para:
#     * contexto de ejecución,
#     * esfuerzo,
#     * prioridad,
#     * checklist de implementación,
#     * lectura de resultados,
#     * siguiente decisión.
# - No había helpers para serializar, resumir o construir planes de forma consistente.
# =============================================================================


# =============================================================================
# Constantes semánticas
# =============================================================================
DIRECTION_UP = "up"
DIRECTION_DOWN = "down"
DIRECTION_STABLE = "stable"

CONFIDENCE_LOW = "baja"
CONFIDENCE_MEDIUM = "media"
CONFIDENCE_HIGH = "alta"

DECISION_SCALE = "scale"
DECISION_ITERATE = "iterate"
DECISION_STOP = "stop"
DECISION_REVIEW = "review"

VALID_DIRECTIONS = {DIRECTION_UP, DIRECTION_DOWN, DIRECTION_STABLE}
VALID_CONFIDENCE = {CONFIDENCE_LOW, CONFIDENCE_MEDIUM, CONFIDENCE_HIGH}
VALID_DECISIONS = {DECISION_SCALE, DECISION_ITERATE, DECISION_STOP, DECISION_REVIEW}


# =============================================================================
# Modelos base
# =============================================================================
@dataclass(frozen=True)
class MetricTarget:
    """
    Define qué se medirá para evaluar una acción.

    direction:
        - up: queremos que suba
        - down: queremos que baje
        - stable: queremos que no empeore claramente / que se mantenga estable
    """

    key: str
    label: str
    direction: str
    min_relative_change: Optional[float] = None
    min_absolute_change: Optional[float] = None
    notes: str = ""

    def __post_init__(self) -> None:
        if self.direction not in VALID_DIRECTIONS:
            raise ValueError(f"direction inválida: {self.direction}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ExperimentContext:
    """
    Contexto operativo de una prueba.

    Sirve para que el experimento sea más ejecutable y menos abstracto.
    """

    where_to_apply: str
    duration_days: int
    review_window_label: str
    effort: str = "medio"
    estimated_minutes_to_apply: int = 15
    owner_goal: str = ""
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SuccessCriteria:
    """
    Criterios de éxito y guardrails.

    primary_rule_text:
        Qué tendría que pasar para considerar que la prueba funcionó.

    guardrail_rule_text:
        Qué no debería empeorar claramente mientras probamos.

    next_decision_if_success / mixed / fail:
        Qué decisión tomar después.
    """

    primary_rule_text: str
    guardrail_rule_text: str = ""
    next_decision_if_success: str = DECISION_SCALE
    next_decision_if_mixed: str = DECISION_ITERATE
    next_decision_if_fail: str = DECISION_STOP

    def __post_init__(self) -> None:
        for value in (
            self.next_decision_if_success,
            self.next_decision_if_mixed,
            self.next_decision_if_fail,
        ):
            if value not in VALID_DECISIONS:
                raise ValueError(f"decision inválida: {value}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ExperimentPlan:
    """
    Unidad operativa estandarizada de una recomendación.

    Esto ya no es solo una recomendación bonita. Es una prueba ejecutable,
    medible y revisable.
    """

    experiment_id: str
    recommendation_id: str
    title: str
    hypothesis: str
    action: str
    why_now: str
    context: ExperimentContext
    primary_metric: MetricTarget
    secondary_metrics: tuple[MetricTarget, ...] = field(default_factory=tuple)
    success_criteria: SuccessCriteria = field(
        default_factory=lambda: SuccessCriteria(primary_rule_text="")
    )
    implementation_steps: tuple[str, ...] = field(default_factory=tuple)
    confidence_label: str = CONFIDENCE_MEDIUM
    priority_label: str = "media"
    tags: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.confidence_label not in VALID_CONFIDENCE:
            raise ValueError(f"confidence_label inválida: {self.confidence_label}")

    @property
    def success_rule_text(self) -> str:
        return self.success_criteria.primary_rule_text

    @property
    def where_to_apply(self) -> str:
        return self.context.where_to_apply

    @property
    def duration_days(self) -> int:
        return self.context.duration_days

    @property
    def review_window_label(self) -> str:
        return self.context.review_window_label

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "recommendation_id": self.recommendation_id,
            "title": self.title,
            "hypothesis": self.hypothesis,
            "action": self.action,
            "why_now": self.why_now,
            "context": self.context.to_dict(),
            "primary_metric": self.primary_metric.to_dict(),
            "secondary_metrics": [m.to_dict() for m in self.secondary_metrics],
            "success_criteria": self.success_criteria.to_dict(),
            "implementation_steps": list(self.implementation_steps),
            "confidence_label": self.confidence_label,
            "priority_label": self.priority_label,
            "tags": list(self.tags),
        }


# =============================================================================
# Helpers de construcción
# =============================================================================
def build_metric_target(
    key: str,
    label: str,
    direction: str,
    *,
    min_relative_change: Optional[float] = None,
    min_absolute_change: Optional[float] = None,
    notes: str = "",
) -> MetricTarget:
    return MetricTarget(
        key=key,
        label=label,
        direction=direction,
        min_relative_change=min_relative_change,
        min_absolute_change=min_absolute_change,
        notes=notes,
    )



def build_experiment_plan(
    *,
    recommendation_id: str,
    title: str,
    hypothesis: str,
    action: str,
    why_now: str,
    where_to_apply: str,
    duration_days: int,
    review_window_label: str,
    primary_metric: MetricTarget,
    secondary_metrics: Optional[Iterable[MetricTarget]] = None,
    success_rule_text: str,
    guardrail_rule_text: str = "",
    implementation_steps: Optional[Iterable[str]] = None,
    confidence_label: str = CONFIDENCE_MEDIUM,
    priority_label: str = "media",
    effort: str = "medio",
    estimated_minutes_to_apply: int = 15,
    owner_goal: str = "",
    tags: Optional[Iterable[str]] = None,
    experiment_id: Optional[str] = None,
) -> ExperimentPlan:
    """
    Constructor estándar para generar planes homogéneos desde recommender/library.
    """
    exp_id = experiment_id or f"exp::{recommendation_id}"

    context = ExperimentContext(
        where_to_apply=where_to_apply,
        duration_days=int(duration_days),
        review_window_label=review_window_label,
        effort=effort,
        estimated_minutes_to_apply=int(estimated_minutes_to_apply),
        owner_goal=owner_goal,
    )

    criteria = SuccessCriteria(
        primary_rule_text=success_rule_text,
        guardrail_rule_text=guardrail_rule_text,
        next_decision_if_success=DECISION_SCALE,
        next_decision_if_mixed=DECISION_ITERATE,
        next_decision_if_fail=DECISION_STOP,
    )

    return ExperimentPlan(
        experiment_id=exp_id,
        recommendation_id=recommendation_id,
        title=title,
        hypothesis=hypothesis,
        action=action,
        why_now=why_now,
        context=context,
        primary_metric=primary_metric,
        secondary_metrics=tuple(secondary_metrics or ()),
        success_criteria=criteria,
        implementation_steps=tuple(x for x in (implementation_steps or ()) if str(x).strip()),
        confidence_label=confidence_label,
        priority_label=priority_label,
        tags=tuple(tags or ()),
    )


# =============================================================================
# Helpers de interpretación
# =============================================================================
def summarize_experiment_for_owner(plan: ExperimentPlan) -> str:
    """
    Resumen corto y entendible para Owner.
    """
    return (
        f"Prueba durante {plan.duration_days} días en {plan.where_to_apply}. "
        f"Queremos mover {plan.primary_metric.label.lower()} y considerar éxito si {plan.success_rule_text.lower()}"
    )



def summarize_experiment_for_admin(plan: ExperimentPlan) -> list[str]:
    """
    Resumen más estructurado para Admin/debug.
    """
    lines = [
        f"Experimento: {plan.title}",
        f"Reco: {plan.recommendation_id}",
        f"Hipótesis: {plan.hypothesis}",
        f"Acción: {plan.action}",
        f"Por qué ahora: {plan.why_now}",
        f"Dónde aplicar: {plan.where_to_apply}",
        f"Duración: {plan.duration_days} días ({plan.review_window_label})",
        f"Métrica principal: {plan.primary_metric.label} [{plan.primary_metric.direction}]",
        f"Éxito: {plan.success_rule_text}",
    ]

    if plan.success_criteria.guardrail_rule_text:
        lines.append(f"Guardrail: {plan.success_criteria.guardrail_rule_text}")

    if plan.secondary_metrics:
        sec = ", ".join(f"{m.label}[{m.direction}]" for m in plan.secondary_metrics)
        lines.append(f"Secundarias: {sec}")

    if plan.implementation_steps:
        lines.append("Checklist:")
        lines.extend([f"- {step}" for step in plan.implementation_steps])

    return lines



def experiment_to_history_payload(plan: ExperimentPlan) -> dict[str, Any]:
    """
    Convierte un plan a payload útil para history_store.
    Esto encaja muy bien con el schema nuevo de experimentos.
    """
    return {
        "insight_id": plan.recommendation_id,
        "primary_metric_key": plan.primary_metric.key,
        "primary_metric_label": plan.primary_metric.label,
        "hypothesis": plan.hypothesis,
        "where_to_apply": plan.where_to_apply,
        "success_rule_text": plan.success_rule_text,
        "review_window_label": plan.review_window_label,
        "duration_days": plan.duration_days,
    }



def classify_result(
    *,
    delta_relative: Optional[float] = None,
    delta_absolute: Optional[float] = None,
    target: MetricTarget,
) -> str:
    """
    Clasificación simple del resultado de una prueba.

    No sustituye criterio humano, pero estandariza una primera lectura.
    Devuelve:
        improved | not_improved | inconclusive
    """
    rel_ok = None
    abs_ok = None

    if target.min_relative_change is not None and delta_relative is not None:
        if target.direction == DIRECTION_UP:
            rel_ok = float(delta_relative) >= float(target.min_relative_change)
        elif target.direction == DIRECTION_DOWN:
            rel_ok = float(delta_relative) <= -abs(float(target.min_relative_change))
        else:
            rel_ok = abs(float(delta_relative)) <= abs(float(target.min_relative_change))

    if target.min_absolute_change is not None and delta_absolute is not None:
        if target.direction == DIRECTION_UP:
            abs_ok = float(delta_absolute) >= float(target.min_absolute_change)
        elif target.direction == DIRECTION_DOWN:
            abs_ok = float(delta_absolute) <= -abs(float(target.min_absolute_change))
        else:
            abs_ok = abs(float(delta_absolute)) <= abs(float(target.min_absolute_change))

    checks = [x for x in (rel_ok, abs_ok) if x is not None]
    if not checks:
        return "inconclusive"
    if all(checks):
        return "improved"
    if any(checks):
        return "inconclusive"
    return "not_improved"