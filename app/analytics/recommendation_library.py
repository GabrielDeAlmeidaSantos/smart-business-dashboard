from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable


# ============================================================
# Tipos base
# ============================================================
TriggerFn = Callable[[dict], bool]
ImpactFn = Callable[[dict], tuple[float, float]]


@dataclass(frozen=True)
class RecommendationSpec:
    """
    Definición reusable de una recomendación.

    Esta versión ya no modela solo "copy de consejo", sino una acción
    mínimamente operativa y medible para el MVP comercial.
    """

    # Identidad
    insight_id: str
    title: str
    goal: str

    # Segmentación
    applies_to: tuple[str, ...]
    subtypes: tuple[str, ...]

    # Priorización
    priority_weight: float
    effort: str
    time_to_apply_min: int
    tags: tuple[str, ...]

    # Activación
    trigger_fn: TriggerFn
    impact_fn: ImpactFn

    # Contenido explicativo
    why_now_template: str
    action_template: str
    if_already_doing_template: str
    strategy_template: str

    # ----------------------------
    # Nuevo: capa de experimento
    # ----------------------------
    hypothesis_template: str
    where_to_apply_template: str
    duration_days: int

    primary_metric_key: str
    primary_metric_label: str

    secondary_metric_keys: tuple[str, ...]
    secondary_metric_labels: tuple[str, ...]

    success_rule_text: str
    confidence_label: str = "media"


@dataclass(frozen=True)
class RecommendationCard:
    """
    Tarjeta final lista para mostrar en el dashboard.

    Ya incluye datos suficientes para:
    - UI Owner
    - seguimiento
    - aprendizaje posterior
    """

    # Identidad
    insight_id: str
    title: str
    goal: str

    # Mensaje principal
    why_now: str
    action: str
    if_already_doing: str
    strategy: str

    # Operatividad
    hypothesis: str
    where_to_apply: str
    duration_days: int
    review_window_label: str

    primary_metric_key: str
    primary_metric_label: str
    secondary_metric_keys: tuple[str, ...]
    secondary_metric_labels: tuple[str, ...]

    success_rule_text: str
    confidence_label: str

    # Score / esfuerzo
    effort: str
    time_to_apply_min: int
    estimated_impact_eur: tuple[float, float]
    priority_weight: float
    priority_label: str

    # Meta
    tags: tuple[str, ...]


# ============================================================
# Helpers de contexto
# ============================================================
def _ctx_float(ctx: dict, key: str, default: float = 0.0) -> float:
    try:
        value = ctx.get(key, default)
        if value is None:
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def _ctx_int(ctx: dict, key: str, default: int = 0) -> int:
    try:
        value = ctx.get(key, default)
        if value is None:
            return int(default)
        return int(value)
    except Exception:
        return int(default)


def _ctx_str(ctx: dict, key: str, default: str = "") -> str:
    value = ctx.get(key, default)
    if value is None:
        return default
    return str(value)


def _safe_fmt(template: str, ctx: dict) -> str:
    safe_ctx = {
        "top_item": ctx.get("top_item", "tu producto o servicio principal"),
        "worst_day": ctx.get("worst_day", "tu día más flojo"),
        "best_day": ctx.get("best_day", "tu mejor día"),
        "ticket_label": ctx.get("ticket_label", "importe medio"),
        "worst_day_gap_eur": f"{_ctx_float(ctx, 'worst_day_gap_eur', 0.0):.0f}",
        "slider_subida_ticket_eur": f"{_ctx_float(ctx, 'slider_subida_ticket_eur', 0.0):.0f}",
        "ventas_peor_dia": _ctx_int(ctx, "ventas_peor_dia", 0),
        "delta_ing_pct": f"{_ctx_float(ctx, 'delta_ing_pct', 0.0):.1f}",
        "top_share_pct": f"{_ctx_float(ctx, 'top_share_pct', 0.0):.1f}",
        "granularity_label": ctx.get("granularity_label", "media"),
    }
    return template.format(**safe_ctx)


def _priority_label(weight: float) -> str:
    if weight >= 0.90:
        return "alta"
    if weight >= 0.78:
        return "media"
    return "baja"


def _review_window_label(duration_days: int) -> str:
    if duration_days <= 7:
        return "revisar en 1 semana"
    if duration_days <= 14:
        return "revisar en 2 semanas"
    if duration_days <= 30:
        return "revisar este mes"
    return f"revisar en {duration_days} días"


# ============================================================
# Triggers reutilizables
# ============================================================
def trig_worst_day_gap(min_gap_eur: float) -> TriggerFn:
    def _fn(ctx: dict) -> bool:
        return _ctx_float(ctx, "worst_day_gap_eur", 0.0) >= float(min_gap_eur)

    return _fn


def trig_low_ticket(min_ventas_peor: int = 5) -> TriggerFn:
    def _fn(ctx: dict) -> bool:
        return _ctx_int(ctx, "ventas_peor_dia", 0) >= int(min_ventas_peor)

    return _fn


def trig_prev_drop(min_drop_pct: float = 3.0) -> TriggerFn:
    def _fn(ctx: dict) -> bool:
        return _ctx_float(ctx, "delta_ing_pct", 0.0) <= -abs(float(min_drop_pct))

    return _fn


def trig_top_share_high(min_share_pct: float = 35.0) -> TriggerFn:
    def _fn(ctx: dict) -> bool:
        return _ctx_float(ctx, "top_share_pct", 0.0) >= float(min_share_pct)

    return _fn


def trig_always(_: dict) -> bool:
    return True


# ============================================================
# Impactos reutilizables
# ============================================================
def impact_from_ticket(low_mult: float, high_mult: float) -> ImpactFn:
    """
    Estimación simple basada en:
    ventas_peor_dia * mejora_ticket * multiplicador.
    """

    def _fn(ctx: dict) -> tuple[float, float]:
        base = _ctx_float(ctx, "ventas_peor_dia", 0) * _ctx_float(
            ctx, "slider_subida_ticket_eur", 0
        )
        return (base * float(low_mult), base * float(high_mult))

    return _fn


def impact_from_gap(low_mult: float, high_mult: float) -> ImpactFn:
    def _fn(ctx: dict) -> tuple[float, float]:
        base = _ctx_float(ctx, "worst_day_gap_eur", 0.0)
        return (base * float(low_mult), base * float(high_mult))

    return _fn


def impact_flat(low: float, high: float) -> ImpactFn:
    def _fn(_: dict) -> tuple[float, float]:
        return (float(low), float(high))

    return _fn


# ============================================================
# Catálogo MVP v3
# ============================================================
RECOMMENDATION_LIBRARY: tuple[RecommendationSpec, ...] = (
    # --------------------------------------------------------
    # Universales
    # --------------------------------------------------------
    RecommendationSpec(
        insight_id="pack_simple",
        title="Sube el importe medio con un pack simple",
        goal="ticket_medio",
        applies_to=("servicios", "retail", "unknown"),
        subtypes=("peluqueria_estetica", "ferreteria", "tienda_general", "unknown"),
        priority_weight=0.92,
        effort="bajo",
        time_to_apply_min=20,
        tags=("ticket", "pack", "valor"),
        trigger_fn=trig_low_ticket(min_ventas_peor=4),
        impact_fn=impact_from_ticket(0.8, 1.6),
        why_now_template=(
            "Ahora mismo hay margen para subir el {ticket_label} y tu día más flojo es {worst_day}."
        ),
        action_template=(
            "Crea un pack simple con tu base fuerte ({top_item}) y un complemento lógico, y pruébalo primero en {worst_day}."
        ),
        if_already_doing_template=(
            "Si ya usas packs, reduce opciones y deja solo 1 o 2 visibles para medir mejor cuál convierte más."
        ),
        strategy_template=(
            "La idea no es descontar, sino hacer más fácil una compra de mayor valor con una propuesta clara y rápida de entender."
        ),
        hypothesis_template=(
            "Si ofreces una opción cerrada de mayor valor en {worst_day}, el importe medio debería subir sin empeorar claramente el volumen."
        ),
        where_to_apply_template="{worst_day}",
        duration_days=14,
        primary_metric_key="avg_ticket_worst_day",
        primary_metric_label="Importe medio en el día flojo",
        secondary_metric_keys=("ops_worst_day", "revenue_worst_day"),
        secondary_metric_labels=("Operaciones en el día flojo", "Ingresos en el día flojo"),
        success_rule_text="Éxito si sube el importe medio del día flojo sin caída clara en operaciones.",
        confidence_label="media",
    ),
    RecommendationSpec(
        insight_id="smart_extra",
        title="Recomienda un extra pequeño al cerrar la venta",
        goal="ticket_medio",
        applies_to=("servicios", "retail", "unknown"),
        subtypes=("peluqueria_estetica", "ferreteria", "tienda_general", "unknown"),
        priority_weight=0.90,
        effort="bajo",
        time_to_apply_min=10,
        tags=("ticket", "extra", "cierre"),
        trigger_fn=trig_low_ticket(min_ventas_peor=4),
        impact_fn=impact_from_ticket(0.6, 1.2),
        why_now_template=(
            "El volumen actual permite probar una mejora simple del cierre sin tocar precios."
        ),
        action_template=(
            "Al terminar la venta o servicio, sugiere un complemento pequeño y lógico en vez de hacer una pregunta abierta."
        ),
        if_already_doing_template=(
            "Si ya lo haces, cambia la frase por una recomendación más concreta: propone solo 1 extra y mide aceptación durante 2 semanas."
        ),
        strategy_template=(
            "Una sugerencia concreta reduce fricción y suele funcionar mejor que preguntar si quieren algo más."
        ),
        hypothesis_template=(
            "Si cierras con una recomendación concreta en vez de una pregunta abierta, debería mejorar la aceptación del extra y el ticket medio."
        ),
        where_to_apply_template="en el cierre de cada venta",
        duration_days=14,
        primary_metric_key="avg_ticket",
        primary_metric_label="Importe medio por operación",
        secondary_metric_keys=("acceptance_extra_rate", "ops_total"),
        secondary_metric_labels=("Aceptación del extra", "Operaciones registradas"),
        success_rule_text="Éxito si sube el importe medio y la recomendación se aplica con constancia suficiente.",
        confidence_label="media",
    ),
    RecommendationSpec(
        insight_id="guided_recommendation",
        title="Cambia la pregunta abierta por una recomendación guiada",
        goal="conversion",
        applies_to=("servicios", "retail", "unknown"),
        subtypes=("peluqueria_estetica", "ferreteria", "tienda_general", "unknown"),
        priority_weight=0.88,
        effort="bajo",
        time_to_apply_min=15,
        tags=("conversion", "guion", "venta"),
        trigger_fn=trig_always,
        impact_fn=impact_flat(60, 180),
        why_now_template=(
            "Es una mejora sencilla de aplicar y útil cuando quieres vender mejor sin cambiar el surtido ni tocar precios."
        ),
        action_template=(
            "Sustituye preguntas abiertas por una frase guiada que conecte {top_item} con un complemento lógico."
        ),
        if_already_doing_template=(
            "Si ya tienes un guion, simplifícalo: una frase breve y siempre igual suele ser más fácil de aplicar por todo el equipo."
        ),
        strategy_template=(
            "La consistencia en el cierre suele vender más que improvisar una recomendación distinta en cada venta."
        ),
        hypothesis_template=(
            "Si el cierre se estandariza con una frase guiada, la ejecución debería ser más constante y la conversión complementaria más fácil de medir."
        ),
        where_to_apply_template="en todas las ventas del periodo",
        duration_days=14,
        primary_metric_key="cross_sell_rate",
        primary_metric_label="Tasa de venta complementaria",
        secondary_metric_keys=("avg_ticket",),
        secondary_metric_labels=("Importe medio por operación",),
        success_rule_text="Éxito si aumenta la venta complementaria o mejora la consistencia del cierre.",
        confidence_label="media",
    ),
    RecommendationSpec(
        insight_id="measure_one_change_only",
        title="Mide una sola mejora durante 2 semanas",
        goal="aprendizaje",
        applies_to=("servicios", "retail", "restauracion", "unknown"),
        subtypes=("peluqueria_estetica", "ferreteria", "tienda_general", "bar_restaurante", "unknown"),
        priority_weight=0.72,
        effort="bajo",
        time_to_apply_min=10,
        tags=("medicion", "aprendizaje", "control"),
        trigger_fn=trig_always,
        impact_fn=impact_flat(0, 0),
        why_now_template=(
            "Si cambias demasiadas cosas a la vez, luego es difícil saber qué te ayudó de verdad."
        ),
        action_template=(
            "Elige una sola mejora de esta lista y mantenla 2 semanas antes de cambiar otra cosa."
        ),
        if_already_doing_template=(
            "Si ya haces pruebas, anota siempre qué cambiaste, cuándo empezó y en qué día o producto la aplicaste."
        ),
        strategy_template=(
            "El objetivo no es hacer más cosas, sino aprender qué acción concreta funciona mejor en tu negocio."
        ),
        hypothesis_template=(
            "Si reduces el número de cambios simultáneos, será más fácil identificar qué acción concreta funciona."
        ),
        where_to_apply_template="sobre una sola acción prioritaria",
        duration_days=14,
        primary_metric_key="experiment_clarity",
        primary_metric_label="Claridad de aprendizaje",
        secondary_metric_keys=("avg_ticket", "revenue_total"),
        secondary_metric_labels=("Importe medio por operación", "Ingresos del periodo"),
        success_rule_text="Éxito si puedes atribuir con más claridad el resultado a una acción concreta.",
        confidence_label="alta",
    ),
    RecommendationSpec(
        insight_id="worst_day_focus",
        title="Concentra una acción concreta en el día más flojo",
        goal="dia_flojo",
        applies_to=("servicios", "retail", "unknown"),
        subtypes=("peluqueria_estetica", "ferreteria", "tienda_general", "unknown"),
        priority_weight=0.96,
        effort="bajo",
        time_to_apply_min=15,
        tags=("dia_flojo", "prioridad", "accion"),
        trigger_fn=trig_worst_day_gap(min_gap_eur=50),
        impact_fn=impact_from_gap(0.25, 0.70),
        why_now_template=(
            "Tu día más flojo es {worst_day} y la diferencia frente al mejor día ronda los {worst_day_gap_eur} € en el periodo."
        ),
        action_template=(
            "Prueba una sola acción en {worst_day}: un pack simple, un extra concreto o una recomendación guiada ligada a {top_item}."
        ),
        if_already_doing_template=(
            "Si ya haces acciones ese día, limita la prueba a una sola franja o a una sola familia de productos para saber qué funciona de verdad."
        ),
        strategy_template=(
            "No repartas el esfuerzo por toda la semana. Empieza donde la mejora es más visible y más fácil de medir."
        ),
        hypothesis_template=(
            "Si concentras una sola mejora en {worst_day}, será más fácil cerrar parte de la brecha frente a {best_day}."
        ),
        where_to_apply_template="{worst_day}",
        duration_days=14,
        primary_metric_key="revenue_worst_day",
        primary_metric_label="Ingresos del día flojo",
        secondary_metric_keys=("avg_ticket_worst_day", "ops_worst_day"),
        secondary_metric_labels=("Importe medio del día flojo", "Operaciones del día flojo"),
        success_rule_text="Éxito si mejora el día flojo sin empeorar claramente el resto de la operativa.",
        confidence_label="alta",
    ),

    # --------------------------------------------------------
    # Servicios / peluquería / estética
    # --------------------------------------------------------
    RecommendationSpec(
        insight_id="rebook_closed_choice",
        title="Cierra la próxima visita con dos opciones concretas",
        goal="frecuencia",
        applies_to=("servicios",),
        subtypes=("peluqueria_estetica",),
        priority_weight=0.95,
        effort="bajo",
        time_to_apply_min=15,
        tags=("frecuencia", "rebooking", "servicios"),
        trigger_fn=trig_worst_day_gap(min_gap_eur=30),
        impact_fn=impact_flat(100, 320),
        why_now_template=(
            "En negocios de servicio, cerrar la siguiente visita ayuda a estabilizar demanda y proteger el día flojo."
        ),
        action_template=(
            "Al terminar el servicio, ofrece dos huecos concretos para la próxima visita en lugar de dejar la decisión abierta."
        ),
        if_already_doing_template=(
            "Si ya propones cita siguiente, registra cuándo se acepta más y prueba a ofrecer primero huecos en {worst_day} si quieres reforzarlo."
        ),
        strategy_template=(
            "La elección cerrada reduce indecisión y suele convertir mejor que dejar la vuelta para otro momento."
        ),
        hypothesis_template=(
            "Si cierras la siguiente visita con dos opciones concretas, debería aumentar la tasa de rebooking y reforzar la recurrencia."
        ),
        where_to_apply_template="al cerrar cada servicio",
        duration_days=14,
        primary_metric_key="rebooking_rate",
        primary_metric_label="Tasa de próxima visita cerrada",
        secondary_metric_keys=("ops_worst_day",),
        secondary_metric_labels=("Operaciones en el día flojo",),
        success_rule_text="Éxito si aumenta la tasa de cita siguiente cerrada respecto a la práctica anterior.",
        confidence_label="media",
    ),
)
# Nota:
# El resto de recomendaciones de tu catálogo se pueden migrar con el mismo patrón.
# Para no alargar demasiado el archivo aquí, puedes replicar la misma estructura
# en las entradas restantes conservando tu copy actual.


# ============================================================
# Matching
# ============================================================
def _matches_business(spec: RecommendationSpec, business_type: str, subtype: str) -> bool:
    """
    Matching ligeramente más robusto:
    - el spec puede aceptar el business_type exacto o 'unknown'
    - el spec puede aceptar el subtype exacto o 'unknown'
    """
    business_ok = (business_type in spec.applies_to) or ("unknown" in spec.applies_to)
    subtype_ok = (subtype in spec.subtypes) or ("unknown" in spec.subtypes)
    return business_ok and subtype_ok


# ============================================================
# Motor
# ============================================================
def build_recommendation_cards(
    ctx: dict,
    business_type: str,
    subtype: str,
) -> list[RecommendationCard]:
    """
    Filtra y construye tarjetas finales a partir del catálogo.
    """
    cards: list[RecommendationCard] = []

    for spec in RECOMMENDATION_LIBRARY:
        if not _matches_business(spec, business_type, subtype):
            continue
        if not spec.trigger_fn(ctx):
            continue

        weight = spec.priority_weight

        card = RecommendationCard(
            insight_id=spec.insight_id,
            title=spec.title,
            goal=spec.goal,
            why_now=_safe_fmt(spec.why_now_template, ctx),
            action=_safe_fmt(spec.action_template, ctx),
            if_already_doing=_safe_fmt(spec.if_already_doing_template, ctx),
            strategy=_safe_fmt(spec.strategy_template, ctx),
            hypothesis=_safe_fmt(spec.hypothesis_template, ctx),
            where_to_apply=_safe_fmt(spec.where_to_apply_template, ctx),
            duration_days=spec.duration_days,
            review_window_label=_review_window_label(spec.duration_days),
            primary_metric_key=spec.primary_metric_key,
            primary_metric_label=spec.primary_metric_label,
            secondary_metric_keys=spec.secondary_metric_keys,
            secondary_metric_labels=spec.secondary_metric_labels,
            success_rule_text=spec.success_rule_text,
            confidence_label=spec.confidence_label,
            effort=spec.effort,
            time_to_apply_min=spec.time_to_apply_min,
            estimated_impact_eur=spec.impact_fn(ctx),
            priority_weight=weight,
            priority_label=_priority_label(weight),
            tags=spec.tags,
        )
        cards.append(card)

    return cards


def apply_history_penalty(
    cards: Iterable[RecommendationCard],
    history_summary: dict | None = None,
) -> list[RecommendationCard]:
    """
    Si una recomendación ya se hizo, ya se vio o el negocio ya la aplica,
    le baja un poco el peso para favorecer rotación.
    """
    history_summary = history_summary or {}
    recent_done = set(history_summary.get("recent_done", []))
    recent_seen = set(history_summary.get("recent_seen", []))
    already_doing = set(history_summary.get("already_doing", []))
    recent_negative = set(history_summary.get("recent_negative", []))
    recent_positive = set(history_summary.get("recent_positive", []))

    rescored: list[RecommendationCard] = []

    for card in cards:
        penalty = 0.0
        bonus = 0.0

        if card.insight_id in recent_done:
            penalty += 0.18
        if card.insight_id in recent_seen:
            penalty += 0.10
        if card.insight_id in already_doing:
            penalty += 0.22
        if card.insight_id in recent_negative:
            penalty += 0.12
        if card.insight_id in recent_positive:
            bonus += 0.05

        new_weight = max(0.05, min(0.99, card.priority_weight - penalty + bonus))

        rescored.append(
            RecommendationCard(
                insight_id=card.insight_id,
                title=card.title,
                goal=card.goal,
                why_now=card.why_now,
                action=card.action,
                if_already_doing=card.if_already_doing,
                strategy=card.strategy,
                hypothesis=card.hypothesis,
                where_to_apply=card.where_to_apply,
                duration_days=card.duration_days,
                review_window_label=card.review_window_label,
                primary_metric_key=card.primary_metric_key,
                primary_metric_label=card.primary_metric_label,
                secondary_metric_keys=card.secondary_metric_keys,
                secondary_metric_labels=card.secondary_metric_labels,
                success_rule_text=card.success_rule_text,
                confidence_label=card.confidence_label,
                effort=card.effort,
                time_to_apply_min=card.time_to_apply_min,
                estimated_impact_eur=card.estimated_impact_eur,
                priority_weight=new_weight,
                priority_label=_priority_label(new_weight),
                tags=card.tags,
            )
        )

    return rescored


def select_top_cards(
    cards: Iterable[RecommendationCard],
    k: int = 3,
) -> list[RecommendationCard]:
    return sorted(cards, key=lambda x: x.priority_weight, reverse=True)[:k]