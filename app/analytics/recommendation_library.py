from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable


# =============================================================================
# OBJETIVO DE ESTA VERSIÓN
# - Ampliar y afinar catálogo por vertical.
# - Reforzar recomendaciones directas de ticket medio en servicios.
# - Revisar pesos base y triggers para que entren cuando toca.
# - Reducir dependencia excesiva de recomendaciones universales.
# - Mejorar equilibrio entre quick wins y acciones de continuidad.
#
# CRITERIO DE DISEÑO
# 1) Mantener compatibilidad con el resto del sistema.
# 2) No introducir lógica demasiado opaca.
# 3) Hacer el catálogo más "producto" y menos lista genérica.
#
# PROBLEMA DE LA VERSIÓN ORIGINAL
# - El catálogo universal tenía demasiado peso relativo.
# - Servicios/peluquería tenía recurrencia, sí, pero poco músculo específico en ticket.
# - Algunos pesos estaban demasiado altos en universales y demasiado bajos en verticales.
# - Faltaban algunas piezas intermedias: quick win vendible + continuidad medible.
# =============================================================================


# ============================================================
# Tipos base
# ============================================================
TriggerFn = Callable[[dict], bool]
ImpactFn = Callable[[dict], tuple[float, float]]


# ============================================================
# Modelos
# ============================================================
@dataclass(frozen=True)
class RecommendationSpec:
    """
    Definición reusable de una recomendación.

    Representa una acción operativa mínima, explicable y medible.
    """

    # Identidad
    insight_id: str
    title: str
    goal: str

    # Segmentación
    applies_to: tuple[str, ...]
    subtypes: tuple[str, ...]

    # Priorización base
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

    # Capa de experimento
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
    """Tarjeta final lista para mostrar en UI y usar en seguimiento."""

    # Identidad
    insight_id: str
    title: str
    goal: str

    # Mensaje
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

    # Score / fricción
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



def trig_ticket_push(
    min_ventas_peor: int = 4,
    min_subida_ticket_eur: float = 2.0,
) -> TriggerFn:
    """
    Trigger específico para acciones de ticket.
    Más fino que trig_low_ticket puro:
    - exige cierto volumen mínimo,
    - exige que el contexto esté simulando una mejora de ticket con sentido.
    """

    def _fn(ctx: dict) -> bool:
        ventas_ok = _ctx_int(ctx, "ventas_peor_dia", 0) >= int(min_ventas_peor)
        subida_ok = _ctx_float(ctx, "slider_subida_ticket_eur", 0.0) >= float(min_subida_ticket_eur)
        return ventas_ok and subida_ok

    return _fn



def trig_strong_ticket_or_gap(
    min_ventas_peor: int = 4,
    min_gap_eur: float = 35.0,
    min_subida_ticket_eur: float = 2.0,
) -> TriggerFn:
    """
    Trigger útil para servicios: entra si hay volumen razonable y además
    existe o una brecha visible en el día flojo o una palanca de ticket clara.
    """

    def _fn(ctx: dict) -> bool:
        ventas_ok = _ctx_int(ctx, "ventas_peor_dia", 0) >= int(min_ventas_peor)
        gap_ok = _ctx_float(ctx, "worst_day_gap_eur", 0.0) >= float(min_gap_eur)
        ticket_ok = _ctx_float(ctx, "slider_subida_ticket_eur", 0.0) >= float(min_subida_ticket_eur)
        return ventas_ok and (gap_ok or ticket_ok)

    return _fn



def trig_drop_or_worst_gap(
    min_drop_pct: float = 3.0,
    min_gap_eur: float = 40.0,
) -> TriggerFn:
    """Activa si hay caída frente a histórico o una brecha operativa clara."""

    def _fn(ctx: dict) -> bool:
        drop_ok = _ctx_float(ctx, "delta_ing_pct", 0.0) <= -abs(float(min_drop_pct))
        gap_ok = _ctx_float(ctx, "worst_day_gap_eur", 0.0) >= float(min_gap_eur)
        return drop_ok or gap_ok

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



def impact_from_ticket_plus_gap(
    ticket_low_mult: float,
    ticket_high_mult: float,
    gap_low_mult: float = 0.0,
    gap_high_mult: float = 0.0,
) -> ImpactFn:
    """
    Mezcla ticket + brecha del día flojo.
    Muy útil en servicios, donde aumentar ticket y estabilizar continuidad
    suelen convivir en la misma decisión comercial.
    """

    def _fn(ctx: dict) -> tuple[float, float]:
        ticket_base = _ctx_float(ctx, "ventas_peor_dia", 0) * _ctx_float(
            ctx, "slider_subida_ticket_eur", 0
        )
        gap_base = _ctx_float(ctx, "worst_day_gap_eur", 0.0)
        low = (ticket_base * float(ticket_low_mult)) + (gap_base * float(gap_low_mult))
        high = (ticket_base * float(ticket_high_mult)) + (gap_base * float(gap_high_mult))
        return (low, high)

    return _fn



def impact_flat(low: float, high: float) -> ImpactFn:
    def _fn(_: dict) -> tuple[float, float]:
        return (float(low), float(high))

    return _fn


# ============================================================
# Catálogo comercial afinado
# ============================================================
RECOMMENDATION_LIBRARY: tuple[RecommendationSpec, ...] = (
    # --------------------------------------------------------
    # Universales
    # --------------------------------------------------------
    # Ajuste importante:
    # - siguen existiendo porque son útiles para MVP cross-vertical,
    # - pero en general bajamos ligeramente su peso para que no pisen tanto a verticales.
    RecommendationSpec(
        insight_id="worst_day_focus",
        title="Concentra una acción concreta en el día más flojo",
        goal="dia_flojo",
        applies_to=("servicios", "retail", "restauracion", "unknown"),
        subtypes=(
            "peluqueria_estetica",
            "ferreteria",
            "tienda_general",
            "bar_restaurante",
            "unknown",
        ),
        priority_weight=0.90,
        effort="bajo",
        time_to_apply_min=15,
        tags=("dia_flojo", "prioridad", "accion", "quick_win"),
        trigger_fn=trig_worst_day_gap(min_gap_eur=50),
        impact_fn=impact_from_gap(0.25, 0.70),
        why_now_template=(
            "Tu día más flojo es {worst_day} y la diferencia frente al mejor día ronda los {worst_day_gap_eur} € en el periodo."
        ),
        action_template=(
            "Prueba una sola acción en {worst_day}: un pack simple, un extra concreto o una recomendación guiada ligada a {top_item}."
        ),
        if_already_doing_template=(
            "Si ya haces acciones ese día, limita la prueba a una sola franja o a una sola familia para saber qué funciona de verdad."
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
    RecommendationSpec(
        insight_id="pack_simple",
        title="Sube el importe medio con un pack simple",
        goal="ticket_medio",
        applies_to=("servicios", "retail", "unknown"),
        subtypes=("peluqueria_estetica", "ferreteria", "tienda_general", "unknown"),
        priority_weight=0.86,
        effort="bajo",
        time_to_apply_min=20,
        tags=("ticket", "pack", "valor", "quick_win"),
        trigger_fn=trig_ticket_push(min_ventas_peor=4, min_subida_ticket_eur=2.0),
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
        applies_to=("servicios", "retail", "restauracion", "unknown"),
        subtypes=(
            "peluqueria_estetica",
            "ferreteria",
            "tienda_general",
            "bar_restaurante",
            "unknown",
        ),
        priority_weight=0.84,
        effort="bajo",
        time_to_apply_min=10,
        tags=("ticket", "extra", "cierre", "quick_win"),
        trigger_fn=trig_ticket_push(min_ventas_peor=4, min_subida_ticket_eur=2.0),
        impact_fn=impact_from_ticket(0.6, 1.2),
        why_now_template=(
            "El volumen actual permite probar una mejora simple del cierre sin tocar precios."
        ),
        action_template=(
            "Al terminar la venta o cerrar el pedido, sugiere un complemento pequeño y lógico en vez de hacer una pregunta abierta."
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
        applies_to=("servicios", "retail", "restauracion", "unknown"),
        subtypes=(
            "peluqueria_estetica",
            "ferreteria",
            "tienda_general",
            "bar_restaurante",
            "unknown",
        ),
        priority_weight=0.80,
        effort="bajo",
        time_to_apply_min=15,
        tags=("conversion", "guion", "venta", "quick_win"),
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
        insight_id="protect_margin_no_discount",
        title="Sustituye descuento por valor añadido pequeño",
        goal="margen",
        applies_to=("servicios", "retail", "restauracion", "unknown"),
        subtypes=(
            "peluqueria_estetica",
            "ferreteria",
            "tienda_general",
            "bar_restaurante",
            "unknown",
        ),
        priority_weight=0.78,
        effort="medio",
        time_to_apply_min=25,
        tags=("margen", "valor", "precio", "estrategico"),
        trigger_fn=trig_prev_drop(min_drop_pct=3.0),
        impact_fn=impact_flat(80, 250),
        why_now_template=(
            "Si los ingresos van por debajo del periodo anterior, bajar precio puede empeorar el problema."
        ),
        action_template=(
            "En vez de recortar precio, añade un pequeño valor visible ligado a {top_item}: consejo, acabado, complemento o ayuda de uso."
        ),
        if_already_doing_template=(
            "Si ya ofreces valor añadido, hazlo más visible en el punto de venta o en el cierre para que el cliente lo perciba mejor."
        ),
        strategy_template=(
            "El objetivo es proteger margen mientras das una razón clara para elegirte o comprar un poco más."
        ),
        hypothesis_template=(
            "Si sustituyes descuento por valor añadido visible, deberías proteger mejor el margen sin empeorar la conversión."
        ),
        where_to_apply_template="en ventas sensibles a precio o comparativa",
        duration_days=14,
        primary_metric_key="gross_margin_signal",
        primary_metric_label="Señal de margen protegido",
        secondary_metric_keys=("avg_ticket", "revenue_total"),
        secondary_metric_labels=("Importe medio por operación", "Ingresos del periodo"),
        success_rule_text="Éxito si mantienes o mejoras ingresos sin depender de rebajas más agresivas.",
        confidence_label="media",
    ),
    RecommendationSpec(
        insight_id="premium_visibility",
        title="Haz más visible una opción de mayor valor",
        goal="mix_valor",
        applies_to=("servicios", "retail", "unknown"),
        subtypes=("peluqueria_estetica", "ferreteria", "tienda_general", "unknown"),
        priority_weight=0.79,
        effort="medio",
        time_to_apply_min=30,
        tags=("premium", "mix", "visibilidad", "ticket"),
        trigger_fn=trig_ticket_push(min_ventas_peor=3, min_subida_ticket_eur=2.0),
        impact_fn=impact_from_ticket(0.7, 1.3),
        why_now_template=(
            "Hay margen para mejorar el mix de ventas sin depender solo de volumen."
        ),
        action_template=(
            "Destaca una opción mejor o más completa relacionada con {top_item}, explicando por qué aporta más valor."
        ),
        if_already_doing_template=(
            "Si ya la tienes visible, revisa si la explicación es demasiado técnica o larga. Hazla más simple y orientada al beneficio."
        ),
        strategy_template=(
            "Una opción premium funciona mejor cuando el cliente entiende rápido qué gana al elegirla."
        ),
        hypothesis_template=(
            "Si haces más visible una opción superior con una explicación simple, debería mejorar el mix de valor por venta."
        ),
        where_to_apply_template="en el punto de decisión o cierre",
        duration_days=14,
        primary_metric_key="premium_mix_rate",
        primary_metric_label="Peso de opciones de mayor valor",
        secondary_metric_keys=("avg_ticket",),
        secondary_metric_labels=("Importe medio por operación",),
        success_rule_text="Éxito si aumentan las ventas de opciones de mayor valor sin añadir demasiada fricción.",
        confidence_label="media",
    ),
    RecommendationSpec(
        insight_id="reduce_dependency_top_item",
        title="Reduce la dependencia de un solo producto o servicio",
        goal="riesgo_mix",
        applies_to=("servicios", "retail", "unknown"),
        subtypes=("peluqueria_estetica", "ferreteria", "tienda_general", "unknown"),
        priority_weight=0.77,
        effort="medio",
        time_to_apply_min=30,
        tags=("mix", "riesgo", "diversificacion", "estrategico"),
        trigger_fn=trig_top_share_high(min_share_pct=35.0),
        impact_fn=impact_flat(70, 240),
        why_now_template=(
            "Ahora mismo una parte alta de tus ingresos depende de {top_item}, alrededor del {top_share_pct}% del total analizado."
        ),
        action_template=(
            "Usa {top_item} como puerta de entrada para recomendar una segunda opción compatible y reducir dependencia de una sola referencia."
        ),
        if_already_doing_template=(
            "Si ya haces venta cruzada desde ese producto, revisa si siempre recomiendas lo mismo. Rotar 2 o 3 opciones suele darte más aprendizaje."
        ),
        strategy_template=(
            "Diversificar mejor el mix reduce riesgo y te ayuda a crecer sin depender tanto de una única referencia fuerte."
        ),
        hypothesis_template=(
            "Si conviertes tu referencia fuerte en puerta de entrada para otra venta compatible, deberías reducir dependencia del top principal."
        ),
        where_to_apply_template="sobre ventas ligadas a {top_item}",
        duration_days=21,
        primary_metric_key="top_item_share",
        primary_metric_label="Peso del producto o servicio principal",
        secondary_metric_keys=("avg_ticket", "revenue_total"),
        secondary_metric_labels=("Importe medio por operación", "Ingresos del periodo"),
        success_rule_text="Éxito si baja la dependencia del top principal sin deteriorar ingresos.",
        confidence_label="media",
    ),
    RecommendationSpec(
        insight_id="measure_one_change_only",
        title="Mide una sola mejora durante 2 semanas",
        goal="aprendizaje",
        applies_to=("servicios", "retail", "restauracion", "unknown"),
        subtypes=(
            "peluqueria_estetica",
            "ferreteria",
            "tienda_general",
            "bar_restaurante",
            "unknown",
        ),
        priority_weight=0.70,
        effort="bajo",
        time_to_apply_min=10,
        tags=("medicion", "aprendizaje", "control", "continuidad"),
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

    # --------------------------------------------------------
    # Servicios / peluquería / estética
    # --------------------------------------------------------
    # Ajuste importante:
    # aquí metemos más variedad REAL de ticket + continuidad.
    RecommendationSpec(
        insight_id="service_upgrade_visible",
        title="Haz visible una versión mejor del servicio principal",
        goal="ticket_medio",
        applies_to=("servicios",),
        subtypes=("peluqueria_estetica",),
        priority_weight=0.93,
        effort="bajo",
        time_to_apply_min=20,
        tags=("ticket", "upgrade", "servicios", "quick_win"),
        trigger_fn=trig_strong_ticket_or_gap(min_ventas_peor=4, min_gap_eur=35.0, min_subida_ticket_eur=2.0),
        impact_fn=impact_from_ticket_plus_gap(0.90, 1.80, 0.05, 0.15),
        why_now_template=(
            "En servicios hay margen para mejorar el importe medio si haces más visible una opción superior ligada a {top_item}."
        ),
        action_template=(
            "Presenta una versión mejor del servicio principal con una diferencia clara de valor, no solo de precio, y pruébala primero en {worst_day}."
        ),
        if_already_doing_template=(
            "Si ya tienes una opción superior, simplifica cómo la explicas: una razón clara para elegirla suele vender mejor que una lista larga de detalles."
        ),
        strategy_template=(
            "La mejora de ticket en servicios suele llegar cuando el cliente percibe un siguiente nivel lógico y fácil de entender."
        ),
        hypothesis_template=(
            "Si explicas mejor una versión superior del servicio, debería subir el importe medio sin necesidad de descuentos."
        ),
        where_to_apply_template="en presupuestos, mostrador o cierre del servicio",
        duration_days=14,
        primary_metric_key="avg_ticket",
        primary_metric_label="Importe medio por servicio",
        secondary_metric_keys=("premium_mix_rate", "revenue_worst_day"),
        secondary_metric_labels=("Peso de servicio superior", "Ingresos del día flojo"),
        success_rule_text="Éxito si sube el importe medio o gana peso la opción superior con una ejecución consistente.",
        confidence_label="alta",
    ),
    RecommendationSpec(
        insight_id="service_addon_closure",
        title="Cierra con un complemento de mantenimiento lógico",
        goal="ticket_medio",
        applies_to=("servicios",),
        subtypes=("peluqueria_estetica",),
        priority_weight=0.91,
        effort="bajo",
        time_to_apply_min=10,
        tags=("ticket", "add_on", "servicios", "quick_win"),
        trigger_fn=trig_ticket_push(min_ventas_peor=4, min_subida_ticket_eur=2.0),
        impact_fn=impact_from_ticket(0.85, 1.55),
        why_now_template=(
            "Hay margen para cerrar cada servicio con una propuesta pequeña y lógica que eleve el importe medio."
        ),
        action_template=(
            "Al terminar {top_item}, propone un complemento de mantenimiento, acabado o cuidado posterior en vez de dejar el cierre abierto."
        ),
        if_already_doing_template=(
            "Si ya recomiendas algo al final, reduce opciones y deja una única recomendación principal para medir mejor aceptación."
        ),
        strategy_template=(
            "En servicios, un complemento pequeño funciona mejor cuando se presenta como continuidad útil y no como venta forzada."
        ),
        hypothesis_template=(
            "Si el cierre incluye un complemento lógico y repetible, debería aumentar el importe medio con poca fricción."
        ),
        where_to_apply_template="al finalizar cada servicio",
        duration_days=14,
        primary_metric_key="avg_ticket",
        primary_metric_label="Importe medio por servicio",
        secondary_metric_keys=("acceptance_extra_rate",),
        secondary_metric_labels=("Aceptación del complemento",),
        success_rule_text="Éxito si mejora el importe medio y la recomendación se aplica con regularidad.",
        confidence_label="alta",
    ),
    RecommendationSpec(
        insight_id="service_bundle_2step",
        title="Convierte un servicio suelto en una propuesta de dos pasos",
        goal="ticket_medio",
        applies_to=("servicios",),
        subtypes=("peluqueria_estetica",),
        priority_weight=0.88,
        effort="medio",
        time_to_apply_min=30,
        tags=("ticket", "bundle", "servicios", "estrategico"),
        trigger_fn=trig_strong_ticket_or_gap(min_ventas_peor=4, min_gap_eur=40.0, min_subida_ticket_eur=2.0),
        impact_fn=impact_from_ticket_plus_gap(1.00, 2.00, 0.08, 0.20),
        why_now_template=(
            "Si quieres subir el valor por servicio sin depender solo de más volumen, puedes estructurar una propuesta de mayor valor alrededor de {top_item}."
        ),
        action_template=(
            "Diseña una propuesta simple de dos pasos: servicio principal + complemento natural, y úsala primero en {worst_day} o en cierres con más margen."
        ),
        if_already_doing_template=(
            "Si ya combinas servicios, revisa si el cliente entiende rápido qué gana. Una estructura más clara suele convertir mejor que muchas combinaciones."
        ),
        strategy_template=(
            "El objetivo es pasar de vender un servicio aislado a vender una solución más completa y de valor superior."
        ),
        hypothesis_template=(
            "Si conviertes servicios sueltos en una propuesta de dos pasos clara, debería subir el importe medio y mejorar el mix de valor."
        ),
        where_to_apply_template="en cierres, presupuestos y recomendaciones del equipo",
        duration_days=21,
        primary_metric_key="avg_ticket",
        primary_metric_label="Importe medio por servicio",
        secondary_metric_keys=("premium_mix_rate", "cross_sell_rate"),
        secondary_metric_labels=("Peso de combinaciones de mayor valor", "Tasa de venta complementaria"),
        success_rule_text="Éxito si aumenta el valor medio por servicio y crecen las ventas de combinaciones útiles.",
        confidence_label="media",
    ),
    RecommendationSpec(
        insight_id="rebook_closed_choice",
        title="Cierra la próxima visita con dos opciones concretas",
        goal="frecuencia",
        applies_to=("servicios",),
        subtypes=("peluqueria_estetica",),
        priority_weight=0.92,
        effort="bajo",
        time_to_apply_min=15,
        tags=("frecuencia", "rebooking", "servicios", "continuidad"),
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
        confidence_label="alta",
    ),
    RecommendationSpec(
        insight_id="service_continuity",
        title="Empuja servicios de continuidad o mantenimiento",
        goal="continuidad",
        applies_to=("servicios",),
        subtypes=("peluqueria_estetica",),
        priority_weight=0.86,
        effort="bajo",
        time_to_apply_min=20,
        tags=("continuidad", "servicios", "recurrencia", "continuidad"),
        trigger_fn=trig_always,
        impact_fn=impact_flat(90, 260),
        why_now_template=(
            "Cuando el negocio depende de la recurrencia, vender continuidad suele ser más rentable que esperar a que el cliente vuelva por su cuenta."
        ),
        action_template=(
            "Define una propuesta simple de mantenimiento asociada a {top_item} y recomiéndala al cerrar el servicio."
        ),
        if_already_doing_template=(
            "Si ya lo haces, ajusta el mensaje según el tipo de servicio previo para que la recomendación suene más natural y útil."
        ),
        strategy_template=(
            "La continuidad funciona mejor cuando se presenta como el siguiente paso lógico, no como una venta añadida."
        ),
        hypothesis_template=(
            "Si presentas continuidad como siguiente paso lógico, debería mejorar la recurrencia futura y la estabilidad del negocio."
        ),
        where_to_apply_template="al cierre del servicio principal",
        duration_days=21,
        primary_metric_key="continuity_offer_acceptance",
        primary_metric_label="Aceptación de propuesta de continuidad",
        secondary_metric_keys=("rebooking_rate",),
        secondary_metric_labels=("Tasa de próxima visita cerrada",),
        success_rule_text="Éxito si más clientes aceptan una continuidad clara o agendan el siguiente paso.",
        confidence_label="media",
    ),
    RecommendationSpec(
        insight_id="reactivate_clients_service",
        title="Recupera clientes que llevan más tiempo sin volver",
        goal="reactivacion",
        applies_to=("servicios",),
        subtypes=("peluqueria_estetica",),
        priority_weight=0.81,
        effort="medio",
        time_to_apply_min=35,
        tags=("reactivacion", "clientes", "servicios", "continuidad"),
        trigger_fn=trig_worst_day_gap(min_gap_eur=40),
        impact_fn=impact_flat(80, 260),
        why_now_template=(
            "Cuando hay un día flojo claro, reactivar clientes puede ser más eficiente que intentar captar desde cero."
        ),
        action_template=(
            "Haz una lista de clientes que hace tiempo no vuelven y contacta con una propuesta concreta pensada para {worst_day}."
        ),
        if_already_doing_template=(
            "Si ya haces reactivación, personaliza el mensaje según el último servicio realizado y evita mensajes demasiado genéricos."
        ),
        strategy_template=(
            "Recuperar clientes antiguos suele costar menos que generar demanda nueva, sobre todo en periodos flojos."
        ),
        hypothesis_template=(
            "Si contactas de forma concreta a clientes inactivos con una propuesta ligada al día flojo, deberías recuperar parte de la demanda."
        ),
        where_to_apply_template="sobre clientes inactivos o dormidos",
        duration_days=14,
        primary_metric_key="reactivation_response_rate",
        primary_metric_label="Respuesta de clientes reactivados",
        secondary_metric_keys=("ops_worst_day", "revenue_worst_day"),
        secondary_metric_labels=("Operaciones en el día flojo", "Ingresos en el día flojo"),
        success_rule_text="Éxito si recuperas respuesta útil o visitas reales en el día flojo.",
        confidence_label="media",
    ),

    # --------------------------------------------------------
    # Retail / ferretería / tienda general
    # --------------------------------------------------------
    RecommendationSpec(
        insight_id="top3_complements",
        title="Haz visibles los 3 complementos más útiles",
        goal="venta_cruzada",
        applies_to=("retail",),
        subtypes=("ferreteria", "tienda_general"),
        priority_weight=0.87,
        effort="bajo",
        time_to_apply_min=20,
        tags=("cross_sell", "retail", "visibilidad", "ticket"),
        trigger_fn=trig_always,
        impact_fn=impact_flat(70, 220),
        why_now_template=(
            "Cuando el cliente no ve rápido qué acompaña una compra, se pierde venta complementaria."
        ),
        action_template=(
            "Selecciona 3 complementos frecuentes y colócalos junto a la categoría o producto base que más salida tiene ahora."
        ),
        if_already_doing_template=(
            "Si ya tienes complementos visibles, reduce la cantidad. Ver pocas opciones útiles suele funcionar mejor que enseñar demasiadas."
        ),
        strategy_template=(
            "En retail, vender la solución completa suele ser más potente que esperar a que el cliente recuerde cada accesorio por su cuenta."
        ),
        hypothesis_template=(
            "Si muestras solo 3 complementos realmente útiles junto al producto base, debería subir la venta cruzada sin generar ruido."
        ),
        where_to_apply_template="junto a {top_item} o su categoría asociada",
        duration_days=14,
        primary_metric_key="cross_sell_rate",
        primary_metric_label="Tasa de venta complementaria",
        secondary_metric_keys=("avg_ticket",),
        secondary_metric_labels=("Importe medio por operación",),
        success_rule_text="Éxito si sube la venta complementaria o el valor medio de la compra.",
        confidence_label="alta",
    ),
    RecommendationSpec(
        insight_id="solution_by_task",
        title="Agrupa productos por tarea, no solo por categoría",
        goal="venta_solucion_completa",
        applies_to=("retail",),
        subtypes=("ferreteria", "tienda_general"),
        priority_weight=0.85,
        effort="medio",
        time_to_apply_min=45,
        tags=("retail", "solucion", "agrupacion", "estrategico"),
        trigger_fn=trig_always,
        impact_fn=impact_flat(90, 300),
        why_now_template=(
            "Para muchos negocios de retail, ordenar por necesidad del cliente vende mejor que ordenar solo por familia de producto."
        ),
        action_template=(
            "Crea una mini propuesta tipo 'para hacer X necesitas esto' y apóyate en tus productos con más salida, como {top_item}."
        ),
        if_already_doing_template=(
            "Si ya agrupas por tarea, prueba títulos más concretos y una selección más corta para que la propuesta se entienda en segundos."
        ),
        strategy_template=(
            "La venta aumenta cuando el cliente percibe que le estás resolviendo una tarea completa y no vendiéndole piezas sueltas."
        ),
        hypothesis_template=(
            "Si presentas conjuntos por tarea real del cliente, debería subir la compra completa y el valor por operación."
        ),
        where_to_apply_template="en exposición, mesa o lineal temático",
        duration_days=21,
        primary_metric_key="complete_solution_sales",
        primary_metric_label="Ventas de solución completa",
        secondary_metric_keys=("avg_ticket",),
        secondary_metric_labels=("Importe medio por operación",),
        success_rule_text="Éxito si más clientes compran el conjunto completo o sube el valor medio del ticket.",
        confidence_label="media",
    ),
    RecommendationSpec(
        insight_id="quick_purchase_visibility",
        title="Destaca una compra rápida en horas o días flojos",
        goal="movimiento_rapido",
        applies_to=("retail",),
        subtypes=("ferreteria", "tienda_general"),
        priority_weight=0.80,
        effort="bajo",
        time_to_apply_min=15,
        tags=("retail", "rotacion", "dia_flojo", "quick_win"),
        trigger_fn=trig_worst_day_gap(min_gap_eur=40),
        impact_fn=impact_flat(60, 180),
        why_now_template=(
            "Cuando hay un día flojo claro, empujar una compra rápida y sencilla puede activar movimiento sin complicar la operación."
        ),
        action_template=(
            "Destaca una referencia de compra fácil o frecuente durante {worst_day}, cerca de la zona de paso o cierre."
        ),
        if_already_doing_template=(
            "Si ya destacas productos en ese día, prueba a cambiar solo una referencia cada 2 semanas para ver cuál mueve mejor la venta."
        ),
        strategy_template=(
            "Las compras rápidas funcionan mejor cuando son visibles, fáciles de entender y no exigen demasiada decisión."
        ),
        hypothesis_template=(
            "Si das más visibilidad a una compra rápida en el día flojo, debería aumentar el movimiento sin añadir fricción operativa."
        ),
        where_to_apply_template="{worst_day}",
        duration_days=14,
        primary_metric_key="revenue_worst_day",
        primary_metric_label="Ingresos del día flojo",
        secondary_metric_keys=("ops_worst_day",),
        secondary_metric_labels=("Operaciones en el día flojo",),
        success_rule_text="Éxito si el día flojo gana movimiento o ingresos sin complejidad extra.",
        confidence_label="media",
    ),
    RecommendationSpec(
        insight_id="avoid_second_visit",
        title="Sugiere el accesorio que evita una segunda visita",
        goal="venta_utilidad",
        applies_to=("retail",),
        subtypes=("ferreteria",),
        priority_weight=0.90,
        effort="bajo",
        time_to_apply_min=15,
        tags=("ferreteria", "complemento", "utilidad", "ticket"),
        trigger_fn=trig_always,
        impact_fn=impact_flat(80, 230),
        why_now_template=(
            "En ferretería, una recomendación útil puede evitar compras incompletas y aumentar el valor de la venta sin forzarla."
        ),
        action_template=(
            "Cuando vendas {top_item} u otras referencias parecidas, sugiere el accesorio o consumible que evita volver por una pieza olvidada."
        ),
        if_already_doing_template=(
            "Si ya lo haces, acota la sugerencia a 1 o 2 accesorios clave en vez de enumerar demasiadas opciones."
        ),
        strategy_template=(
            "La mejor venta cruzada en ferretería no suena a venta añadida: suena a ayuda para que el trabajo salga bien a la primera."
        ),
        hypothesis_template=(
            "Si recomiendas el accesorio que evita una segunda visita, debería subir la utilidad percibida y el valor medio de la compra."
        ),
        where_to_apply_template="en ventas ligadas a tareas técnicas o de reparación",
        duration_days=14,
        primary_metric_key="cross_sell_rate",
        primary_metric_label="Tasa de venta complementaria útil",
        secondary_metric_keys=("avg_ticket",),
        secondary_metric_labels=("Importe medio por operación",),
        success_rule_text="Éxito si se acepta mejor la recomendación y crece el valor de la venta.",
        confidence_label="alta",
    ),
    RecommendationSpec(
        insight_id="quick_project_bundle",
        title="Convierte una compra puntual en una solución de proyecto",
        goal="ticket_medio",
        applies_to=("retail",),
        subtypes=("ferreteria", "tienda_general"),
        priority_weight=0.86,
        effort="medio",
        time_to_apply_min=30,
        tags=("ticket", "bundle", "retail", "estrategico"),
        trigger_fn=trig_strong_ticket_or_gap(min_ventas_peor=4, min_gap_eur=35.0, min_subida_ticket_eur=2.0),
        impact_fn=impact_from_ticket_plus_gap(0.75, 1.50, 0.04, 0.12),
        why_now_template=(
            "Hay margen para pasar de vender piezas sueltas a vender una compra más completa y de mayor valor."
        ),
        action_template=(
            "Prepara una propuesta cerrada tipo proyecto o tarea completa alrededor de {top_item}, con 2 o 3 elementos clave fáciles de explicar."
        ),
        if_already_doing_template=(
            "Si ya haces propuestas completas, simplifica el mensaje y deja muy claro qué problema resuelve el conjunto."
        ),
        strategy_template=(
            "Cuando el cliente entiende la solución completa, suele comprar mejor y olvidar menos accesorios importantes."
        ),
        hypothesis_template=(
            "Si empaquetas compras frecuentes como solución completa, debería subir el ticket medio y la compra más útil por visita."
        ),
        where_to_apply_template="en referencias o categorías de compra recurrente",
        duration_days=21,
        primary_metric_key="avg_ticket",
        primary_metric_label="Importe medio por operación",
        secondary_metric_keys=("complete_solution_sales", "cross_sell_rate"),
        secondary_metric_labels=("Ventas de solución completa", "Tasa de venta complementaria"),
        success_rule_text="Éxito si sube el ticket medio y más clientes compran el conjunto completo.",
        confidence_label="media",
    ),
    RecommendationSpec(
        insight_id="slow_item_visibility",
        title="Da más salida a una referencia con poca visibilidad",
        goal="visibilidad_producto",
        applies_to=("retail",),
        subtypes=("tienda_general",),
        priority_weight=0.75,
        effort="medio",
        time_to_apply_min=25,
        tags=("tienda", "visibilidad", "producto", "continuidad"),
        trigger_fn=trig_always,
        impact_fn=impact_flat(50, 170),
        why_now_template=(
            "No todo el crecimiento viene de vender más de lo que ya destaca; a veces mejora más una referencia útil mal expuesta."
        ),
        action_template=(
            "Elige una referencia que tenga sentido junto a {top_item} y dale una posición más visible o una explicación más clara."
        ),
        if_already_doing_template=(
            "Si ya rotas producto visible, cambia solo una variable cada vez: posición, texto o acompañamiento."
        ),
        strategy_template=(
            "La visibilidad mejora cuando el cliente entiende rápido para qué sirve el producto y por qué le conviene ahora."
        ),
        hypothesis_template=(
            "Si una referencia útil gana visibilidad y contexto, debería aumentar su salida sin depender de rebaja."
        ),
        where_to_apply_template="sobre una referencia secundaria útil",
        duration_days=14,
        primary_metric_key="slow_item_sales",
        primary_metric_label="Salida de referencia secundaria",
        secondary_metric_keys=("avg_ticket",),
        secondary_metric_labels=("Importe medio por operación",),
        success_rule_text="Éxito si gana tracción una referencia útil que antes pasaba desapercibida.",
        confidence_label="baja",
    ),

    # --------------------------------------------------------
    # Restauración
    # --------------------------------------------------------
    RecommendationSpec(
        insight_id="restaurant_focus_worst_day",
        title="Refuerza el día más flojo con una propuesta simple",
        goal="dia_flojo",
        applies_to=("restauracion",),
        subtypes=("bar_restaurante",),
        priority_weight=0.83,
        effort="bajo",
        time_to_apply_min=20,
        tags=("restauracion", "dia_flojo", "propuesta", "quick_win"),
        trigger_fn=trig_worst_day_gap(min_gap_eur=50),
        impact_fn=impact_from_gap(0.20, 0.55),
        why_now_template=(
            "Tu día más flojo es {worst_day} y conviene concentrar ahí una acción sencilla antes de cambiar varias cosas a la vez."
        ),
        action_template=(
            "Define una propuesta simple y visible solo para {worst_day}, fácil de entender y de aplicar por el equipo."
        ),
        if_already_doing_template=(
            "Si ya haces acciones en ese día, reduce la complejidad y deja una sola propuesta para medir mejor si funciona."
        ),
        strategy_template=(
            "En restauración suele funcionar mejor una propuesta clara y repetible que muchas acciones pequeñas a la vez."
        ),
        hypothesis_template=(
            "Si refuerzas el día flojo con una sola propuesta clara, debería mejorar el movimiento o el ticket sin dispersión."
        ),
        where_to_apply_template="{worst_day}",
        duration_days=14,
        primary_metric_key="revenue_worst_day",
        primary_metric_label="Ingresos del día flojo",
        secondary_metric_keys=("avg_ticket_worst_day", "ops_worst_day"),
        secondary_metric_labels=("Ticket medio del día flojo", "Operaciones del día flojo"),
        success_rule_text="Éxito si el día flojo gana tracción con una propuesta simple y consistente.",
        confidence_label="media",
    ),
    RecommendationSpec(
        insight_id="restaurant_guided_extra",
        title="Sugiere un complemento claro al cerrar el pedido",
        goal="ticket_medio",
        applies_to=("restauracion",),
        subtypes=("bar_restaurante",),
        priority_weight=0.84,
        effort="bajo",
        time_to_apply_min=10,
        tags=("restauracion", "ticket", "complemento", "quick_win"),
        trigger_fn=trig_ticket_push(min_ventas_peor=4, min_subida_ticket_eur=2.0),
        impact_fn=impact_from_ticket(0.5, 1.0),
        why_now_template=(
            "Hay margen para subir el importe medio con una recomendación sencilla al cerrar el pedido."
        ),
        action_template=(
            "Al cerrar el pedido, recomienda un complemento concreto y habitual en lugar de dejar la decisión abierta."
        ),
        if_already_doing_template=(
            "Si ya se hacen sugerencias, deja una sola frase estándar para que el equipo la aplique con más constancia."
        ),
        strategy_template=(
            "La recomendación guiada funciona mejor cuando es breve, natural y fácil de repetir."
        ),
        hypothesis_template=(
            "Si el cierre del pedido incluye una sugerencia concreta y repetible, debería subir el ticket medio con baja fricción."
        ),
        where_to_apply_template="en el cierre de cada pedido",
        duration_days=14,
        primary_metric_key="avg_ticket",
        primary_metric_label="Ticket medio",
        secondary_metric_keys=("ops_total",),
        secondary_metric_labels=("Pedidos registrados",),
        success_rule_text="Éxito si sube el ticket medio sin deteriorar el ritmo de atención.",
        confidence_label="media",
    ),
    RecommendationSpec(
        insight_id="restaurant_menu_pairing",
        title="Empareja el producto principal con un acompañamiento rentable",
        goal="ticket_medio",
        applies_to=("restauracion",),
        subtypes=("bar_restaurante",),
        priority_weight=0.86,
        effort="bajo",
        time_to_apply_min=15,
        tags=("restauracion", "ticket", "pairing", "quick_win"),
        trigger_fn=trig_ticket_push(min_ventas_peor=4, min_subida_ticket_eur=2.0),
        impact_fn=impact_from_ticket(0.65, 1.20),
        why_now_template=(
            "En restauración, el ticket medio suele mejorar cuando el equipo empareja bien el pedido base con un acompañamiento lógico."
        ),
        action_template=(
            "Define 1 o 2 emparejamientos claros a partir de {top_item} y haz que el equipo los sugiera siempre con una frase breve."
        ),
        if_already_doing_template=(
            "Si ya se sugieren acompañamientos, reduce variedad y prioriza los que dejan mejor margen o mejor aceptación."
        ),
        strategy_template=(
            "No se trata de ofrecer de todo, sino de hacer muy fácil la elección adicional más natural y rentable."
        ),
        hypothesis_template=(
            "Si el producto base se acompaña siempre con una sugerencia rentable y coherente, debería subir el ticket medio."
        ),
        where_to_apply_template="en comandas y cierre de pedido",
        duration_days=14,
        primary_metric_key="avg_ticket",
        primary_metric_label="Ticket medio",
        secondary_metric_keys=("cross_sell_rate",),
        secondary_metric_labels=("Aceptación de acompañamiento",),
        success_rule_text="Éxito si aumenta la aceptación del acompañamiento y mejora el ticket medio.",
        confidence_label="alta",
    ),
    RecommendationSpec(
        insight_id="restaurant_repeat_visit_push",
        title="Empuja una segunda visita con una razón concreta para volver",
        goal="recurrencia",
        applies_to=("restauracion",),
        subtypes=("bar_restaurante",),
        priority_weight=0.80,
        effort="medio",
        time_to_apply_min=25,
        tags=("restauracion", "recurrencia", "continuidad", "estrategico"),
        trigger_fn=trig_drop_or_worst_gap(min_drop_pct=3.0, min_gap_eur=45.0),
        impact_fn=impact_flat(90, 240),
        why_now_template=(
            "Cuando hay caída o un día flojo claro, mover una segunda visita puede ser más rentable que depender solo del tráfico espontáneo."
        ),
        action_template=(
            "Define una razón concreta para volver en pocos días: nueva sugerencia, franja recomendada o propuesta asociada a {worst_day}."
        ),
        if_already_doing_template=(
            "Si ya intentas repetir visita, haz la invitación más específica y menos genérica para que el cliente recuerde mejor cuándo volver."
        ),
        strategy_template=(
            "La recurrencia mejora cuando se da un motivo claro y cercano para volver, no una invitación vaga."
        ),
        hypothesis_template=(
            "Si das una razón concreta para volver pronto, debería aumentar la segunda visita o el retorno en el periodo siguiente."
        ),
        where_to_apply_template="al final del servicio o experiencia",
        duration_days=21,
        primary_metric_key="repeat_visit_signal",
        primary_metric_label="Señal de segunda visita",
        secondary_metric_keys=("revenue_worst_day",),
        secondary_metric_labels=("Ingresos del día flojo",),
        success_rule_text="Éxito si aparecen más repeticiones o mejora la señal de retorno en el periodo siguiente.",
        confidence_label="media",
    ),
)


# ============================================================
# Matching
# ============================================================
def _matches_business(spec: RecommendationSpec, business_type: str, subtype: str) -> bool:
    """
    Matching robusto:
    - acepta business_type exacto o fallback unknown
    - acepta subtype exacto o fallback unknown
    """
    business_ok = (business_type in spec.applies_to) or ("unknown" in spec.applies_to)
    subtype_ok = (subtype in spec.subtypes) or ("unknown" in spec.subtypes)
    return business_ok and subtype_ok


# ============================================================
# Construcción de cards
# ============================================================
def build_recommendation_cards(
    ctx: dict,
    business_type: str,
    subtype: str,
) -> list[RecommendationCard]:
    """
    Filtra y construye tarjetas finales a partir del catálogo.

    Nota de diseño:
    aquí no meto lógica extra de score por vertical.
    El catálogo ya viene más afinado por pesos y triggers.
    Eso hace el sistema más transparente y fácil de mantener.
    """
    cards: list[RecommendationCard] = []

    for spec in RECOMMENDATION_LIBRARY:
        if not _matches_business(spec, business_type, subtype):
            continue
        if not spec.trigger_fn(ctx):
            continue

        weight = spec.priority_weight

        cards.append(
            RecommendationCard(
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
        )

    return cards


# ============================================================
# Penalización ligera por histórico
# ============================================================
def apply_history_penalty(
    cards: Iterable[RecommendationCard],
    history_summary: dict | None = None,
) -> list[RecommendationCard]:
    """
    Ajuste ligero para favorecer rotación y evitar insistencia excesiva.

    Esta capa sigue siendo deliberadamente suave.
    En esta versión reduzco un poco castigos fuertes para no matar demasiado
    recomendaciones verticales potentes que sí pueden seguir siendo válidas.
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
            penalty += 0.14
        if card.insight_id in recent_seen:
            penalty += 0.08
        if card.insight_id in already_doing:
            penalty += 0.16
        if card.insight_id in recent_negative:
            penalty += 0.10
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