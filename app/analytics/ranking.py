from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from math import sqrt
from typing import Iterable, Optional

from .history_store import HistoryStore
from .recommendation_library import RecommendationCard


# =============================================================================
# OBJETIVO DE ESTA VERSIÓN
# - Recalibrar el ranking para que el top 3 mezcle mejor:
#   1) acción estratégica,
#   2) subida de ticket,
#   3) recurrencia/retención.
# - Dar más peso a acciones de ticket cuando realmente parezcan prometedoras.
# - Evitar que una recomendación con impacto bruto muy alto arrase a las demás.
# - Mantener diversidad por familia, pero sin forzar resultados absurdos.
#
# IDEA CLAVE:
# Antes el score dependía demasiado del impacto relativo lineal contra el máximo.
# Eso provoca dos problemas:
#   - si una card tiene un impacto muy superior, aplasta al resto;
#   - el ranking queda demasiado "monocultivo".
#
# En esta versión:
# - comprimimos el impacto con sqrt() para reducir dominancia;
# - añadimos bonus por intención/familia (ticket, recurrencia, estratégico);
# - metemos un bonus de equilibrio para que el top mezcle mejor familias útiles;
# - seguimos usando select_plan con diversidad pragmática.
# =============================================================================


# ============================================================
# Modelo principal
# ============================================================
@dataclass(frozen=True)
class ScoredRecommendation:
    """Recomendación puntuada y lista para selección final."""

    card: RecommendationCard
    score: float
    reason: str


# ============================================================
# Helpers de score base
# ============================================================
def _impact_mid(card: RecommendationCard) -> float:
    """Impacto medio estimado en euros."""
    low, high = card.estimated_impact_eur
    return 0.5 * (float(low) + float(high))



def _effort_penalty(effort: str) -> float:
    """
    Penalización por fricción operativa.
    Ajustada para no castigar en exceso acciones útiles de valor medio/alto.
    """
    e = (effort or "").strip().lower()
    return {
        "bajo": 0.00,
        "media": 0.08,
        "medio": 0.08,
        "alta": 0.18,
    }.get(e, 0.10)



def _time_penalty(minutes: int) -> float:
    """
    Penalización suave por tiempo estimado.
    Menos agresiva que antes para no hundir acciones estratégicas razonables.
    """
    try:
        m = int(minutes)
    except Exception:
        m = 30

    # Antes era m/240 hasta 0.25: demasiado lineal y agresivo.
    # Ahora reducimos pendiente y techo.
    return min(0.16, max(0.0, m / 360.0))



def _confidence_bonus(confidence_label: str) -> float:
    """Bonus/penalización por confianza estimada."""
    c = (confidence_label or "").strip().lower()
    return {
        "alta": 0.10,
        "media": 0.04,
        "baja": -0.04,
    }.get(c, 0.0)



def _priority_editorial_bonus(priority_weight: float) -> float:
    """
    Bonus suave por prioridad editorial/base del catálogo.
    Mantiene influencia, pero no debe mandar sobre todo lo demás.
    """
    return max(0.0, min(0.18, (float(priority_weight) - 0.68) * 0.7))



def _quick_win_bonus(card: RecommendationCard) -> float:
    """
    Bonus ligero para acciones fáciles de vender y aplicar.
    Sigue existiendo, pero ya no domina sobre valor estratégico.
    """
    bonus = 0.0

    effort = (card.effort or "").strip().lower()
    if effort == "bajo":
        bonus += 0.03

    try:
        mins = int(card.time_to_apply_min)
    except Exception:
        mins = 30

    if mins <= 15:
        bonus += 0.03
    elif mins <= 30:
        bonus += 0.015

    if (card.confidence_label or "").strip().lower() == "alta":
        bonus += 0.015

    return min(0.07, bonus)



def _safe_text_parts(card: RecommendationCard) -> list[str]:
    """
    Junta texto útil para inferir familia/intención sin depender de un único campo.
    Esto hace el ranking más robusto si el catálogo no es totalmente homogéneo.
    """
    values = [
        getattr(card, "title", ""),
        getattr(card, "goal", ""),
        getattr(card, "action", ""),
        getattr(card, "why_now", ""),
        getattr(card, "strategy", ""),
        " ".join(getattr(card, "tags", []) or []),
        getattr(card, "primary_metric_label", ""),
    ]
    return [str(v).strip().lower() for v in values if str(v).strip()]



def _contains_any(text_parts: list[str], keywords: tuple[str, ...]) -> bool:
    blob = " | ".join(text_parts)
    return any(k in blob for k in keywords)



def _ticket_signal(card: RecommendationCard) -> float:
    """
    Detecta si la recomendación empuja aumento de ticket / venta adicional.

    Se le da más peso porque el usuario quiere que verticales con margen claro
    no infra-prioricen este tipo de acciones.

    No usamos un bonus enorme: suficiente para empujar, no para distorsionar.
    """
    txt = _safe_text_parts(card)

    strong = (
        "ticket",
        "importe medio",
        "upsell",
        "up-sell",
        "venta adicional",
        "venta cruzada",
        "cross sell",
        "cross-sell",
        "pack",
        "combo",
        "premium",
        "añadir",
        "añade",
        "extra",
        "complemento",
    )
    medium = (
        "subir importe",
        "mejorar ticket",
        "mejorar cesta",
        "cesta media",
        "upgrade",
        "mejor margen",
    )

    score = 0.0
    if _contains_any(txt, strong):
        score += 0.11
    if _contains_any(txt, medium):
        score += 0.05

    # Si la métrica principal ya está claramente ligada a ticket,
    # damos una pequeña señal adicional.
    metric = str(getattr(card, "primary_metric_label", "")).lower()
    if "ticket" in metric or "importe medio" in metric:
        score += 0.04

    return min(0.16, score)



def _recurrence_signal(card: RecommendationCard) -> float:
    """
    Detecta recomendaciones de recurrencia/frecuencia/retención.
    Queremos que compitan mejor en el top 3 frente a quick wins tácticos.
    """
    txt = _safe_text_parts(card)
    strong = (
        "recurrencia",
        "retención",
        "repetición",
        "repeat",
        "recompra",
        "recompra",
        "volver",
        "fidel",
        "seguimiento",
        "recordatorio",
        "segunda visita",
        "próxima visita",
        "proxima visita",
        "reactivar",
    )
    medium = (
        "cliente habitual",
        "clientes habituales",
        "cliente recurrente",
        "frecuencia",
        "retener",
        "whatsapp",
        "agenda",
        "base de clientes",
    )

    score = 0.0
    if _contains_any(txt, strong):
        score += 0.10
    if _contains_any(txt, medium):
        score += 0.05
    return min(0.15, score)



def _strategic_signal(card: RecommendationCard) -> float:
    """
    Detecta acciones con carácter más estratégico/comercial estructural.
    Esto ayuda a que el top 3 no quede lleno solo de acciones tácticas rápidas.
    """
    txt = _safe_text_parts(card)
    strong = (
        "mix",
        "precio",
        "pricing",
        "margen",
        "estándar",
        "estandar",
        "proceso",
        "sistema",
        "posicionamiento",
        "estructura",
        "oferta",
        "paquete",
        "paquetes",
        "familia de servicios",
    )
    medium = (
        "argumentario",
        "rutina",
        "guion",
        "propuesta",
        "ordenar",
        "organizar",
        "priorizar",
        "visibilidad",
    )

    score = 0.0
    if _contains_any(txt, strong):
        score += 0.08
    if _contains_any(txt, medium):
        score += 0.04
    return min(0.12, score)



def _family_balance_bonus(card: RecommendationCard) -> float:
    """
    Bonus agregado ligero que favorece cartas con una señal clara de familia útil.

    No es diversidad real todavía; eso lo hace select_plan.
    Esto solo evita que el score bruto esté ciego a la intención comercial.
    """
    ticket = _ticket_signal(card)
    recurrence = _recurrence_signal(card)
    strategic = _strategic_signal(card)

    active_families = sum(x > 0 for x in (ticket, recurrence, strategic))

    # Una card que toca más de una familia útil puede ser muy valiosa.
    # Pero mantenemos el bonus pequeño para no meter ruido artificial.
    if active_families >= 2:
        return 0.04
    if active_families == 1:
        return 0.02
    return 0.0



def _impact_component(mid: float, max_mid: float) -> tuple[float, str]:
    """
    Componente de impacto comprimido.

    Problema anterior:
    - usar impacto lineal vs max_mid hace que una sola recomendación con mucho impacto
      se dispare demasiado.

    Solución:
    - compresión con raíz cuadrada.
    - sigue premiando el impacto, pero aplana distancias extremas.
    """
    if max_mid <= 0:
        return 0.0, "impact_norm=0.00 impact_comp=0.00"

    raw_norm = max(0.0, min(1.0, mid / max_mid))
    compressed_norm = sqrt(raw_norm)
    component = 0.34 * compressed_norm
    return component, f"impact_raw={raw_norm:.2f} impact_cmp={compressed_norm:.2f}"


# ============================================================
# History helpers
# ============================================================
def _iter_history_items(hist: dict) -> list[dict]:
    """Devuelve lista plana de items históricos. Soporta varias formas de guardado."""
    if not hist or not isinstance(hist, dict):
        return []

    out: list[dict] = []

    periods = hist.get("periods")
    if isinstance(periods, dict):
        for pk, pdata in periods.items():
            if isinstance(pdata, dict):
                items = pdata.get("items") or []
                if isinstance(items, list):
                    for it in items:
                        if isinstance(it, dict):
                            row = dict(it)
                            row.setdefault("period_key", pk)
                            out.append(row)

    recs = hist.get("recommendations")
    if isinstance(recs, list):
        for r in recs:
            if isinstance(r, dict):
                items = r.get("items") or []
                if isinstance(items, list):
                    for it in items:
                        if isinstance(it, dict):
                            row = dict(it)
                            row.setdefault("period_key", r.get("period"))
                            out.append(row)

    for k, v in hist.items():
        if k in ("periods", "recommendations"):
            continue
        if isinstance(v, dict) and isinstance(v.get("items"), list):
            for it in v["items"]:
                if isinstance(it, dict):
                    row = dict(it)
                    row.setdefault("period_key", k)
                    out.append(row)

    return out



def _parse_iso(dt_str: str) -> Optional[datetime]:
    try:
        return datetime.fromisoformat(str(dt_str))
    except Exception:
        return None



def _last_seen_item(items: list[dict], insight_id: str) -> Optional[dict]:
    seen = [it for it in items if it.get("insight_id") == insight_id]
    if not seen:
        return None

    def key(it: dict):
        dt = _parse_iso(it.get("updated_at") or it.get("generated_at") or "")
        return dt or datetime(1970, 1, 1)

    seen.sort(key=key)
    return seen[-1]



def _history_adjustment(
    hist: dict | None,
    insight_id: str,
    *,
    recency_days: int = 45,
) -> tuple[float, str]:
    """
    Ajuste por historial reciente:
    - penaliza repetición reciente
    - recompensa improved/positive
    - castiga not_improved/negative
    - castiga skip

    Se ha suavizado un poco para no aplastar demasiado cartas buenas por una sola señal pasada.
    """
    if not hist:
        return 0.0, "sin_historial"

    items = _iter_history_items(hist)
    if not items:
        return 0.0, "historial_vacio"

    last = _last_seen_item(items, insight_id)
    if not last:
        return 0.0, "no_recomendada_antes"

    status = str(last.get("status") or "").strip().lower()
    outcome = str(last.get("outcome") or "").strip().lower()

    last_dt = _parse_iso(last.get("updated_at") or "") or datetime(1970, 1, 1)
    is_recent = last_dt >= (datetime.now() - timedelta(days=int(recency_days)))

    adj = 0.0
    reasons: list[str] = []

    if is_recent and status in ("planned", "done", "active"):
        adj -= 0.10
        reasons.append("repeticion_reciente")

    if is_recent and status == "already_doing":
        adj -= 0.05
        reasons.append("ya_lo_hacen")

    if outcome in ("improved", "positivo", "positive"):
        adj += 0.14
        reasons.append("resultado_positivo")
    elif outcome in ("not_improved", "negative", "negativo"):
        adj -= 0.14
        reasons.append("resultado_negativo")
    elif outcome in ("inconclusive", "inconcluso"):
        adj -= 0.04
        reasons.append("resultado_inconcluso")
    elif outcome in ("neutral", "neutro"):
        adj -= 0.02
        reasons.append("resultado_neutro")

    if status in ("skipped", "saltado", "saltar"):
        adj -= 0.06
        reasons.append("saltada")

    # Acotamos para que el histórico ayude, pero no secuestre el ranking entero.
    adj = max(-0.18, min(0.18, adj))

    if not reasons:
        reasons.append("sin_senal_fuerte")

    return adj, f"{'+' if adj >= 0 else ''}{adj:.2f}|" + ",".join(reasons)


# ============================================================
# Score principal
# ============================================================
def score_recommendations(
    candidates: Iterable[RecommendationCard],
    history_store: HistoryStore | None = None,
    *,
    recency_days: int = 45,
) -> list[ScoredRecommendation]:
    """
    Ranking comercial recalibrado.

    Componentes:
    - impacto comprimido (evita dominancia excesiva)
    - prioridad editorial
    - confianza
    - quick win
    - bonus por familia/intención:
        * ticket
        * recurrencia
        * estratégico
        * equilibrio de familias
    - penalización por esfuerzo
    - penalización por tiempo
    - ajuste por histórico
    """
    cards = list(candidates)
    if not cards:
        return []

    hist = history_store.load() if history_store is not None else {}

    mids = [_impact_mid(c) for c in cards]
    max_mid = max(mids) if mids else 1.0
    if max_mid <= 0:
        max_mid = 1.0

    scored: list[ScoredRecommendation] = []

    for card in cards:
        mid = _impact_mid(card)
        impact_component, impact_reason = _impact_component(mid, max_mid)

        editorial_component = _priority_editorial_bonus(card.priority_weight)
        confidence_component = _confidence_bonus(card.confidence_label)
        quick_win_component = _quick_win_bonus(card)

        # Nuevas señales de negocio que corrigen el sesgo excesivo hacia una sola clase.
        ticket_component = _ticket_signal(card)
        recurrence_component = _recurrence_signal(card)
        strategic_component = _strategic_signal(card)
        balance_component = _family_balance_bonus(card)

        effort_component = _effort_penalty(card.effort)
        time_component = _time_penalty(card.time_to_apply_min)

        hist_adj, hist_reason = _history_adjustment(
            hist,
            card.insight_id,
            recency_days=recency_days,
        )

        score = (
            impact_component
            + editorial_component
            + confidence_component
            + quick_win_component
            + ticket_component
            + recurrence_component
            + strategic_component
            + balance_component
            - effort_component
            - time_component
            + hist_adj
        )

        reason = (
            f"{impact_reason} "
            f"editorial={editorial_component:.2f} "
            f"confidence={confidence_component:.2f} "
            f"quickwin={quick_win_component:.2f} "
            f"ticket={ticket_component:.2f} "
            f"recurrence={recurrence_component:.2f} "
            f"strategic={strategic_component:.2f} "
            f"balance={balance_component:.2f} "
            f"effort=-{effort_component:.2f} "
            f"time=-{time_component:.2f} "
            f"history={hist_reason}"
        )

        scored.append(
            ScoredRecommendation(
                card=card,
                score=round(float(score), 4),
                reason=reason,
            )
        )

    scored.sort(key=lambda x: x.score, reverse=True)
    return scored


# ============================================================
# Selección final para Owner/Admin
# ============================================================
def _same_family(a: RecommendationCard, b: RecommendationCard) -> bool:
    """
    Evita meter recomendaciones demasiado parecidas.

    Criterio:
    - mismo goal y al menos 1 tag compartido
    - o 2+ tags compartidos

    Mantengo esta lógica porque ya era razonable.
    No conviene endurecerla mucho más o empezarás a forzar mezclas artificiales.
    """
    shared_tags = set(a.tags).intersection(set(b.tags))

    if a.goal == b.goal and len(shared_tags) >= 1:
        return True

    return len(shared_tags) >= 2



def _family_bucket(card: RecommendationCard) -> str:
    """
    Bucket comercial aproximado para mejorar la diversidad sin forzar rarezas.

    Orden de prioridad:
    - si tiene señal clara de ticket -> ticket
    - si tiene señal clara de recurrencia -> recurrence
    - si tiene señal clara de estratégico -> strategic
    - si no, misc

    Este bucket NO reemplaza _same_family. Solo ayuda a desempatar y equilibrar.
    """
    if _ticket_signal(card) >= 0.10:
        return "ticket"
    if _recurrence_signal(card) >= 0.09:
        return "recurrence"
    if _strategic_signal(card) >= 0.08:
        return "strategic"
    return "misc"



def select_plan(
    scored: list[ScoredRecommendation],
    k: int = 3,
) -> list[ScoredRecommendation]:
    """
    Selección Top-k con diversidad pragmática.

    Objetivo real:
    - priorizar score;
    - evitar clones;
    - favorecer mezcla razonable de buckets útiles;
    - NO forzar diversidad absurda si no existe oferta suficiente.

    Estrategia:
    1) coger el mejor score;
    2) en las siguientes posiciones, priorizar candidatos no clon y de bucket aún no cubierto;
    3) si no hay, rellenar con los mejores restantes.
    """
    k = max(0, int(k))
    if k == 0:
        return []
    if not scored:
        return []

    picked: list[ScoredRecommendation] = []
    used_buckets: set[str] = set()

    # Primera: la mejor por score, sin discusión.
    first = scored[0]
    picked.append(first)
    used_buckets.add(_family_bucket(first.card))

    if len(picked) >= k:
        return picked

    # Segunda y siguientes: intentamos variedad razonable si el score acompaña.
    while len(picked) < k:
        remaining = [x for x in scored if x not in picked]
        if not remaining:
            break

        chosen: ScoredRecommendation | None = None

        # Paso 1: mejor candidato que no sea clon y aporte bucket nuevo.
        for item in remaining:
            is_clone = any(_same_family(item.card, prev.card) for prev in picked)
            if is_clone:
                continue

            bucket = _family_bucket(item.card)
            if bucket not in used_buckets:
                chosen = item
                break

        # Paso 2: si no existe bucket nuevo, buscamos el mejor no-clon.
        if chosen is None:
            for item in remaining:
                is_clone = any(_same_family(item.card, prev.card) for prev in picked)
                if not is_clone:
                    chosen = item
                    break

        # Paso 3: si todo es parecido, rellena por score. Mejor eso que forzar rarezas.
        if chosen is None:
            chosen = remaining[0]

        picked.append(chosen)
        used_buckets.add(_family_bucket(chosen.card))

    return picked