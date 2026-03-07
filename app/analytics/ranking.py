from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Iterable, Optional

from .history_store import HistoryStore
from .recommendation_library import RecommendationCard


# ============================================================
# Modelo principal
# ============================================================
@dataclass(frozen=True)
class ScoredRecommendation:
    """
    Recomendación puntuada y lista para selección final.
    """

    card: RecommendationCard
    score: float
    reason: str


# ============================================================
# Helpers de score base
# ============================================================
def _impact_mid(card: RecommendationCard) -> float:
    low, high = card.estimated_impact_eur
    return 0.5 * (float(low) + float(high))


def _effort_penalty(effort: str) -> float:
    """
    Penaliza acciones con más fricción operativa.
    """
    e = (effort or "").strip().lower()
    return {
        "bajo": 0.00,
        "media": 0.12,
        "medio": 0.12,
        "alta": 0.28,
    }.get(e, 0.15)


def _time_penalty(minutes: int) -> float:
    """
    Penalización suave por tiempo estimado de aplicación.
    """
    try:
        m = int(minutes)
    except Exception:
        m = 30
    return min(0.25, max(0.0, m / 240.0))


def _confidence_bonus(confidence_label: str) -> float:
    c = (confidence_label or "").strip().lower()
    return {
        "alta": 0.10,
        "media": 0.04,
        "baja": -0.04,
    }.get(c, 0.0)


def _priority_editorial_bonus(priority_weight: float) -> float:
    """
    Bonus suave por prioridad editorial/base del catálogo.
    """
    return max(0.0, min(0.20, (float(priority_weight) - 0.70)))


def _quick_win_bonus(card: RecommendationCard) -> float:
    """
    Bonus ligero para acciones fáciles de vender y aplicar.
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
        bonus += 0.01

    if (card.confidence_label or "").strip().lower() == "alta":
        bonus += 0.02

    return min(0.08, bonus)


# ============================================================
# History helpers
# ============================================================
def _iter_history_items(hist: dict) -> list[dict]:
    """
    Devuelve lista plana de items históricos.
    Soporta varias formas de guardado.
    """
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

    # Penalización ligera por repetición reciente
    if is_recent and status in ("planned", "done", "active"):
        adj -= 0.12
        reasons.append("repeticion_reciente")

    # already_doing penaliza menos: indica que la idea no es nueva,
    # pero no necesariamente que sea mala o irrelevante
    if is_recent and status == "already_doing":
        adj -= 0.06
        reasons.append("ya_lo_hacen")

    if outcome in ("improved", "positivo", "positive"):
        adj += 0.16
        reasons.append("resultado_positivo")
    elif outcome in ("not_improved", "negative", "negativo"):
        adj -= 0.18
        reasons.append("resultado_negativo")
    elif outcome in ("inconclusive", "inconcluso"):
        adj -= 0.05
        reasons.append("resultado_inconcluso")
    elif outcome in ("neutral", "neutro"):
        adj -= 0.02
        reasons.append("resultado_neutro")

    if status in ("skipped", "saltado", "saltar"):
        adj -= 0.08
        reasons.append("saltada")

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
    Scoring comercial orientado a RecommendationCard.

    Componentes:
    - impacto normalizado
    - bonus por prioridad editorial
    - bonus por confianza
    - bonus quick win
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
        impact_norm = (mid / max_mid) if max_mid else 0.0

        impact_component = 0.45 * impact_norm
        editorial_component = _priority_editorial_bonus(card.priority_weight)
        confidence_component = _confidence_bonus(card.confidence_label)
        quick_win_component = _quick_win_bonus(card)
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
            - effort_component
            - time_component
            + hist_adj
        )

        reason = (
            f"impact={impact_norm:.2f} "
            f"editorial={editorial_component:.2f} "
            f"confidence={confidence_component:.2f} "
            f"quickwin={quick_win_component:.2f} "
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
    """
    shared_tags = set(a.tags).intersection(set(b.tags))

    if a.goal == b.goal and len(shared_tags) >= 1:
        return True

    return len(shared_tags) >= 2


def select_plan(
    scored: list[ScoredRecommendation],
    k: int = 3,
) -> list[ScoredRecommendation]:
    """
    Selección Top-k con diversidad pragmática:
    - prioriza score
    - evita clones temáticos
    - rellena si hace falta
    """
    k = max(0, int(k))
    if k == 0:
        return []

    picked: list[ScoredRecommendation] = []

    for item in scored:
        if not picked:
            picked.append(item)
            if len(picked) >= k:
                return picked
            continue

        is_too_similar = any(_same_family(item.card, prev.card) for prev in picked)
        if is_too_similar:
            continue

        picked.append(item)
        if len(picked) >= k:
            return picked

    if len(picked) < k:
        for item in scored:
            if item in picked:
                continue
            picked.append(item)
            if len(picked) >= k:
                break

    return picked