from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from analytics.insights import Insight
from analytics.history_store import HistoryStore


@dataclass(frozen=True)
class ScoredInsight:
    insight: Insight
    score: float
    reason: str


def _impact_mid(ins: Insight) -> float:
    low, high = ins.estimated_impact_eur
    return 0.5 * (float(low) + float(high))


def _effort_penalty(effort: str) -> float:
    """Penaliza acciones con más esfuerzo para priorizar quick wins."""
    e = (effort or "").strip().lower()
    return {"baja": 0.0, "media": 0.15, "alta": 0.35}.get(e, 0.2)


def _time_penalty(minutes: int) -> float:
    """Penalización suave por tiempo (0..0.30)."""
    try:
        m = int(minutes)
    except Exception:
        m = 60
    return min(0.30, max(0.0, m / 240.0))


# ----------------------------
# History helpers (compatibles con varias estructuras)
# ----------------------------
def _iter_history_items(hist: dict) -> list[dict]:
    """
    Devuelve una lista plana de items históricos [{insight_id, status, outcome, updated_at, period_key?}, ...]
    Soporta estructuras:
      A) hist["periods"][period_key]["items"]
      B) hist["recommendations"][...]["items"]  (legacy)
      C) hist[period_key]["items"] (por si guardas directo)
    """
    if not hist or not isinstance(hist, dict):
        return []

    out: list[dict] = []

    # A) periods dict
    periods = hist.get("periods")
    if isinstance(periods, dict):
        for pk, pdata in periods.items():
            if isinstance(pdata, dict):
                items = pdata.get("items") or []
                if isinstance(items, list):
                    for it in items:
                        if isinstance(it, dict):
                            it2 = dict(it)
                            it2.setdefault("period_key", pk)
                            out.append(it2)

    # B) legacy recommendations list
    recs = hist.get("recommendations")
    if isinstance(recs, list):
        for r in recs:
            if isinstance(r, dict):
                items = r.get("items") or []
                if isinstance(items, list):
                    for it in items:
                        if isinstance(it, dict):
                            out.append(it)

    # C) direct dict keyed by period_key
    # (si alguien guardó periodos al nivel raíz)
    for k, v in hist.items():
        if k in ("periods", "recommendations"):
            continue
        if isinstance(v, dict) and isinstance(v.get("items"), list):
            for it in v["items"]:
                if isinstance(it, dict):
                    it2 = dict(it)
                    it2.setdefault("period_key", k)
                    out.append(it2)

    return out


def _feedback_adjustment(hist: dict | None, insight_id: str) -> tuple[float, str]:
    """
    Ajuste por historial:
      - penaliza repetición si aparece recientemente (planned/done)
      - recompensa improved, castiga not_improved
      - castiga skipped
    """
    if not hist:
        return 0.0, "sin_historial"

    items = _iter_history_items(hist)
    if not items:
        return 0.0, "historial_vacio"

    seen = [it for it in items if it.get("insight_id") == insight_id]
    if not seen:
        return 0.0, "no_recomendado_antes"

    last = seen[-1]
    status = (last.get("status") or "").strip().lower()
    outcome = (last.get("outcome") or "").strip().lower()

    # Penaliza repetición si estaba planned/done recientemente
    rep_penalty = -0.20 if status in ("planned", "done") else 0.0

    if outcome == "improved":
        return rep_penalty + 0.25, "feedback:improved"
    if outcome == "not_improved":
        return rep_penalty - 0.25, "feedback:not_improved"
    if status == "skipped":
        return rep_penalty - 0.10, "feedback:skipped"
    return rep_penalty, "feedback:unknown"


def rank_insights(
    candidates: list[Insight],
    history_store: HistoryStore,
    period_key: str,
) -> list[ScoredInsight]:
    """
    Ranking transparente:
      - impacto normalizado (mid)
      - penalizaciones: esfuerzo + tiempo
      - ajuste por historial (repetición/outcome)
    """
    hist = history_store.load()

    # Normaliza impacto usando solo insights de revenue para evitar que "risk/time" rompan escalas
    revenue_mids = [_impact_mid(x) for x in candidates if getattr(x, "impact_type", "") == "revenue"]
    max_mid = max(revenue_mids) if revenue_mids else 1.0

    scored: list[ScoredInsight] = []
    for ins in candidates:
        mid = _impact_mid(ins)
        impact_norm = (mid / max_mid) if max_mid else 0.0

        # Si no es revenue, no lo mates: dale una base baja para que aparezca si es útil
        base = 0.70 * impact_norm if ins.impact_type == "revenue" else 0.10

        pen = _effort_penalty(ins.effort) + _time_penalty(ins.time_to_apply_min)
        adj, adj_reason = _feedback_adjustment(hist, ins.insight_id)

        score = base - pen + adj
        reason = f"impact={impact_norm:.2f} base={base:.2f} pen={pen:.2f} adj={adj:.2f} ({adj_reason})"

        scored.append(
            ScoredInsight(
                insight=ins,
                score=round(float(score), 4),
                reason=reason,
            )
        )

    scored.sort(key=lambda x: x.score, reverse=True)
    return scored


def select_plan(scored: list[ScoredInsight], k: int = 3) -> list[ScoredInsight]:
    """Top-k."""
    return scored[: max(0, int(k))]