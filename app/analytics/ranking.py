# app/analytics/ranking.py
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

from .insights import Insight
from .history_store import HistoryStore


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
# History helpers
# ----------------------------
def _iter_history_items(hist: dict) -> list[dict]:
    """
    Devuelve lista plana de items históricos.
    Soporta:
      A) hist["periods"][period_key]["items"]
      B) hist["recommendations"][...]["items"] (v1)
      C) hist[period_key]["items"] (legacy raro)
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
                            it2 = dict(it)
                            it2.setdefault("period_key", pk)
                            out.append(it2)

    recs = hist.get("recommendations")
    if isinstance(recs, list):
        for r in recs:
            if isinstance(r, dict):
                items = r.get("items") or []
                if isinstance(items, list):
                    for it in items:
                        if isinstance(it, dict):
                            it2 = dict(it)
                            it2.setdefault("period_key", r.get("period"))
                            out.append(it2)

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


def _parse_iso(dt_str: str) -> Optional[datetime]:
    """Parse seguro del ISO que tú guardas (YYYY-MM-DDTHH:MM:SS)."""
    try:
        return datetime.fromisoformat(str(dt_str))
    except Exception:
        return None


def _last_seen_item(items: list[dict], insight_id: str) -> Optional[dict]:
    """Devuelve el item más reciente por updated_at."""
    seen = [it for it in items if it.get("insight_id") == insight_id]
    if not seen:
        return None

    def key(it: dict):
        dt = _parse_iso(it.get("updated_at") or it.get("generated_at") or "")
        # si no parsea, lo mandamos al pasado
        return dt or datetime(1970, 1, 1)

    seen.sort(key=key)
    return seen[-1]


def _feedback_adjustment(
    hist: dict | None,
    insight_id: str,
    *,
    recency_days: int = 45,
) -> tuple[float, str]:
    """
    Ajuste por historial (con recencia):
      - penaliza repetición reciente si estaba planned/done
      - recompensa improved, castiga not_improved
      - castiga skipped
    """
    if not hist:
        return 0.0, "sin_historial"

    items = _iter_history_items(hist)
    if not items:
        return 0.0, "historial_vacio"

    last = _last_seen_item(items, insight_id)
    if not last:
        return 0.0, "no_recomendado_antes"

    status = (last.get("status") or "").strip().lower()
    outcome = (last.get("outcome") or "").strip().lower()

    last_dt = _parse_iso(last.get("updated_at") or "") or datetime(1970, 1, 1)
    is_recent = last_dt >= (datetime.now() - timedelta(days=int(recency_days)))

    # Penaliza repetición SOLO si es reciente
    rep_penalty = (-0.20 if is_recent and status in ("planned", "done") else 0.0)

    if outcome == "improved":
        return rep_penalty + 0.25, f"feedback:improved(recent={is_recent})"
    if outcome == "not_improved":
        return rep_penalty - 0.25, f"feedback:not_improved(recent={is_recent})"
    if status == "skipped":
        return rep_penalty - 0.10, f"feedback:skipped(recent={is_recent})"
    return rep_penalty, f"feedback:unknown(recent={is_recent})"


def rank_insights(
    candidates: list[Insight],
    history_store: HistoryStore,
    period_key: str,
    *,
    recency_days: int = 45,
) -> list[ScoredInsight]:
    """
    Ranking transparente:
      - impacto normalizado (mid) solo en revenue
      - penalizaciones: esfuerzo + tiempo
      - ajuste por historial (recencia/outcome)
    """
    hist = history_store.load()

    revenue_mids = [_impact_mid(x) for x in candidates if getattr(x, "impact_type", "") == "revenue"]
    max_mid = max(revenue_mids) if revenue_mids else 1.0

    scored: list[ScoredInsight] = []
    for ins in candidates:
        mid = _impact_mid(ins)
        impact_norm = (mid / max_mid) if max_mid else 0.0

        base = 0.70 * impact_norm if ins.impact_type == "revenue" else 0.10
        pen = _effort_penalty(ins.effort) + _time_penalty(ins.time_to_apply_min)
        adj, adj_reason = _feedback_adjustment(hist, ins.insight_id, recency_days=recency_days)

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


def select_plan(
    scored: list[ScoredInsight],
    k: int = 3,
    *,
    require_revenue_for_owner: bool = True,
) -> list[ScoredInsight]:
    """
    Selección Top-k con diversidad por kpi_target.
    - Evita 3 acciones con el mismo KPI.
    - Owner: prioriza revenue (si require_revenue_for_owner=True).
    """
    k = max(0, int(k))
    if k == 0:
        return []

    pool = scored[:]

    # Owner: prioriza revenue
    if require_revenue_for_owner:
        revenue = [s for s in pool if getattr(s.insight, "impact_type", "") == "revenue"]
        nonrev = [s for s in pool if getattr(s.insight, "impact_type", "") != "revenue"]
        pool = revenue + nonrev  # mantiene orden relativo

    picked: list[ScoredInsight] = []
    used_kpis: set[str] = set()

    for s in pool:
        kpi = str(getattr(s.insight, "kpi_target", "") or "")
        if kpi and kpi in used_kpis:
            continue
        picked.append(s)
        if kpi:
            used_kpis.add(kpi)
        if len(picked) >= k:
            break

    # Si no llegamos a k por diversidad, rellenamos sin restricción
    if len(picked) < k:
        for s in pool:
            if s in picked:
                continue
            picked.append(s)
            if len(picked) >= k:
                break

    return picked