# app/analytics/impact.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ImpactEstimate:
    low_eur: float
    high_eur: float
    basis_eur: float
    horizon_multiplier: int
    method: str
    notes: str


def _clamp_float(x: float, *, lo: float | None = None, hi: float | None = None) -> float:
    try:
        v = float(x)
    except Exception:
        v = 0.0
    if lo is not None:
        v = max(lo, v)
    if hi is not None:
        v = min(hi, v)
    return v


def _clamp_int(x: int, *, lo: int | None = None, hi: int | None = None) -> int:
    try:
        v = int(x)
    except Exception:
        v = 0
    if lo is not None:
        v = max(lo, v)
    if hi is not None:
        v = min(hi, v)
    return v


def estimate_ticket_uplift_impact(
    *,
    tickets_in_range_for_target_day: int,
    uplift_eur_per_ticket: float,
    horizon_multiplier: int,
    low_factor: float = 0.60,
    high_factor: float = 1.20,
    granularity: Optional[str] = None,          # "ticket" | "row" | None
    conservative_factor_row: float = 0.50,      # solo si granularity=="row"
) -> ImpactEstimate:
    """
    Estima impacto de elevar el ticket/importe medio en el peor día.

    Convención:
      - basis_eur = tickets_en_rango_del_peor_dia * uplift
      - impact = basis_eur * horizon_multiplier
      - low/high = impact * low_factor/high_factor

    Guardrail:
      - Si granularity == "row" (sin ticket_id), aplica conservative_factor_row a los tickets
        para evitar inflar (líneas != tickets).
    """
    t_raw = _clamp_int(tickets_in_range_for_target_day, lo=0)
    u = _clamp_float(uplift_eur_per_ticket, lo=0.0)
    hm = _clamp_int(horizon_multiplier, lo=1)

    # Factors sane defaults
    lf = _clamp_float(low_factor, lo=0.0, hi=2.0)
    hf = _clamp_float(high_factor, lo=0.0, hi=3.0)
    if hf < lf:
        hf = lf

    # Conservative adjustment if row-level
    notes_extra = ""
    t_used = t_raw
    if (granularity or "").strip().lower() == "row":
        cf = _clamp_float(conservative_factor_row, lo=0.1, hi=1.0)
        t_used = max(1, int(round(t_raw * cf))) if t_raw > 0 else 0
        notes_extra = f" | granularity=row => conservative_factor={cf:.2f} (tickets ajustados {t_raw}->{t_used})"

    basis = float(t_used * u)
    impact = basis * float(hm)

    low = _clamp_float(impact * lf, lo=0.0)
    high = _clamp_float(impact * hf, lo=0.0)
    if high < low:
        high = low

    method = "range_multiplier"
    if (granularity or "").strip().lower() == "row":
        method = "range_multiplier_conservative_row"

    return ImpactEstimate(
        low_eur=low,
        high_eur=high,
        basis_eur=basis,
        horizon_multiplier=hm,
        method=method,
        notes="Base EN EL RANGO; escalado por multiplicador del rango (1×/3×/12×)."
        + notes_extra,
    )