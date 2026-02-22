from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ImpactEstimate:
    low_eur: float
    high_eur: float
    basis_eur: float
    horizon_multiplier: int
    method: str
    notes: str


def _clamp_nonneg(x: float) -> float:
    try:
        x = float(x)
    except Exception:
        x = 0.0
    return max(0.0, x)


def estimate_ticket_uplift_impact(
    *,
    tickets_in_range_for_target_day: int,
    uplift_eur_per_ticket: float,
    horizon_multiplier: int,
    low_factor: float = 0.60,
    high_factor: float = 1.20,
) -> ImpactEstimate:
    t = int(max(0, tickets_in_range_for_target_day))
    u = _clamp_nonneg(uplift_eur_per_ticket)
    hm = int(max(1, horizon_multiplier))

    basis = float(t * u)
    impact = basis * hm

    low = _clamp_nonneg(impact * float(low_factor))
    high = _clamp_nonneg(impact * float(high_factor))
    if high < low:
        high = low

    return ImpactEstimate(
        low_eur=low,
        high_eur=high,
        basis_eur=basis,
        horizon_multiplier=hm,
        method="range_multiplier",
        notes="Base EN EL RANGO; escalado por multiplicador del rango (1×/3×/12×).",
    )