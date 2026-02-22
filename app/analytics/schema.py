from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class ImpactEstimate:
    """Impacto estimado en € (conservador)."""
    low_eur: float
    high_eur: float
    basis_eur: float          # base en rango (sin multiplicador)
    horizon_multiplier: int   # 1/3/12
    method: str               # "range_multiplier"
    notes: str                # trazabilidad breve


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
    """Impacto por subir ticket en un día objetivo (peor día), coherente con vNext.

    Base (rango):
      basis_eur = tickets_en_rango * uplift

    Horizonte:
      impact = basis_eur * horizon_multiplier

    Rango conservador (vendible):
      low = impact * low_factor
      high = impact * high_factor

    Esto evita inflar por calendarios y mantiene trazabilidad.
    """
    t = int(max(0, tickets_in_range_for_target_day))
    u = _clamp_nonneg(uplift_eur_per_ticket)
    hm = int(max(1, horizon_multiplier))

    basis = float(t * u)
    impact = basis * hm

    low = _clamp_nonneg(impact * float(low_factor))
    high = _clamp_nonneg(impact * float(high_factor))

    # Asegura orden
    if high < low:
        high = low

    return ImpactEstimate(
        low_eur=low,
        high_eur=high,
        basis_eur=basis,
        horizon_multiplier=hm,
        method="range_multiplier",
        notes="Base calculada EN EL RANGO; escalado por multiplicador del rango (1×/3×/12×).",
    )