from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import pandas as pd

from .features import CoreFeatures
from .impact import estimate_extra_revenue_for_worst_day


ImpactType = Literal["revenue", "risk", "time"]
Effort = Literal["baja", "media", "alta"]
KPI = Literal["ticket_medio", "ingresos", "mix_servicios", "riesgo_dependencia", "frecuencia"]


@dataclass(frozen=True)
class Insight:
    """
    Insight accionable con metadata vendible (no técnica).

    Campos clave para retención:
      - impacto estimado (rango €)
      - esfuerzo / tiempo
      - KPI objetivo (para seguimiento)
      - evidencia (por qué sale ahora)
      - acción concreta (qué hacer)
    """
    id: str
    title: str
    evidence: str
    action_hint: str

    impact_type: ImpactType
    kpi_target: KPI

    estimated_impact_eur: Tuple[float, float]  # (low, high)
    effort: Effort
    time_to_apply_min: int

    confidence: float


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _eur_range(low: float, high: float) -> Tuple[float, float]:
    low = float(max(0.0, low))
    high = float(max(low, high))
    return (low, high)


def _top_product_stats(df: pd.DataFrame) -> dict:
    """
    Stats del producto top para estimaciones simples de bundles/upsell.
    """
    if df.empty:
        return {"top_name": "-", "top_rev": 0.0, "top_rev_share": 0.0, "top_tickets": 0}

    total_rev = float(df["revenue"].sum()) or 0.0
    g = (
        df.groupby("producto", as_index=False)["revenue"]
        .sum()
        .sort_values("revenue", ascending=False)
    )
    top_row = g.iloc[0]
    top_name = str(top_row["producto"])
    top_rev = float(top_row["revenue"])
    top_rev_share = (top_rev / total_rev * 100.0) if total_rev else 0.0

    top_tickets = int((df["producto"] == top_name).sum())

    return {
        "top_name": top_name,
        "top_rev": top_rev,
        "top_rev_share": float(top_rev_share),
        "top_tickets": top_tickets,
        "total_rev": total_rev,
        "total_tickets": int(len(df)),
    }


def generate_insights(
    df_f: pd.DataFrame,
    core: CoreFeatures,
    business_type: str,
    uplift_eur_per_ticket: float,
    horizon: str,
) -> list[Insight]:
    """
    Genera insights con:
      - impacto estimado (rango)
      - esfuerzo/tiempo
      - KPI objetivo
    Sin "frases fijas": cada insight lleva evidencia numérica.
    """
    out: list[Insight] = []
    top = _top_product_stats(df_f)

    # 1) Subir ticket en el peor día (impacto real usando el estimador corregido)
    #    Rango: low = 60% del escenario, high = 120% (para no vender humo).
    base = estimate_extra_revenue_for_worst_day(
        df_filtered=df_f,
        worst_day_name=core.peor_dia_nombre,
        uplift_eur_per_ticket=float(uplift_eur_per_ticket),
        horizon=horizon,
    )
    out.append(
        Insight(
            id="uplift_worst_day_ticket",
            title=f"Subir ticket en {core.peor_dia_nombre}",
            evidence=(
                f"Gap {core.mejor_dia_nombre} vs {core.peor_dia_nombre}: "
                f"{core.gap_mejor_peor:,.2f}€ en el rango."
            ).replace(",", "."),
            action_hint="Implementa 2 extras (A/B) y ofrécelos solo el día flojo durante 2 semanas.",
            impact_type="revenue",
            kpi_target="ticket_medio",
            estimated_impact_eur=_eur_range(base * 0.60, base * 1.20),
            effort="baja",
            time_to_apply_min=20,
            confidence=0.85,
        )
    )

    # 2) Ancla: bundle sobre el servicio top (impacto: pequeña subida en tickets del top)
    # Heurística conservadora:
    #   - Afecta entre 10% y 25% de tickets del servicio top
    #   - uplift medio entre 1€ y 2.5€
    # => impacto = top_tickets * adopción * uplift
    if top["top_tickets"] > 0:
        adopt_low, adopt_high = 0.10, 0.25
        uplift_low, uplift_high = 1.0, 2.5
        low = top["top_tickets"] * adopt_low * uplift_low
        high = top["top_tickets"] * adopt_high * uplift_high

        out.append(
            Insight(
                id="service_anchor_top",
                title=f"Bundle del servicio top ({top['top_name']})",
                evidence=(
                    f"El servicio top concentra {top['top_rev_share']:.1f}% del ingreso en el rango."
                ),
                action_hint="Crea 1 bundle fijo: servicio top + 1 extra (2 opciones A/B) con guion de 1 frase.",
                impact_type="revenue",
                kpi_target="ticket_medio",
                estimated_impact_eur=_eur_range(low, high),
                effort="media",
                time_to_apply_min=60,
                confidence=0.70,
            )
        )

    # 3) Riesgo dependencia Top 3 (acción, no comentario)
    if core.pct_top3_ingresos >= 55:
        out.append(
            Insight(
                id="top3_concentration_high",
                title="Reducir dependencia del Top 3",
                evidence=f"Top 3 concentra {core.pct_top3_ingresos:.1f}% de ingresos.",
                action_hint="Empuja el #4/#5 con 1 pack fijo y mide adopción semanal (sin descuentos).",
                impact_type="risk",
                kpi_target="riesgo_dependencia",
                estimated_impact_eur=_eur_range(0.0, 0.0),
                effort="media",
                time_to_apply_min=45,
                confidence=0.80,
            )
        )
    else:
        # Aunque esté “controlada”, lo convertimos en acción útil (rotación de extras)
        out.append(
            Insight(
                id="top3_concentration_ok_action",
                title="Rotación semanal de extras (para no estancarte)",
                evidence=f"Top 3 concentra {core.pct_top3_ingresos:.1f}% (razonable).",
                action_hint="Rota 1 extra recomendado por semana y registra tasa de aceptación (sí/no).",
                impact_type="revenue",
                kpi_target="mix_servicios",
                estimated_impact_eur=_eur_range(30.0, 180.0),  # conservador y pequeño
                effort="baja",
                time_to_apply_min=15,
                confidence=0.60,
            )
        )

    # 4) Muchos tickets con 1 unidad (proxy: upsell)
    if core.pct_tickets_unidad_1 >= 50:
        # Impacto simple: entre 8% y 18% de tickets totales * uplift 1€..2€
        adopt_low, adopt_high = 0.08, 0.18
        uplift_low, uplift_high = 1.0, 2.0
        low = core.tickets_total * adopt_low * uplift_low
        high = core.tickets_total * adopt_high * uplift_high

        out.append(
            Insight(
                id="upsell_low_items",
                title="Subir ítems por ticket (muchos tickets de 1 unidad)",
                evidence=f"{core.pct_tickets_unidad_1:.1f}% de tickets parecen ser de 1 unidad.",
                action_hint="Añade 1 complemento natural (2 opciones) en el cobro. No más de 1 pregunta.",
                impact_type="revenue",
                kpi_target="ticket_medio",
                estimated_impact_eur=_eur_range(low, high),
                effort="baja",
                time_to_apply_min=25,
                confidence=0.70,
            )
        )

    # Playbook específico por industria (v1)
    if business_type == "servicios":
        out.append(
            Insight(
                id="service_rebooking_simple",
                title="Rebooking simple (subir frecuencia sin descuento)",
                evidence="Servicios suelen ganar mucho más por frecuencia que por precio.",
                action_hint="Al final: '¿Te reservo ya para X semanas?' + 2 opciones (fecha/hora).",
                impact_type="revenue",
                kpi_target="frecuencia",
                estimated_impact_eur=_eur_range(50.0, 250.0),
                effort="media",
                time_to_apply_min=40,
                confidence=0.55,
            )
        )

    return out