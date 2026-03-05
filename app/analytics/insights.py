# app/analytics/insights.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Tuple, Optional
import math

import pandas as pd

from .impact import estimate_ticket_uplift_impact
from .kpis import eur
from .features import CoreFeatures, compute_core_features


ImpactType = Literal["revenue", "risk", "time"]
Effort = Literal["baja", "media", "alta"]
KPI = Literal["ticket_medio", "ingresos", "mix_servicios", "riesgo_dependencia", "frecuencia"]


_PLACEHOLDER_PRODUCTS = {"-", "SIN_PRODUCTO", "sin_producto", "nan", "none", ""}


def _eur_range(low: float, high: float) -> Tuple[float, float]:
    """Normaliza un rango € (no negativo y high >= low)."""
    low = float(max(0.0, low))
    high = float(max(low, high))
    return (low, high)


def _clean_product(df: pd.DataFrame) -> pd.DataFrame:
    """Excluye placeholders de producto para cálculos tipo Top/Concentración."""
    if df is None or df.empty or "producto" not in df.columns:
        return df
    p = df["producto"].astype(str).str.strip().str.lower()
    mask = ~p.isin({x.lower() for x in _PLACEHOLDER_PRODUCTS})
    return df.loc[mask].copy()


def _top_product_stats(df: pd.DataFrame) -> dict:
    """Stats del producto top para estimaciones simples (en el rango seleccionado), ticket-aware si hay ticket_id."""
    if df is None or df.empty:
        return {
            "top_name": "-",
            "top_rev": 0.0,
            "top_rev_share": 0.0,
            "top_tickets": 0,
            "total_rev": 0.0,
            "total_tickets": 0,
            "granularity": "row",
        }

    if "producto" not in df.columns or "revenue" not in df.columns:
        return {
            "top_name": "-",
            "top_rev": 0.0,
            "top_rev_share": 0.0,
            "top_tickets": 0,
            "total_rev": float(df["revenue"].sum()) if "revenue" in df.columns else 0.0,
            "total_tickets": int(len(df)),
            "granularity": "row",
        }

    df2 = _clean_product(df)
    if df2.empty:
        # si todo era placeholder, devolvemos “vacío”
        return {
            "top_name": "-",
            "top_rev": 0.0,
            "top_rev_share": 0.0,
            "top_tickets": 0,
            "total_rev": float(df["revenue"].sum()) if "revenue" in df.columns else 0.0,
            "total_tickets": int(len(df)),
            "granularity": "row",
        }

    total_rev = float(df2["revenue"].sum()) or 0.0

    g = df2.groupby("producto", as_index=False)["revenue"].sum().sort_values("revenue", ascending=False)
    top_row = g.iloc[0]
    top_name = str(top_row["producto"])
    top_rev = float(top_row["revenue"])
    top_rev_share = (top_rev / total_rev * 100.0) if total_rev else 0.0

    # tickets: real si hay ticket_id, aprox si no
    if "ticket_id" in df2.columns and df2["ticket_id"].notna().any():
        top_tickets = int(df2.loc[df2["producto"] == top_name, "ticket_id"].nunique())
        total_tickets = int(df2["ticket_id"].nunique())
        gran = "ticket"
    else:
        top_tickets = int((df2["producto"] == top_name).sum())
        total_tickets = int(len(df2))
        gran = "row"

    return {
        "top_name": top_name,
        "top_rev": top_rev,
        "top_rev_share": float(top_rev_share),
        "top_tickets": top_tickets,
        "total_rev": total_rev,
        "total_tickets": total_tickets,
        "granularity": gran,
    }


# ---------------------------------------------------------------------
# Modelos de datos (contrato con UI + ranking)
# ---------------------------------------------------------------------
@dataclass(frozen=True)
class Insight:
    """
    Insight accionable con metadata vendible (no técnica).

    NOTA: La app usa `insight_id`, no `id`.
    """
    insight_id: str
    title: str
    evidence: str
    action_hint: str

    impact_type: ImpactType
    kpi_target: KPI

    # Impacto ya ESCALADO al horizonte seleccionado (1×/3×/12× rango)
    estimated_impact_eur: Tuple[float, float]  # (low, high)

    effort: Effort
    time_to_apply_min: int

    confidence: float


# ---------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------
def generate_insights_core(
    df_f: pd.DataFrame,
    *,
    business_type: str,
    uplift_eur_per_ticket: float,
    horizon_multiplier: int,
    horizon_label: str,
    peor_dia_nombre: str,
    mejor_dia_nombre: str,
    gap_mejor_peor_eur: float,
    ventas_peor_dia_en_rango: int,
    features: CoreFeatures,
) -> list[Insight]:
    """
    Genera insights coherentes con:
      - impacto base calculado EN EL RANGO
      - escalado por horizonte_multiplier (1/3/12)
      - evidencia trazable (sin humo)
    """
    out: list[Insight] = []
    hm = int(max(1, horizon_multiplier))

    # Ajuste de confianza si NO hay ticket_id (granularity row)
    conf_shift = -0.15 if features.granularity != "ticket" else 0.0

    top = _top_product_stats(df_f)

    # 1) Subir ticket en el peor día (impacto coherente: rango × multiplicador)
    est = estimate_ticket_uplift_impact(
        tickets_in_range_for_target_day=int(ventas_peor_dia_en_rango),
        uplift_eur_per_ticket=float(uplift_eur_per_ticket),
        horizon_multiplier=hm,
    )

    out.append(
        Insight(
            insight_id="uplift_worst_day_ticket",
            title=f"Subir {features.ticket_medio:,.0f}→ +{uplift_eur_per_ticket:.0f}€ en {peor_dia_nombre}".replace(",", "."),
            evidence=(
                f"Gap {mejor_dia_nombre} vs {peor_dia_nombre} en el rango: {eur(gap_mejor_peor_eur)}. "
                f"Base: {ventas_peor_dia_en_rango} {('tickets' if features.granularity=='ticket' else 'líneas')} en {peor_dia_nombre}."
            ),
            action_hint="Implementa 2 extras (A/B) y ofrécelos solo el día flojo durante 2 semanas.",
            impact_type="revenue",
            kpi_target="ticket_medio",
            estimated_impact_eur=_eur_range(est.low_eur, est.high_eur),
            effort="baja",
            time_to_apply_min=20,
            confidence=float(max(0.35, min(0.90, 0.85 + conf_shift))),
        )
    )

    # 2) Bundle sobre el top (ticket-aware si hay ticket_id)
    if top["top_tickets"] > 0 and top["top_name"] not in _PLACEHOLDER_PRODUCTS:
        adopt_low, adopt_high = 0.10, 0.25
        uplift_low, uplift_high = 1.0, 2.5

        low_rango = top["top_tickets"] * adopt_low * uplift_low
        high_rango = top["top_tickets"] * adopt_high * uplift_high

        out.append(
            Insight(
                insight_id="service_anchor_top",
                title=f"Bundle del top ({top['top_name']})",
                evidence=f"El top concentra {top['top_rev_share']:.1f}% del ingreso del rango ({eur(top['total_rev'])}).",
                action_hint="Crea 1 bundle fijo: top + 1 extra (2 opciones A/B) con guion de 1 frase.",
                impact_type="revenue",
                kpi_target="ticket_medio",
                estimated_impact_eur=_eur_range(low_rango * hm, high_rango * hm),
                effort="media",
                time_to_apply_min=60,
                confidence=float(max(0.30, min(0.85, 0.70 + conf_shift))),
            )
        )

    # 3) Riesgo dependencia Top 3 (medido)
    pct_top3 = float(features.pct_top3_ingresos)

    if pct_top3 >= 55.0:
        out.append(
            Insight(
                insight_id="top3_concentration_high",
                title="Reducir dependencia del Top 3",
                evidence=f"Top 3 concentra {pct_top3:.1f}% de ingresos en el rango.",
                action_hint="Empuja el #4/#5 con 1 pack fijo y mide adopción semanal (sin descuentos).",
                impact_type="risk",
                kpi_target="riesgo_dependencia",
                estimated_impact_eur=_eur_range(0.0, 0.0),
                effort="media",
                time_to_apply_min=45,
                confidence=float(max(0.35, min(0.85, 0.80 + conf_shift))),
            )
        )
    else:
        # Impacto trazable: % de ingresos del rango (no números inventados)
        low = features.ingresos_total * 0.003 * hm
        high = features.ingresos_total * 0.012 * hm

        out.append(
            Insight(
                insight_id="top3_concentration_ok_action",
                title="Rotación semanal de extras (para no estancarte)",
                evidence=f"Top 3 concentra {pct_top3:.1f}% (diversificación razonable).",
                action_hint="Rota 1 extra recomendado por semana y registra tasa de aceptación (sí/no).",
                impact_type="revenue",
                kpi_target="mix_servicios",
                estimated_impact_eur=_eur_range(low, high),
                effort="baja",
                time_to_apply_min=15,
                confidence=float(max(0.30, min(0.75, 0.60 + conf_shift))),
            )
        )

    # 4) Upsell por “tickets de 1 unidad” (SOLO si granularity=ticket)
    pct_u1 = float(features.pct_tickets_unidad_1)
    if features.granularity == "ticket" and pct_u1 >= 50.0:
        adopt_low, adopt_high = 0.08, 0.18
        uplift_low, uplift_high = 1.0, 2.0

        tickets_total = float(features.tickets_total)

        low_rango = tickets_total * adopt_low * uplift_low
        high_rango = tickets_total * adopt_high * uplift_high

        out.append(
            Insight(
                insight_id="upsell_low_items",
                title="Subir ítems por ticket (muchos tickets de 1 unidad)",
                evidence=f"{pct_u1:.1f}% de tickets son de 1 unidad (medido por ticket_id).",
                action_hint="Añade 1 complemento natural (2 opciones) en el cobro. No más de 1 pregunta.",
                impact_type="revenue",
                kpi_target="ticket_medio",
                estimated_impact_eur=_eur_range(low_rango * hm, high_rango * hm),
                effort="baja",
                time_to_apply_min=25,
                confidence=float(max(0.35, min(0.80, 0.70))),
            )
        )

    # Playbook por industria: impacto como % de ingresos (trazable)
    bt = str(business_type or "").strip().lower()
    if bt == "servicios":
        low = features.ingresos_total * 0.005 * hm
        high = features.ingresos_total * 0.020 * hm
        out.append(
            Insight(
                insight_id="service_rebooking_simple",
                title="Rebooking simple (subir frecuencia sin descuento)",
                evidence="Servicios suelen ganar más por frecuencia que por precio.",
                action_hint="Al final: '¿Te reservo ya para X semanas?' + 2 opciones (fecha/hora).",
                impact_type="revenue",
                kpi_target="frecuencia",
                estimated_impact_eur=_eur_range(low, high),
                effort="media",
                time_to_apply_min=40,
                confidence=float(max(0.30, min(0.70, 0.55 + conf_shift))),
            )
        )

    return out


# ---------------------------------------------------------------------
# Wrapper compatible con tu app actual
# ---------------------------------------------------------------------
def generate_insights(
    *,
    df_range: pd.DataFrame,
    kpis,
    profile_type: str,
    profile_subtype: str,
    peor_dia_nombre: str,
    mejor_dia_nombre: str,
    gap_dias_eur: float,
    ventas_peor_dia: int,
    top_item: str,
    slider_subida_ticket_eur: float,
    horizon_multiplier: int,
    horizon_label: str,
    features: Optional[CoreFeatures] = None,
) -> list[Insight]:
    """
    Firma compatible con tu `streamlit_app.py`.

    Cambios:
    - Ya NO hay placeholders.
    - Si no pasas `features`, se calculan aquí con `compute_core_features` (auto-detect ticket_id).
    """
    if features is None:
        # ticket-aware si existe ticket_id
        features = compute_core_features(df_range, ticket_id_col="ticket_id")

    insights = generate_insights_core(
        df_f=df_range,
        business_type=str(profile_type),
        uplift_eur_per_ticket=float(slider_subida_ticket_eur),
        horizon_multiplier=int(horizon_multiplier),
        horizon_label=str(horizon_label),
        peor_dia_nombre=str(peor_dia_nombre),
        mejor_dia_nombre=str(mejor_dia_nombre),
        gap_mejor_peor_eur=float(gap_dias_eur),
        ventas_peor_dia_en_rango=int(ventas_peor_dia),
        features=features,
    )

    # Limpieza: elimina NaN/Inf por seguridad
    out: list[Insight] = []
    for ins in insights:
        vals = [ins.estimated_impact_eur[0], ins.estimated_impact_eur[1], ins.confidence]
        if any(isinstance(v, float) and (math.isnan(v) or math.isinf(v)) for v in vals):
            continue
        out.append(ins)

    return out