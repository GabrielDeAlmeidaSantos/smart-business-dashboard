# app/analytics/insights.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Tuple
import math
import pandas as pd

from .impact import estimate_ticket_uplift_impact

ImpactType = Literal["revenue", "risk", "time"]
Effort = Literal["baja", "media", "alta"]
KPI = Literal["ticket_medio", "ingresos", "mix_servicios", "riesgo_dependencia", "frecuencia"]


def _eur_range(low: float, high: float) -> Tuple[float, float]:
    """Normaliza un rango € (no negativo y high >= low)."""
    low = float(max(0.0, low))
    high = float(max(low, high))
    return (low, high)


def _top_product_stats(df: pd.DataFrame) -> dict:
    """Stats del producto top para estimaciones simples de bundles/upsell (en el rango seleccionado)."""
    if df is None or df.empty:
        return {
            "top_name": "-",
            "top_rev": 0.0,
            "top_rev_share": 0.0,
            "top_tickets": 0,
            "total_rev": 0.0,
            "total_tickets": 0,
        }

    if "producto" not in df.columns or "revenue" not in df.columns:
        return {
            "top_name": "-",
            "top_rev": 0.0,
            "top_rev_share": 0.0,
            "top_tickets": 0,
            "total_rev": 0.0,
            "total_tickets": int(len(df)),
        }

    total_rev = float(df["revenue"].sum()) or 0.0
    g = df.groupby("producto", as_index=False)["revenue"].sum().sort_values("revenue", ascending=False)

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
# LÓGICA "CORE" (ahora coherente con el nuevo horizonte por multiplicador)
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
) -> list[Insight]:
    """
    Genera insights coherentes con:
      - impacto base calculado EN EL RANGO
      - escalado por `horizon_multiplier` (1/3/12)
      - evidencia trazable (sin humo)
    """
    out: list[Insight] = []
    top = _top_product_stats(df_f)

    hm = int(max(1, horizon_multiplier))

    # 1) Subir ticket en el peor día (impacto coherente: rango × multiplicador)
    est = estimate_ticket_uplift_impact(
        tickets_in_range_for_target_day=int(ventas_peor_dia_en_rango),
        uplift_eur_per_ticket=float(uplift_eur_per_ticket),
        horizon_multiplier=hm,
    )

    out.append(
        Insight(
            insight_id="uplift_worst_day_ticket",
            title=f"Subir ticket en {peor_dia_nombre}",
            evidence=(
                f"Gap {mejor_dia_nombre} vs {peor_dia_nombre}: "
                f"{float(gap_mejor_peor_eur):,.2f}€ en el rango."
            ).replace(",", "."),
            action_hint="Implementa 2 extras (A/B) y ofrécelos solo el día flojo durante 2 semanas.",
            impact_type="revenue",
            kpi_target="ticket_medio",
            estimated_impact_eur=_eur_range(est.low_eur, est.high_eur),
            effort="baja",
            time_to_apply_min=20,
            confidence=0.85,
        )
    )

    # 2) Ancla: bundle sobre el servicio top (top_tickets está en el rango ⇒ escalar por hm)
    if top["top_tickets"] > 0:
        adopt_low, adopt_high = 0.10, 0.25
        uplift_low, uplift_high = 1.0, 2.5

        low_rango = top["top_tickets"] * adopt_low * uplift_low
        high_rango = top["top_tickets"] * adopt_high * uplift_high

        out.append(
            Insight(
                insight_id="service_anchor_top",
                title=f"Bundle del servicio top ({top['top_name']})",
                evidence=f"El servicio top concentra {top['top_rev_share']:.1f}% del ingreso en el rango.",
                action_hint="Crea 1 bundle fijo: servicio top + 1 extra (2 opciones A/B) con guion de 1 frase.",
                impact_type="revenue",
                kpi_target="ticket_medio",
                estimated_impact_eur=_eur_range(low_rango * hm, high_rango * hm),
                effort="media",
                time_to_apply_min=60,
                confidence=0.70,
            )
        )

    # 3) Riesgo dependencia Top 3 (sin €; solo KPI de riesgo)
    # Nota: ahora mismo no tienes pct_top3 real en este wrapper; lo conectaremos cuando pases features/context.
    pct_top3 = 0.0
    if pct_top3 >= 55:
        out.append(
            Insight(
                insight_id="top3_concentration_high",
                title="Reducir dependencia del Top 3",
                evidence=f"Top 3 concentra {pct_top3:.1f}% de ingresos.",
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
        out.append(
            Insight(
                insight_id="top3_concentration_ok_action",
                title="Rotación semanal de extras (para no estancarte)",
                evidence=f"Top 3 concentra {pct_top3:.1f}% (razonable o no medido).",
                action_hint="Rota 1 extra recomendado por semana y registra tasa de aceptación (sí/no).",
                impact_type="revenue",
                kpi_target="mix_servicios",
                # Estos importes eran “en el rango”; si los muestras como horizonte, escala
                estimated_impact_eur=_eur_range(30.0 * hm, 180.0 * hm),
                effort="baja",
                time_to_apply_min=15,
                confidence=0.60,
            )
        )

    # 4) Muchos tickets con 1 unidad (proxy upsell)
    pct_tickets_unidad_1 = 0.0
    if pct_tickets_unidad_1 >= 50:
        adopt_low, adopt_high = 0.08, 0.18
        uplift_low, uplift_high = 1.0, 2.0
        tickets_total = float(len(df_f))

        low_rango = tickets_total * adopt_low * uplift_low
        high_rango = tickets_total * adopt_high * uplift_high

        out.append(
            Insight(
                insight_id="upsell_low_items",
                title="Subir ítems por ticket (muchos tickets de 1 unidad)",
                evidence=f"{pct_tickets_unidad_1:.1f}% de tickets parecen ser de 1 unidad (pendiente de conectar).",
                action_hint="Añade 1 complemento natural (2 opciones) en el cobro. No más de 1 pregunta.",
                impact_type="revenue",
                kpi_target="ticket_medio",
                estimated_impact_eur=_eur_range(low_rango * hm, high_rango * hm),
                effort="baja",
                time_to_apply_min=25,
                confidence=0.70,
            )
        )

    # Playbook específico por industria (valores fijos ⇒ también escalar para coherencia de UI)
    if business_type == "servicios":
        out.append(
            Insight(
                insight_id="service_rebooking_simple",
                title="Rebooking simple (subir frecuencia sin descuento)",
                evidence="Servicios suelen ganar mucho más por frecuencia que por precio.",
                action_hint="Al final: '¿Te reservo ya para X semanas?' + 2 opciones (fecha/hora).",
                impact_type="revenue",
                kpi_target="frecuencia",
                estimated_impact_eur=_eur_range(50.0 * hm, 250.0 * hm),
                effort="media",
                time_to_apply_min=40,
                confidence=0.55,
            )
        )

    return out


# ---------------------------------------------------------------------
# WRAPPER COMPATIBLE CON TU app.py (firma nueva)
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
) -> list[Insight]:
    """
    Firma compatible con tu `app.py` actual.

    Importante:
    - No “inventa” métricas no disponibles (pct_top3, pct_unidad_1), quedan a 0.0 hasta que conectemos features/context.
    - Todo impacto devuelto ya está escalado al horizonte (1×/3×/12× rango).
    """
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
    )

    # Limpieza: elimina NaN/Inf por seguridad
    out: list[Insight] = []
    for ins in insights:
        vals = [ins.estimated_impact_eur[0], ins.estimated_impact_eur[1], ins.confidence]
        if any(isinstance(v, float) and (math.isnan(v) or math.isinf(v)) for v in vals):
            continue
        out.append(ins)

    return out