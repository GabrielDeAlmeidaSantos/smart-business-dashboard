# app/analytics/features.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd


MAPA_DIAS = {
    0: "Lunes",
    1: "Martes",
    2: "Miércoles",
    3: "Jueves",
    4: "Viernes",
    5: "Sábado",
    6: "Domingo",
}

# Placeholders a excluir en cálculos “de negocio”
_PLACEHOLDER_PRODUCTS = {"-", "SIN_PRODUCTO", "sin_producto", "nan", "none", ""}


@dataclass(frozen=True)
class CoreFeatures:
    """
    Features base para diagnóstico y recomendación.
    """
    ingresos_total: float

    tickets_total: int
    ticket_medio: float
    unidades_total: float

    peor_dia_nombre: str
    mejor_dia_nombre: str
    gap_mejor_peor: float

    # Tickets por ocurrencia del peor día (promedio)
    peor_dia_tickets_por_ocurrencia: float
    peor_dia_ocurrencias_en_rango: int

    # Concentración
    pct_top3_ingresos: float

    # Distribución: % tickets con 1 unidad (si hay ticket_id: real; si no: aproximado)
    pct_tickets_unidad_1: float

    # Honestidad de cálculo
    granularity: str  # "ticket" o "row"
    note: str


def _detect_ticket_col(df: pd.DataFrame, ticket_id_col: Optional[str] = None) -> Optional[str]:
    """Auto-detect simple: usa ticket_id_col si existe, si no prueba 'ticket_id'."""
    if ticket_id_col and ticket_id_col in df.columns:
        return ticket_id_col
    if "ticket_id" in df.columns:
        try:
            if df["ticket_id"].notna().any():
                return "ticket_id"
        except Exception:
            return None
    return None


def _ticket_level(df: pd.DataFrame, ticket_col: str) -> pd.DataFrame:
    """Agrega a nivel ticket (1 fila por ticket)."""
    return (
        df.groupby(ticket_col, as_index=False)
        .agg(
            fecha=("fecha", "min"),
            revenue=("revenue", "sum"),
            cantidad=("cantidad", "sum"),
        )
    )


def _clean_product_for_top(df: pd.DataFrame) -> pd.DataFrame:
    """Excluye placeholders de producto para cálculos de concentración."""
    if "producto" not in df.columns:
        return df
    p = df["producto"].astype(str).str.strip().str.lower()
    mask = ~p.isin({x.lower() for x in _PLACEHOLDER_PRODUCTS})
    return df.loc[mask].copy()


def compute_core_features(df: pd.DataFrame, *, ticket_id_col: Optional[str] = None) -> CoreFeatures:
    """
    Extrae features esenciales del rango filtrado.

    - Si existe ticket_id: cálculos reales a nivel ticket.
    - Si no: fallback (1 fila ~ 1 ticket) y lo marca en granularity/note.
    """
    if df is None or df.empty:
        return CoreFeatures(
            ingresos_total=0.0,
            tickets_total=0,
            ticket_medio=0.0,
            unidades_total=0.0,
            peor_dia_nombre="-",
            mejor_dia_nombre="-",
            gap_mejor_peor=0.0,
            peor_dia_tickets_por_ocurrencia=0.0,
            peor_dia_ocurrencias_en_rango=0,
            pct_top3_ingresos=0.0,
            pct_tickets_unidad_1=0.0,
            granularity="row",
            note="empty",
        )

    # Requisitos mínimos
    for col in ("fecha", "revenue", "cantidad"):
        if col not in df.columns:
            raise ValueError(f"Falta columna requerida en features: '{col}'")

    ticket_col = _detect_ticket_col(df, ticket_id_col=ticket_id_col)

    # Base para tickets (ticket-level si se puede)
    if ticket_col:
        base = _ticket_level(df, ticket_col)
        granularity = "ticket"
        note = f"by_ticket({ticket_col})"
    else:
        base = df[["fecha", "revenue", "cantidad"]].copy()
        granularity = "row"
        note = "fallback_row_equals_ticket"

    ingresos_total = float(base["revenue"].sum())
    tickets_total = int(len(base))
    ticket_medio = float(ingresos_total / tickets_total) if tickets_total else 0.0
    unidades_total = float(base["cantidad"].sum())

    # Día de semana
    tmp = base.copy()
    tmp["dow"] = tmp["fecha"].dt.dayofweek
    tmp["dia"] = tmp["dow"].map(MAPA_DIAS)

    ingresos_dia = (
        tmp.groupby(["dow", "dia"], as_index=False)["revenue"]
        .sum()
        .rename(columns={"revenue": "ingresos"})
        .sort_values("dow")
    )

    peor = ingresos_dia.sort_values("ingresos").iloc[0]
    mejor = ingresos_dia.sort_values("ingresos", ascending=False).iloc[0]
    gap = float(mejor["ingresos"] - peor["ingresos"])

    peor_dia = str(peor["dia"])
    df_peor = tmp[tmp["dia"] == peor_dia]
    ocurrencias = int(df_peor["fecha"].dt.date.nunique())
    tickets_por_oc = float((len(df_peor) / ocurrencias) if ocurrencias else 0.0)

    # Concentración Top3 (si hay producto)
    pct_top3 = 0.0
    if "producto" in df.columns:
        df_prod = _clean_product_for_top(df)
        if not df_prod.empty:
            top3_ing = (
                df_prod.groupby("producto", as_index=False)["revenue"]
                .sum()
                .sort_values("revenue", ascending=False)
                .head(3)["revenue"]
                .sum()
            )
            pct_top3 = (float(top3_ing) / ingresos_total * 100.0) if ingresos_total else 0.0

    # % tickets con 1 unidad:
    # - ticket mode: real (sum cantidad por ticket <=1)
    # - row mode: aproximado (cantidad por fila <=1)
    if tickets_total:
        pct_u1 = float((base["cantidad"] <= 1).mean() * 100.0)
    else:
        pct_u1 = 0.0

    return CoreFeatures(
        ingresos_total=ingresos_total,
        tickets_total=tickets_total,
        ticket_medio=ticket_medio,
        unidades_total=unidades_total,
        peor_dia_nombre=peor_dia,
        mejor_dia_nombre=str(mejor["dia"]),
        gap_mejor_peor=gap,
        peor_dia_tickets_por_ocurrencia=float(tickets_por_oc),
        peor_dia_ocurrencias_en_rango=ocurrencias,
        pct_top3_ingresos=float(pct_top3),
        pct_tickets_unidad_1=float(pct_u1),
        granularity=granularity,
        note=note,
    )