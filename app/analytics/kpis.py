# app/analytics/kpis.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any

import pandas as pd


# ----------------------------
# Formatos
# ----------------------------
def eur(x: float) -> str:
    """Formato € estilo ES (63.539,00 €)."""
    try:
        x = float(x)
    except Exception:
        x = 0.0
    s = f"{x:,.2f}"
    s = s.replace(",", "X").replace(".", ",").replace("X", ".")
    return f"{s} €"


def pct_str(delta: float, base: float, *, eps: float = 1e-9) -> str:
    """Devuelve '+3.2%' o '-1.1%'. Si base≈0 => '' para evitar porcentajes sin sentido."""
    try:
        base = float(base)
        delta = float(delta)
    except Exception:
        return ""

    if abs(base) < eps:
        return ""

    p = (delta / base) * 100.0
    sign = "+" if p >= 0 else ""
    return f"{sign}{p:.1f}%"


# ----------------------------
# Config + bundles
# ----------------------------
@dataclass(frozen=True)
class KPIConfig:
    """
    Config para calcular KPIs de forma correcta.

    - ticket_id_col:
        Si existe en el dataset, se agregará por ticket para calcular ventas y ticket medio reales.
        Si es None o no existe, fallback: 1 fila = 1 ticket (aprox).
    - revenue_col / qty_col / date_col:
        Nombres de columnas.
    - allow_negative_revenue:
        Si False, filtra revenue < 0 (si tu dataset no tiene devoluciones).
    """
    ticket_id_col: Optional[str] = None
    revenue_col: str = "revenue"
    qty_col: str = "cantidad"
    date_col: str = "fecha"
    product_col: str = "producto"
    allow_negative_revenue: bool = True


@dataclass(frozen=True)
class KPIBundle:
    ingresos: float
    ventas: int
    ticket: float
    unidades: float

    granularity: str  # "ticket" o "row"
    note: str         # explicación breve para admin/debug

    # Etiquetas “honestas” para UI
    ventas_label: str
    ticket_label: str

    # Calidad (mínimo útil para warnings)
    quality: Dict[str, Any]


# ----------------------------
# Helpers
# ----------------------------
def _ensure_cols(df: pd.DataFrame, cols: list[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas para KPIs: {missing}")


def _detect_ticket_id_col(df: pd.DataFrame, cfg: KPIConfig) -> Optional[str]:
    """Auto-detect simple: si cfg.ticket_id_col viene None, intenta 'ticket_id'."""
    if cfg.ticket_id_col and cfg.ticket_id_col in df.columns:
        return cfg.ticket_id_col
    if cfg.ticket_id_col is None and "ticket_id" in df.columns:
        # solo lo usamos si tiene algo (no todo NaN)
        try:
            if df["ticket_id"].notna().any():
                return "ticket_id"
        except Exception:
            return None
    return None


def _as_ticket_level(df: pd.DataFrame, cfg: KPIConfig, ticket_col: str) -> pd.DataFrame:
    """
    Convierte a nivel ticket.
    Devuelve df con una fila por ticket:
      - fecha: min fecha por ticket
      - revenue: suma revenue por ticket
      - cantidad: suma cantidad por ticket
    """
    _ensure_cols(df, [ticket_col, cfg.date_col, cfg.revenue_col, cfg.qty_col])

    g = df.groupby(ticket_col, as_index=False).agg(
        **{
            cfg.date_col: (cfg.date_col, "min"),
            cfg.revenue_col: (cfg.revenue_col, "sum"),
            cfg.qty_col: (cfg.qty_col, "sum"),
        }
    )
    return g


def _quality_report(df: pd.DataFrame, cfg: KPIConfig, *, ticket_mode: bool) -> Dict[str, Any]:
    """Métricas básicas de calidad para warnings."""
    q: Dict[str, Any] = {}
    try:
        q["rows"] = int(len(df))
        q["pct_nan_revenue"] = float(df[cfg.revenue_col].isna().mean() * 100.0) if cfg.revenue_col in df.columns else None
        q["pct_nan_qty"] = float(df[cfg.qty_col].isna().mean() * 100.0) if cfg.qty_col in df.columns else None
        q["ticket_mode"] = bool(ticket_mode)
    except Exception:
        return {"rows": int(len(df)), "ticket_mode": bool(ticket_mode)}
    return q


# ----------------------------
# KPIs
# ----------------------------
def compute_kpis(df_: pd.DataFrame, cfg: KPIConfig | None = None) -> KPIBundle:
    """KPIs robustos con fallback honesto + autodetección de ticket_id."""
    cfg = cfg or KPIConfig()

    if df_ is None or df_.empty:
        return KPIBundle(
            ingresos=0.0, ventas=0, ticket=0.0, unidades=0.0,
            granularity="row", note="empty",
            ventas_label="Líneas", ticket_label="Importe medio por línea",
            quality={"rows": 0, "ticket_mode": False},
        )

    _ensure_cols(df_, [cfg.date_col, cfg.revenue_col, cfg.qty_col])

    df = df_.copy()

    if not cfg.allow_negative_revenue:
        df = df[df[cfg.revenue_col] >= 0]

    ticket_col = _detect_ticket_id_col(df, cfg)

    # Modo ticket si hay ticket_id_col válido
    if ticket_col:
        df_t = _as_ticket_level(df, cfg, ticket_col=ticket_col)
        ingresos = float(df_t[cfg.revenue_col].sum())
        ventas = int(df_t.shape[0])
        ticket = float(ingresos / ventas) if ventas else 0.0
        unidades = float(df_t[cfg.qty_col].sum())

        return KPIBundle(
            ingresos=ingresos,
            ventas=ventas,
            ticket=ticket,
            unidades=unidades,
            granularity="ticket",
            note=f"by_ticket({ticket_col})",
            ventas_label="Tickets",
            ticket_label="Ticket medio",
            quality=_quality_report(df_t, cfg, ticket_mode=True),
        )

    # Fallback: 1 fila = 1 ticket (aprox)
    ingresos = float(df[cfg.revenue_col].sum())
    ventas = int(df.shape[0])
    ticket = float(ingresos / ventas) if ventas else 0.0
    unidades = float(df[cfg.qty_col].sum())

    return KPIBundle(
        ingresos=ingresos,
        ventas=ventas,
        ticket=ticket,
        unidades=unidades,
        granularity="row",
        note="fallback_row_equals_ticket",
        ventas_label="Líneas (aprox.)",
        ticket_label="Importe medio por línea (aprox.)",
        quality=_quality_report(df, cfg, ticket_mode=False),
    )


# ----------------------------
# Ingresos por día de semana (y ventas del peor día)
# ----------------------------
def ingresos_por_dia_semana(df_f: pd.DataFrame, cfg: KPIConfig | None = None):
    """Ingresos por día de semana + info (mejor/peor/gap/ventas_peor)."""
    cfg = cfg or KPIConfig()
    _ensure_cols(df_f, [cfg.date_col, cfg.revenue_col])

    mapa_dias = {
        0: "Lunes",
        1: "Martes",
        2: "Miércoles",
        3: "Jueves",
        4: "Viernes",
        5: "Sábado",
        6: "Domingo",
    }

    df_sem = df_f.copy()
    df_sem["dia_semana"] = df_sem[cfg.date_col].dt.dayofweek
    df_sem["dia_nombre"] = df_sem["dia_semana"].map(mapa_dias)

    ingresos_dia = (
        df_sem.groupby(["dia_semana", "dia_nombre"], as_index=False)[cfg.revenue_col]
        .sum()
        .sort_values("dia_semana")
        .rename(columns={cfg.revenue_col: "ingresos"})
    )

    peor = ingresos_dia.sort_values("ingresos").iloc[0]
    mejor = ingresos_dia.sort_values("ingresos", ascending=False).iloc[0]
    gap = float(mejor["ingresos"] - peor["ingresos"])

    ticket_col = _detect_ticket_id_col(df_sem, cfg)

    # Ventas por día
    if ticket_col:
        ventas_por_dia = (
            df_sem.groupby("dia_nombre")[ticket_col]
            .nunique()
            .reset_index()
            .rename(columns={ticket_col: "ventas"})
        )
        granularity = "ticket"
        note = f"by_ticket({ticket_col})"
    else:
        ventas_por_dia = (
            df_sem.groupby("dia_nombre", as_index=False)
            .size()
            .rename(columns={"size": "ventas"})
        )
        granularity = "row"
        note = "fallback_row_equals_ticket"

    ventas_peor = int(
        ventas_por_dia.loc[ventas_por_dia["dia_nombre"] == peor["dia_nombre"], "ventas"].iloc[0]
    )

    info = {
        "df_sem": df_sem,
        "ingresos_dia": ingresos_dia,
        "peor": peor,
        "mejor": mejor,
        "gap": gap,
        "ventas_peor": ventas_peor,
        "granularity": granularity,
        "note": note,
    }
    return ingresos_dia, info


# ----------------------------
# Serie temporal
# ----------------------------
def build_serie_tiempo(df_f: pd.DataFrame, agrupacion: str, cfg: KPIConfig | None = None) -> pd.DataFrame:
    """Serie temporal agregada por día/semana/mes (ingresos)."""
    cfg = cfg or KPIConfig()
    _ensure_cols(df_f, [cfg.date_col, cfg.revenue_col])

    df_temp = df_f.copy()

    if agrupacion == "Día":
        df_temp["periodo"] = df_temp[cfg.date_col].dt.floor("D")
    elif agrupacion == "Semana":
        # Inicio de semana “anclado” (pandas Period.start_time)
        df_temp["periodo"] = df_temp[cfg.date_col].dt.to_period("W-MON").apply(lambda r: r.start_time)
    else:  # Mes
        df_temp["periodo"] = df_temp[cfg.date_col].dt.to_period("M").dt.to_timestamp()

    serie = (
        df_temp.groupby("periodo", as_index=False)[cfg.revenue_col]
        .sum()
        .sort_values("periodo")
        .rename(columns={cfg.revenue_col: "ingresos"})
    )
    return serie