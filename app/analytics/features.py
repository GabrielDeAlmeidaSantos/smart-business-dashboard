from __future__ import annotations

from dataclasses import dataclass
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

    # Distribución simple: % de tickets con 1 unidad (proxy de upsell)
    pct_tickets_unidad_1: float


def compute_core_features(df: pd.DataFrame) -> CoreFeatures:
    """
    Extrae features esenciales del rango filtrado.

    Nota:
        Se asume 1 fila = 1 ticket (por ahora).
        Si luego tienes ticket_id, aquí se agruparía primero.
    """
    ingresos_total = float(df["revenue"].sum())
    tickets_total = int(len(df))
    ticket_medio = float(df["revenue"].mean()) if tickets_total else 0.0
    unidades_total = float(df["cantidad"].sum())

    # Día de semana
    tmp = df.copy()
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

    # Tickets por ocurrencia del peor día
    peor_dia = str(peor["dia"])
    df_peor = tmp[tmp["dia"] == peor_dia]
    ocurrencias = int(df_peor["fecha"].dt.date.nunique())  # nº de días distintos
    tickets_por_oc = (len(df_peor) / ocurrencias) if ocurrencias else 0.0

    # Concentración Top3
    top3_ing = (
        df.groupby("producto", as_index=False)["revenue"]
        .sum()
        .sort_values("revenue", ascending=False)
        .head(3)["revenue"]
        .sum()
    )
    pct_top3 = (float(top3_ing) / ingresos_total * 100.0) if ingresos_total else 0.0

    # Proxy upsell: tickets con 1 unidad (si cantidad ~ unidades por ticket)
    pct_u1 = float((df["cantidad"] <= 1).mean() * 100.0) if tickets_total else 0.0

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
    )