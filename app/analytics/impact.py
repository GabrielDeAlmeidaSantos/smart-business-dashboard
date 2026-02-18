from __future__ import annotations

import pandas as pd


def estimate_extra_revenue_for_worst_day(
    df_filtered: pd.DataFrame,
    worst_day_name: str,
    uplift_eur_per_ticket: float,
    horizon: str,
) -> float:
    """
    Estima impacto extra por mejorar el ticket SOLO en el peor día.

    Problema que corrige:
        En tu versión actual, 'ventas_peor' ya depende del rango filtrado,
        y luego multiplicas por 3/12, lo que puede inflar.

    Enfoque correcto:
        1) Calculamos tickets por ocurrencia del peor día dentro del rango (promedio).
        2) Proyectamos a horizonte usando ritmo semanal estimado del rango.

    Args:
        df_filtered: DataFrame ya filtrado por rango.
        worst_day_name: 'Miércoles', etc.
        uplift_eur_per_ticket: +€ por ticket en ese día.
        horizon: 'Mes' | 'Trimestre' | 'Año'

    Returns:
        Impacto estimado en € para el horizonte.
    """
    if df_filtered.empty or uplift_eur_per_ticket <= 0:
        return 0.0

    tmp = df_filtered.copy()
    tmp["dow"] = tmp["fecha"].dt.dayofweek
    mapa = {0:"Lunes",1:"Martes",2:"Miércoles",3:"Jueves",4:"Viernes",5:"Sábado",6:"Domingo"}
    tmp["dia"] = tmp["dow"].map(mapa)

    df_worst = tmp[tmp["dia"] == worst_day_name]
    if df_worst.empty:
        return 0.0

    # nº de ocurrencias del peor día en el rango (p.ej. nº de miércoles)
    occ = int(df_worst["fecha"].dt.date.nunique())
    tickets_per_occ = len(df_worst) / occ if occ else 0.0

    # Proyección por semanas (ritmo del rango)
    start = tmp["fecha"].min().normalize()
    end = tmp["fecha"].max().normalize()
    days = max(int((end - start).days) + 1, 1)
    weeks = days / 7.0

    # Ocurrencias por semana ~ 1 (un miércoles por semana)
    worst_occ_per_week = occ / weeks if weeks > 0 else 1.0

    if horizon == "Mes":
        weeks_h = 4.345  # promedio
    elif horizon == "Trimestre":
        weeks_h = 13.035
    else:
        weeks_h = 52.142

    projected_occ = worst_occ_per_week * weeks_h
    projected_tickets = projected_occ * tickets_per_occ

    return float(projected_tickets * uplift_eur_per_ticket)