from __future__ import annotations

from dataclasses import dataclass
import pandas as pd


@dataclass(frozen=True)
class DatasetConfig:
    """
    Configuración del dataset.

    row_is_ticket:
        - True: cada fila representa un ticket/venta (tu supuesto actual).
        - False: cada fila representa una línea de ticket y hay que agrupar por ticket_id.
    """
    row_is_ticket: bool = True
    ticket_id_col: str | None = None


def ensure_schema(df: pd.DataFrame) -> pd.DataFrame:
    """
    Valida el esquema mínimo requerido y normaliza fechas.

    Requeridos:
        - fecha: datetime
        - revenue: importe (float)
        - cantidad: unidades (num)
        - producto: nombre servicio/producto (str)

    Returns:
        DataFrame limpio y ordenado por fecha.
    """
    required_cols = {"fecha", "revenue", "cantidad", "producto"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas en el dataset procesado: {sorted(missing)}")

    out = df.copy()
    out["fecha"] = pd.to_datetime(out["fecha"], errors="coerce")
    out = out.dropna(subset=["fecha"]).sort_values("fecha")

    # Tipos defensivos
    out["revenue"] = pd.to_numeric(out["revenue"], errors="coerce").fillna(0.0)
    out["cantidad"] = pd.to_numeric(out["cantidad"], errors="coerce").fillna(0.0)
    out["producto"] = out["producto"].astype(str).fillna("")

    return out