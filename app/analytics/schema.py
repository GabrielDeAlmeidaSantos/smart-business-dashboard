from __future__ import annotations

from dataclasses import dataclass
import pandas as pd


@dataclass(frozen=True)
class SchemaConfig:
    required_cols: tuple[str, ...] = ("fecha", "revenue", "cantidad", "producto")
    allow_negative_revenue: bool = True
    allow_negative_quantity: bool = True
    drop_zero_rows: bool = False


def safe_date_str(dt) -> str:
    try:
        return pd.to_datetime(dt).date().isoformat()
    except Exception:
        return "-"


def _coerce_numeric_series(s: pd.Series) -> pd.Series:
    if s is None:
        return pd.Series(dtype="float64")

    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce")

    x = s.astype(str).str.strip()
    x = x.str.replace("€", "", regex=False).str.replace("\u00A0", "", regex=False)

    has_comma = x.str.contains(",", na=False)
    has_dot = x.str.contains(r"\.", na=False)

    both = has_comma & has_dot
    if both.any():
        last_comma = x.str.rfind(",")
        last_dot = x.str.rfind(".")
        es_mask = both & (last_comma > last_dot)
        us_mask = both & ~es_mask

        x = x.where(~es_mask, x.str.replace(".", "", regex=False).str.replace(",", ".", regex=False))
        x = x.where(~us_mask, x.str.replace(",", "", regex=False))

    only_comma = has_comma & ~has_dot
    x = x.where(~only_comma, x.str.replace(",", ".", regex=False))

    return pd.to_numeric(x, errors="coerce")


def ensure_schema(df: pd.DataFrame, config: SchemaConfig | None = None) -> pd.DataFrame:
    config = config or SchemaConfig()

    if df is None or df.empty:
        raise ValueError("Dataset vacío o None.")

    missing = set(config.required_cols) - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas en el dataset procesado: {sorted(missing)}")

    out = df.copy()

    out["fecha"] = pd.to_datetime(out["fecha"], errors="coerce")
    out = out.dropna(subset=["fecha"])

    out["revenue"] = _coerce_numeric_series(out["revenue"]).fillna(0.0).astype(float)
    out["cantidad"] = _coerce_numeric_series(out["cantidad"]).fillna(0.0).astype(float)

    out["producto"] = out["producto"].fillna("-").astype(str).str.strip()
    out.loc[out["producto"].eq(""), "producto"] = "-"

    if not config.allow_negative_revenue:
        out = out[out["revenue"] >= 0]

    if not config.allow_negative_quantity:
        out = out[out["cantidad"] >= 0]

    if config.drop_zero_rows:
        out = out[(out["revenue"] != 0.0) | (out["cantidad"] != 0.0)]

    out = out.sort_values("fecha")

    if out.empty:
        raise ValueError("Tras normalizar el schema, el dataset quedó vacío (fechas inválidas o filtros).")

    return out