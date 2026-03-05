# app/analytics/schema.py
from __future__ import annotations

from dataclasses import dataclass
import re
import pandas as pd


PLACEHOLDER_PRODUCT = "SIN_PRODUCTO"


@dataclass(frozen=True)
class SchemaConfig:
    required_cols: tuple[str, ...] = ("fecha", "revenue", "cantidad", "producto")
    allow_negative_revenue: bool = True
    allow_negative_quantity: bool = True
    drop_zero_rows: bool = False
    dayfirst: bool = True  # ES-friendly


def safe_date_str(dt) -> str:
    try:
        return pd.to_datetime(dt).date().isoformat()
    except Exception:
        return "-"


_CURRENCY_RE = re.compile(r"[€$£]|eur|euro", flags=re.IGNORECASE)


def _coerce_numeric_series(s: pd.Series) -> pd.Series:
    """
    Convierte a numérico soportando ES/US y negativos contables:
      - "1.234,56" / "1,234.56"
      - "€ 12,50"
      - "(1.234,56)" -> -1234.56
    """
    if s is None:
        return pd.Series(dtype="float64")

    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce")

    x = s.astype(str).str.strip()

    # limpiar moneda y espacios
    x = x.str.replace(_CURRENCY_RE, "", regex=True)
    x = x.str.replace("\u00A0", "", regex=False)
    x = x.str.replace(r"\s+", "", regex=True)

    # negativos con paréntesis
    neg_mask = x.str.match(r"^\(.*\)$", na=False)
    if neg_mask.any():
        x = x.where(~neg_mask, x.str.replace(r"^\(|\)$", "", regex=True))

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

    out = pd.to_numeric(x, errors="coerce")  # pandas.to_numeric docs :contentReference[oaicite:1]{index=1}
    if neg_mask.any():
        out = out.where(~neg_mask, -out)

    return out


def ensure_schema(df: pd.DataFrame, config: SchemaConfig | None = None) -> pd.DataFrame:
    """
    Normaliza un df al schema esperado por la app.
    Adjunta métricas de calidad en `out.attrs["quality"]`.
    """
    config = config or SchemaConfig()

    if df is None or df.empty:
        raise ValueError("Dataset vacío o None.")

    missing = set(config.required_cols) - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas en el dataset procesado: {sorted(missing)}")

    out = df.copy()

    # --- fechas (dayfirst ES) ---
    before_rows = len(out)
    out["fecha"] = pd.to_datetime(out["fecha"], errors="coerce", dayfirst=config.dayfirst)  # :contentReference[oaicite:2]{index=2}
    invalid_dates = int(out["fecha"].isna().sum())
    out = out.dropna(subset=["fecha"])

    # --- numéricos (sin silenciar sin medir) ---
    rev_raw = out["revenue"]
    qty_raw = out["cantidad"]

    rev_num = _coerce_numeric_series(rev_raw)
    qty_num = _coerce_numeric_series(qty_raw)

    rev_nan = int(rev_num.isna().sum())
    qty_nan = int(qty_num.isna().sum())

    # Aquí sí rellenamos NaN para que el resto de la app no explote,
    # pero dejamos trazabilidad en quality.
    out["revenue"] = rev_num.fillna(0.0).astype(float)
    out["cantidad"] = qty_num.fillna(0.0).astype(float)

    # --- producto ---
    prod = out["producto"].astype(str).fillna("").str.strip()
    prod = prod.where(prod.ne(""), PLACEHOLDER_PRODUCT)
    out["producto"] = prod

    placeholders = int((out["producto"].astype(str).str.strip().str.upper() == PLACEHOLDER_PRODUCT).sum())

    # --- reglas ---
    if not config.allow_negative_revenue:
        out = out[out["revenue"] >= 0]

    if not config.allow_negative_quantity:
        out = out[out["cantidad"] >= 0]

    if config.drop_zero_rows:
        out = out[(out["revenue"] != 0.0) | (out["cantidad"] != 0.0)]

    out = out.sort_values("fecha")

    if out.empty:
        raise ValueError("Tras normalizar el schema, el dataset quedó vacío (fechas inválidas o filtros).")

    # --- quality report (para UI/warnings) ---
    out.attrs["quality"] = {
        "rows_in": int(before_rows),
        "rows_after_valid_dates": int(len(out)),
        "invalid_dates_dropped": int(invalid_dates),
        "revenue_nan_before_fill": int(rev_nan),
        "cantidad_nan_before_fill": int(qty_nan),
        "placeholders_producto": int(placeholders),
        "dayfirst": bool(config.dayfirst),
    }

    return out