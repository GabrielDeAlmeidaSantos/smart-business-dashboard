# src/pipeline.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import re
from typing import Dict, List, Optional, Tuple

import pandas as pd


# =============================================================================
# Objetivo del pipeline (v1 robusta)
# - Acepta Excels/CSVs “reales” con nombres de columnas variados
# - Normaliza a un esquema interno TRACIABLE:
#     fecha, producto, cantidad, precio_unitario, importe_total, revenue
# - Calcula revenue con reglas claras:
#     1) Si existe importe_total -> revenue = importe_total
#     2) Si no, si existe precio_unitario -> revenue = cantidad * precio_unitario
# - Deja metadata detallada para debug/venta (trazabilidad)
# =============================================================================


# ----------------------------
# Config: esquema interno
# ----------------------------
ESQUEMA_INTERNO = ["fecha", "producto", "cantidad", "precio_unitario", "importe_total"]  # revenue se calcula

# Sinónimos (separados: unitario vs total) — esto es clave
SINONIMOS = {
    "fecha": [
        "fecha", "dia", "día", "fecha venta", "fecha_venta", "fecha de venta",
        "date", "fecha pedido", "fecha_pedido", "created_at", "timestamp",
        "fecha factura", "fecha_factura", "emitido", "issued_at"
    ],
    "producto": [
        "producto", "producto/servicio", "servicio", "articulo", "artículo", "item",
        "nombre producto", "nombre_producto", "descripcion", "descripción", "product",
        "concepto", "detalle", "producto descripcion", "desc"
    ],
    "cantidad": [
        "cantidad", "uds", "unidades", "unidad", "qty", "quantity", "n", "numero", "número",
        "cant", "q", "cantidad vendida", "uds vendidas"
    ],
    # Unitario (IMPORTANTÍSIMO separarlo)
    "precio_unitario": [
        "precio unitario", "precio_unitario", "unit price", "unit_price", "pvp",
        "precio", "price", "precio uds", "precio por unidad", "importe unitario"
    ],
    # Total por línea/ticket (IMPORTANTÍSIMO separarlo)
    "importe_total": [
        "importe", "importe total", "importe_total", "total", "total linea", "total línea",
        "amount", "line total", "line_total", "total (€)", "importe (€)", "subtotal", "neto"
    ],
}

# Keywords extra para “contains match” (heurística secundaria)
KEYWORDS = {
    "fecha": ["fecha", "dia", "date", "created", "timestamp", "issued", "emitido"],
    "producto": ["producto", "servicio", "articulo", "item", "descripcion", "concepto", "detalle", "product", "desc"],
    "cantidad": ["cantidad", "uds", "unidades", "qty", "quantity", "numero", "cant"],
    "precio_unitario": ["unit", "unitario", "pvp", "precio", "price"],
    "importe_total": ["importe", "total", "amount", "subtotal", "neto", "€", "eur"],
}


@dataclass(frozen=True)
class PipelinePaths:
    """Rutas utilizadas por el pipeline."""
    input_dir: Path = Path("data/input")
    processed_dir: Path = Path("data/processed")
    input_file: Optional[Path] = None  # si None, auto-detect
    output_clean: Path = Path("data/processed/ventas_limpias.parquet")
    output_kpis: Path = Path("data/processed/kpis.parquet")
    output_meta: Path = Path("data/processed/metadata.json")


# =============================================================================
# Utils: normalización de columnas
# =============================================================================
def _normalize_col_name(name: str) -> str:
    """Normaliza nombres de columnas para facilitar matching."""
    name = str(name).strip().lower()
    name = name.replace("_", " ")
    name = re.sub(r"\s+", " ", name)

    # quitar acentos simple
    name = (
        name.replace("á", "a").replace("é", "e").replace("í", "i")
        .replace("ó", "o").replace("ú", "u").replace("ñ", "n")
    )
    return name


def _build_synonyms_norm() -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    for k, vals in SINONIMOS.items():
        out[k] = [_normalize_col_name(v) for v in vals]
    return out


SYN_NORM = _build_synonyms_norm()


def detect_column_mapping(columns: List[str]) -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    Detecta mapeo {col_original -> col_estandar} y también devuelve
    {col_estandar -> col_original} para trazabilidad.

    Matching:
      1) exact match vs sinónimos normalizados
      2) contains match con keywords (heurística secundaria)
    """
    cols_norm = {c: _normalize_col_name(c) for c in columns}
    mapping: Dict[str, str] = {}

    # 1) exact match
    for col_orig, col_norm in cols_norm.items():
        for std, opts_norm in SYN_NORM.items():
            if col_norm in opts_norm:
                mapping[col_orig] = std
                break

    # 2) contains match
    for col_orig, col_norm in cols_norm.items():
        if col_orig in mapping:
            continue
        for std, keys in KEYWORDS.items():
            if any(k in col_norm for k in keys):
                mapping[col_orig] = std
                break

    # reverse (último gana si hay colisiones)
    reverse: Dict[str, str] = {}
    for orig, std in mapping.items():
        reverse[std] = orig

    return mapping, reverse


# =============================================================================
# Utils: parseo numérico robusto (ES/US mixto)
# =============================================================================
_CURRENCY_RE = re.compile(r"[€$£]|eur|euro", flags=re.IGNORECASE)


def parse_number_series(s: pd.Series) -> pd.Series:
    """
    Convierte una serie a numérico soportando:
      - "12,50"
      - "1.234,56"
      - "€ 12,50"
      - "1,234.56"

    Estrategia robusta:
      - limpiar moneda y espacios
      - si contiene coma y punto: decidir según el ÚLTIMO separador
      - si solo coma: coma decimal
      - si solo punto: punto decimal
    """
    if s is None:
        return pd.Series(dtype="float64")

    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce")

    x = s.astype(str).str.strip()
    x = x.str.replace(_CURRENCY_RE, "", regex=True)
    x = x.str.replace("\u00A0", "", regex=False)
    x = x.str.replace(r"\s+", "", regex=True)

    has_comma = x.str.contains(",", na=False)
    has_dot = x.str.contains(r"\.", na=False)

    # Si hay ambos, decidimos por el último separador:
    # ES: "1.234,56" => última coma después del punto => quitar puntos, coma->punto
    # US: "1,234.56" => último punto después de la coma => quitar comas
    both = has_comma & has_dot
    if both.any():
        last_comma = x.str.rfind(",")
        last_dot = x.str.rfind(".")
        es_mask = both & (last_comma > last_dot)
        us_mask = both & ~es_mask

        x = x.where(~es_mask, x.str.replace(".", "", regex=False).str.replace(",", ".", regex=False))
        x = x.where(~us_mask, x.str.replace(",", "", regex=False))

    # Solo coma => coma decimal
    only_comma = has_comma & ~has_dot
    x = x.where(~only_comma, x.str.replace(",", ".", regex=False))

    return pd.to_numeric(x, errors="coerce")


# =============================================================================
# IO
# =============================================================================
def find_latest_input(input_dir: Path) -> Path:
    """Devuelve el archivo más reciente en input_dir (xlsx/xls/csv)."""
    if not input_dir.exists():
        raise FileNotFoundError(f"No existe la carpeta de entrada: {input_dir}")

    files: List[Path] = []
    for ext in ("*.xlsx", "*.xls", "*.csv"):
        files.extend(input_dir.glob(ext))

    if not files:
        raise FileNotFoundError(f"No hay archivos .xlsx/.xls/.csv en: {input_dir}")

    return sorted(files, key=lambda p: p.stat().st_mtime, reverse=True)[0]


def load_input_file(path: Path, sheet_name: Optional[str] = None) -> Tuple[pd.DataFrame, dict]:
    """
    Carga un archivo de entrada (xlsx/xls/csv) con heurísticas robustas.
    Devuelve (df, meta_read).
    """
    if not path.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {path}")

    meta_read = {"path": str(path), "kind": path.suffix.lower(), "sheet_used": None, "csv_sep": None, "csv_encoding": None}

    if path.suffix.lower() == ".csv":
        # Intentos robustos para CSV: encodings y separadores
        encodings = ["utf-8", "utf-8-sig", "latin-1"]
        for enc in encodings:
            try:
                df = pd.read_csv(path, sep=None, engine="python", encoding=enc)
                meta_read["csv_encoding"] = enc
                meta_read["csv_sep"] = "auto(engine=python)"
                return df, meta_read
            except Exception:
                pass

        # Fallback: sep ; con latin-1
        df = pd.read_csv(path, sep=";", encoding="latin-1")
        meta_read["csv_encoding"] = "latin-1"
        meta_read["csv_sep"] = ";"
        return df, meta_read

    # Excel
    xl = pd.ExcelFile(path)
    if sheet_name and sheet_name in xl.sheet_names:
        meta_read["sheet_used"] = sheet_name
        return xl.parse(sheet_name=sheet_name), meta_read

    # Elegir la hoja más “razonable”: no vacía y con >=2 cols
    for sh in xl.sheet_names:
        df_try = xl.parse(sheet_name=sh)
        if df_try is not None and not df_try.empty and len(df_try.columns) >= 2:
            meta_read["sheet_used"] = sh
            return df_try, meta_read

    meta_read["sheet_used"] = xl.sheet_names[0]
    return xl.parse(sheet_name=xl.sheet_names[0]), meta_read


# =============================================================================
# Core: normalización y limpieza
# =============================================================================
def normalize_dataframe(df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    """
    Renombra columnas al esquema interno (fecha, producto, cantidad, precio_unitario, importe_total).
    Devuelve:
      - df_normalizado (con columnas internas presentes; algunas pueden ser NaN si no existían)
      - meta_mapping (trazabilidad del mapeo)
    """
    mapping, reverse = detect_column_mapping(list(df.columns))
    df2 = df.rename(columns=mapping).copy()

    # Asegurar columnas internas (rellenar si faltan)
    for col in ESQUEMA_INTERNO:
        if col not in df2.columns:
            df2[col] = pd.NA

    # Recortar al esquema interno
    df2 = df2[ESQUEMA_INTERNO].copy()

    meta = {
        "mapping_detected": mapping,
        "mapping_reverse": reverse,
        "original_columns": list(df.columns),
        "normalized_columns": list(df2.columns),
    }
    return df2, meta


def _infer_price_mode_from_headers(mapping_reverse: Dict[str, str]) -> str:
    """
    Usa el nombre original de la columna mapeada para decidir si es unitario o total.
    Devuelve: "line_total" | "unit_price" | "unknown"
    """
    # Columna original para cada estándar (si existe)
    orig_unit = mapping_reverse.get("precio_unitario", "")
    orig_total = mapping_reverse.get("importe_total", "")

    def n(x: str) -> str:
        return _normalize_col_name(x)

    u = n(orig_unit)
    t = n(orig_total)

    # Si explícitamente tenemos total -> line_total
    if any(k in t for k in ["total", "importe total", "subtotal", "neto", "amount", "line total"]):
        return "line_total"

    # Si explícitamente unitario -> unit_price
    if any(k in u for k in ["unit", "unitario", "pvp", "precio unitario"]):
        return "unit_price"

    return "unknown"


def _corr_is_total_like(df: pd.DataFrame) -> bool:
    """
    Heurística estadística: si el precio/importe correlaciona fuertemente con cantidad,
    suele ser total de línea (crece con cantidad). Si no, suele ser unitario.
    """
    try:
        cand = df[["cantidad", "_precio_raw"]].dropna()
        cand = cand[(cand["cantidad"] > 0) & (cand["_precio_raw"] >= 0)]
        if len(cand) < 30:
            return False
        corr = cand["cantidad"].corr(cand["_precio_raw"])
        return bool(corr is not None and corr >= 0.80)
    except Exception:
        return False


def clean_and_compute_revenue(df: pd.DataFrame, meta_mapping: dict) -> Tuple[pd.DataFrame, dict]:
    """
    Limpieza + cálculo de revenue robusto.

    Reglas:
      - fecha parseable
      - producto no vacío
      - cantidad > 0
      - precio_unitario e importe_total parseados a numérico
      - revenue:
          si importe_total tiene valores válidos -> revenue = importe_total
          elif precio_unitario tiene valores válidos -> revenue = cantidad * precio_unitario
          else -> error

    Devuelve (df_clean, meta_clean)
    """
    stats = {
        "rows_in": int(len(df)),
        "rows_dropped_all_nan": 0,
        "rows_dropped_na_core": 0,
        "rows_dropped_rules": 0,
        "price_mode": "unknown",  # "line_total" | "unit_price" | "mixed/unknown"
        "revenue_source": "unknown",  # "importe_total" | "precio_unitario*cantidad"
        "sanity": {},
    }

    df0 = df.dropna(how="all").copy()
    stats["rows_dropped_all_nan"] = int(len(df) - len(df0))

    # fecha
    df0["fecha"] = pd.to_datetime(df0["fecha"], errors="coerce")

    # producto
    df0["producto"] = df0["producto"].fillna("-").astype(str).str.strip()
    df0.loc[df0["producto"].eq(""), "producto"] = "-"

    # numéricos
    df0["cantidad"] = parse_number_series(df0["cantidad"]).astype(float)
    df0["precio_unitario"] = parse_number_series(df0["precio_unitario"]).astype(float)
    df0["importe_total"] = parse_number_series(df0["importe_total"]).astype(float)

    before = len(df0)
    df0 = df0.dropna(subset=["fecha", "producto", "cantidad"])
    stats["rows_dropped_na_core"] = int(before - len(df0))

    before2 = len(df0)
    df0 = df0[df0["cantidad"] > 0]
    df0 = df0[df0["producto"].str.len() > 0]
    stats["rows_dropped_rules"] = int(before2 - len(df0))

    if df0.empty:
        return df0, stats

    # Decide modo preferido por headers + heurística
    reverse = meta_mapping.get("mapping_reverse") or {}
    header_mode = _infer_price_mode_from_headers(reverse)

    # Para heurística correlación, usamos una columna raw candidata:
    # - si hay importe_total con datos, la usamos
    # - si no, usamos precio_unitario
    df0["_precio_raw"] = df0["importe_total"]
    if df0["_precio_raw"].isna().all() or (df0["_precio_raw"] == 0).all():
        df0["_precio_raw"] = df0["precio_unitario"]

    corr_total_like = _corr_is_total_like(df0)

    # Construcción revenue:
    # 1) Si importe_total tiene suficientes valores válidos, prioriza total
    importe_valid = df0["importe_total"].notna() & (df0["importe_total"] >= 0)
    unit_valid = df0["precio_unitario"].notna() & (df0["precio_unitario"] >= 0)

    has_importe = int(importe_valid.sum())
    has_unit = int(unit_valid.sum())

    # Política:
    # - Si header dice line_total -> usa importe_total si existe; si no, fallback unit
    # - Si header dice unit_price -> usa unit si existe; si no, fallback importe
    # - Si unknown:
    #     - si importe_total tiene cobertura alta -> usa importe_total
    #     - elif corr_total_like True -> usa importe_total si existe
    #     - else usa unit_price si existe
    if header_mode == "line_total":
        if has_importe > 0:
            df0["revenue"] = df0["importe_total"].clip(lower=0.0)
            stats["price_mode"] = "line_total"
            stats["revenue_source"] = "importe_total"
        elif has_unit > 0:
            df0["revenue"] = (df0["cantidad"] * df0["precio_unitario"]).clip(lower=0.0)
            stats["price_mode"] = "unit_price"
            stats["revenue_source"] = "precio_unitario*cantidad"
        else:
            raise ValueError("No se pudo calcular revenue: no hay importe_total ni precio_unitario válidos.")
    elif header_mode == "unit_price":
        if has_unit > 0:
            df0["revenue"] = (df0["cantidad"] * df0["precio_unitario"]).clip(lower=0.0)
            stats["price_mode"] = "unit_price"
            stats["revenue_source"] = "precio_unitario*cantidad"
        elif has_importe > 0:
            df0["revenue"] = df0["importe_total"].clip(lower=0.0)
            stats["price_mode"] = "line_total"
            stats["revenue_source"] = "importe_total"
        else:
            raise ValueError("No se pudo calcular revenue: no hay precio_unitario ni importe_total válidos.")
    else:
        # unknown
        # cobertura alta de importe_total -> total
        coverage_importe = has_importe / max(len(df0), 1)
        if has_importe > 0 and coverage_importe >= 0.70:
            df0["revenue"] = df0["importe_total"].clip(lower=0.0)
            stats["price_mode"] = "line_total"
            stats["revenue_source"] = "importe_total"
        elif corr_total_like and has_importe > 0:
            df0["revenue"] = df0["importe_total"].clip(lower=0.0)
            stats["price_mode"] = "line_total"
            stats["revenue_source"] = "importe_total"
        elif has_unit > 0:
            df0["revenue"] = (df0["cantidad"] * df0["precio_unitario"]).clip(lower=0.0)
            stats["price_mode"] = "unit_price"
            stats["revenue_source"] = "precio_unitario*cantidad"
        elif has_importe > 0:
            df0["revenue"] = df0["importe_total"].clip(lower=0.0)
            stats["price_mode"] = "line_total"
            stats["revenue_source"] = "importe_total"
        else:
            raise ValueError("No se pudo calcular revenue: no hay importe_total ni precio_unitario válidos.")

    # Sanity checks para metadata (no filtra, solo informa)
    def _q(s: pd.Series, qs=(0.05, 0.50, 0.95)) -> dict:
        try:
            s2 = s.dropna()
            if s2.empty:
                return {"min": None, "p05": None, "p50": None, "p95": None, "max": None}
            return {
                "min": float(s2.min()),
                "p05": float(s2.quantile(qs[0])),
                "p50": float(s2.quantile(qs[1])),
                "p95": float(s2.quantile(qs[2])),
                "max": float(s2.max()),
            }
        except Exception:
            return {"min": None, "p05": None, "p50": None, "p95": None, "max": None}

    stats["sanity"] = {
        "cantidad": _q(df0["cantidad"]),
        "precio_unitario": _q(df0["precio_unitario"]),
        "importe_total": _q(df0["importe_total"]),
        "revenue": _q(df0["revenue"]),
    }

    # Tipos finales
    df0["cantidad"] = df0["cantidad"].astype(float)
    df0["precio_unitario"] = df0["precio_unitario"].astype(float)
    df0["importe_total"] = df0["importe_total"].astype(float)
    df0["revenue"] = df0["revenue"].astype(float)

    # ordenar
    df0 = df0.sort_values("fecha")

    # cleanup auxiliares
    df0 = df0.drop(columns=["_precio_raw"], errors="ignore")

    return df0, stats


# =============================================================================
# KPIs simples del pipeline (solo informativos)
# =============================================================================
def calculate_kpis(df: pd.DataFrame) -> pd.DataFrame:
    """KPIs básicos del negocio (para metadata/quick check)."""
    if df is None or df.empty:
        return pd.DataFrame([{
            "revenue_total": 0.0,
            "ordenes_totales_aprox": 0,
            "ticket_medio_aprox": 0.0,
            "unidades_vendidas": 0.0,
            "productos_unicos": 0,
            "fecha_inicio": pd.NaT,
            "fecha_fin": pd.NaT,
        }])

    return pd.DataFrame([{
        "revenue_total": float(df["revenue"].sum()),
        "ordenes_totales_aprox": int(len(df)),
        "ticket_medio_aprox": float(df["revenue"].mean()) if len(df) else 0.0,
        "unidades_vendidas": float(df["cantidad"].sum()),
        "productos_unicos": int(df["producto"].nunique()),
        "fecha_inicio": df["fecha"].min(),
        "fecha_fin": df["fecha"].max(),
    }])


# =============================================================================
# Outputs
# =============================================================================
def save_outputs(df_clean: pd.DataFrame, df_kpis: pd.DataFrame, meta: dict, paths: PipelinePaths) -> None:
    """Guarda outputs del pipeline."""
    paths.processed_dir.mkdir(parents=True, exist_ok=True)

    # Output principal: deja columnas mínimas + trazables
    # (la app usa fecha, producto, cantidad, revenue)
    keep_cols = ["fecha", "producto", "cantidad", "revenue", "precio_unitario", "importe_total"]
    out_df = df_clean[keep_cols].copy()

    out_df.to_parquet(paths.output_clean, index=False)
    df_kpis.to_parquet(paths.output_kpis, index=False)

    paths.output_meta.parent.mkdir(parents=True, exist_ok=True)
    with open(paths.output_meta, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2, default=str)


# =============================================================================
# Run
# =============================================================================
def run_pipeline(paths: PipelinePaths = PipelinePaths(), sheet_name: Optional[str] = None) -> None:
    """Ejecuta pipeline completo."""
    input_path = paths.input_file or find_latest_input(paths.input_dir)

    df_raw, meta_read = load_input_file(input_path, sheet_name=sheet_name)

    df_norm, meta_mapping = normalize_dataframe(df_raw)

    df_clean, stats = clean_and_compute_revenue(df_norm, meta_mapping)

    if df_clean is None or df_clean.empty:
        raise ValueError(
            "Tras limpiar, no quedaron filas válidas.\n"
            "Revisa fechas/números o envíame una muestra del Excel/CSV para ajustar reglas."
        )

    df_kpis = calculate_kpis(df_clean)

    meta = {
        "input": meta_read,
        "rows_raw": int(len(df_raw)),
        "rows_norm": int(len(df_norm)),
        "rows_clean": int(len(df_clean)),
        "mapping": meta_mapping,
        "stats": stats,
        "date_min": str(df_clean["fecha"].min()),
        "date_max": str(df_clean["fecha"].max()),
        "columns_out": list(df_clean.columns),
    }

    save_outputs(df_clean, df_kpis, meta, paths)

    print("✅ Pipeline ejecutado correctamente.")
    print(f"📄 Input: {input_path}")
    print(f"🧭 price_mode: {stats.get('price_mode')}")
    print(f"🧮 revenue_source: {stats.get('revenue_source')}")
    print(df_kpis.to_string(index=False))


if __name__ == "__main__":
    run_pipeline()