from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import re
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


# =============================================================================
# OBJETIVO DEL PIPELINE (v2 más robusta)
# - Aceptar Excels/CSVs reales con cabeceras imperfectas y estructuras variadas.
# - Normalizar a un esquema interno TRAZABLE.
# - Dejar claro cuándo una métrica es exacta y cuándo es aproximada.
# - Evitar lecturas engañosas si faltan columnas clave o la señal es débil.
# - Generar metadata útil para debug, calidad de datos y venta.
#
# CAMBIOS IMPORTANTES RESPECTO A LA VERSIÓN ANTERIOR
# 1) Mejor metadata sobre exactitud vs aproximación.
# 2) Validación más explícita de columnas clave.
# 3) Más trazabilidad del matching de columnas.
# 4) Señales de fiabilidad de granularidad / ticket / revenue.
# 5) Mejor soporte para Excels reales con filas vacías, hojas raras o columnas ambiguas.
# =============================================================================


# ----------------------------
# Config: esquema interno
# ----------------------------
ESQUEMA_INTERNO = [
    "fecha",
    "producto",
    "cantidad",
    "precio_unitario",
    "importe_total",
    "ticket_id",  # opcional; si no existe se rellena NA
]  # revenue se calcula

PLACEHOLDER_PRODUCT = "SIN_PRODUCTO"


# ----------------------------
# Sinónimos y keywords
# ----------------------------
SINONIMOS = {
    "fecha": [
        "fecha", "dia", "día", "fecha venta", "fecha_venta", "fecha de venta",
        "date", "fecha pedido", "fecha_pedido", "created_at", "timestamp",
        "fecha factura", "fecha_factura", "emitido", "issued_at",
    ],
    "producto": [
        "producto", "producto/servicio", "servicio", "articulo", "artículo", "item",
        "nombre producto", "nombre_producto", "descripcion", "descripción", "product",
        "concepto", "detalle", "producto descripcion", "desc",
    ],
    "cantidad": [
        "cantidad", "uds", "unidades", "unidad", "qty", "quantity", "n", "numero", "número",
        "cant", "q", "cantidad vendida", "uds vendidas",
    ],
    "precio_unitario": [
        "precio unitario", "precio_unitario", "unit price", "unit_price", "pvp",
        "precio", "price", "precio uds", "precio por unidad", "importe unitario",
    ],
    "importe_total": [
        "importe", "importe total", "importe_total", "total", "total linea", "total línea",
        "amount", "line total", "line_total", "total (€)", "importe (€)", "subtotal", "neto",
    ],
    "ticket_id": [
        "ticket", "ticket id", "ticket_id", "id ticket", "numero ticket", "nº ticket", "nro ticket",
        "factura", "num factura", "nº factura", "invoice", "invoice id", "invoice_id",
        "pedido", "order", "order id", "order_id", "id pedido", "receipt", "receipt id",
    ],
}

KEYWORDS = {
    "fecha": ["fecha", "dia", "date", "created", "timestamp", "issued", "emitido"],
    "producto": ["producto", "servicio", "articulo", "item", "descripcion", "concepto", "detalle", "product", "desc"],
    "cantidad": ["cantidad", "uds", "unidades", "qty", "quantity", "numero", "cant"],
    "precio_unitario": ["unit", "unitario", "pvp", "precio", "price"],
    "importe_total": ["importe", "total", "amount", "subtotal", "neto", "€", "eur"],
    "ticket_id": ["ticket", "factura", "invoice", "pedido", "order", "receipt"],
}


@dataclass(frozen=True)
class PipelinePaths:
    """Rutas utilizadas por el pipeline."""

    input_dir: Path = Path("data/input")
    processed_dir: Path = Path("data/processed")
    input_file: Optional[Path] = None
    output_clean: Path = Path("data/processed/ventas_limpias.parquet")
    output_kpis: Path = Path("data/processed/kpis.parquet")
    output_meta: Path = Path("data/processed/metadata.json")

    allow_negative_revenue: bool = False
    dayfirst: bool = True

    @staticmethod
    def for_client(client_id: str) -> "PipelinePaths":
        base = Path("data/clients") / str(client_id)
        input_dir = base / "input"
        processed_dir = base / "processed"
        return PipelinePaths(
            input_dir=input_dir,
            processed_dir=processed_dir,
            input_file=None,
            output_clean=processed_dir / "ventas_limpias.parquet",
            output_kpis=processed_dir / "kpis.parquet",
            output_meta=processed_dir / "metadata.json",
        )


# =============================================================================
# Utils: normalización de columnas
# =============================================================================
def _normalize_col_name(name: str) -> str:
    """Normaliza nombres de columnas para facilitar matching."""
    name = str(name).strip().lower()
    name = name.replace("_", " ")
    name = re.sub(r"\s+", " ", name)
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


# =============================================================================
# Parseo numérico robusto (se usa también en scoring)
# =============================================================================
_CURRENCY_RE = re.compile(r"[€$£]|eur|euro", flags=re.IGNORECASE)


def parse_number_series(s: pd.Series) -> pd.Series:
    """
    Convierte una serie a numérico soportando:
      - 12,50
      - 1.234,56
      - € 12,50
      - 1,234.56
      - (1.234,56) -> -1234.56
    """
    if s is None:
        return pd.Series(dtype="float64")

    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce")

    x = s.astype(str).str.strip()
    x = x.str.replace(_CURRENCY_RE, "", regex=True)
    x = x.str.replace("\u00A0", "", regex=False)
    x = x.str.replace(r"\s+", "", regex=True)

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

    out = pd.to_numeric(x, errors="coerce")

    if neg_mask.any():
        out = out.where(~neg_mask, -out)

    return out


# =============================================================================
# Matching de columnas
# =============================================================================
def _detect_candidates(columns: List[str]) -> Dict[str, List[str]]:
    """Devuelve candidatos por estándar: {std: [col_orig,...]}."""
    cols_norm = {c: _normalize_col_name(c) for c in columns}
    cand: Dict[str, List[str]] = {k: [] for k in SYN_NORM.keys()}

    # 1) exact match contra sinónimos
    for col_orig, col_norm in cols_norm.items():
        for std, opts_norm in SYN_NORM.items():
            if col_norm in opts_norm:
                cand[std].append(col_orig)

    # 2) contains match con keywords solo si no estaba ya
    for col_orig, col_norm in cols_norm.items():
        for std, keys in KEYWORDS.items():
            if col_orig in cand.get(std, []):
                continue
            if any(k in col_norm for k in keys):
                cand.setdefault(std, []).append(col_orig)

    for std in list(cand.keys()):
        seen = set()
        out = []
        for c in cand[std]:
            if c not in seen:
                seen.add(c)
                out.append(c)
        cand[std] = out

    return cand



def _score_candidate(df_raw: pd.DataFrame, col: str, std: str, dayfirst: bool) -> float:
    """Score heurístico para resolver colisiones (más alto = mejor)."""
    if col not in df_raw.columns:
        return -1.0

    s = df_raw[col]

    if std == "fecha":
        dt = pd.to_datetime(s, errors="coerce", dayfirst=dayfirst)
        ok = float(dt.notna().mean())
        var = float(dt.dropna().nunique() / max(len(dt.dropna()), 1))
        return 0.85 * ok + 0.15 * min(1.0, var * 10)

    if std in ("cantidad", "precio_unitario", "importe_total"):
        num = parse_number_series(s)
        ok = float(num.notna().mean())
        nonzero = float((num.fillna(0) != 0).mean())
        try:
            var = float(num.dropna().var()) if num.dropna().shape[0] > 1 else 0.0
        except Exception:
            var = 0.0
        var_score = 1.0 if var > 0 else 0.2
        return 0.70 * ok + 0.20 * nonzero + 0.10 * var_score

    if std == "producto":
        x = s.astype(str).fillna("").str.strip()
        ok = float((x != "").mean())
        avg_len = float(x.map(len).mean()) if len(x) else 0.0
        uniq = float(x.nunique() / max(len(x), 1))
        return 0.60 * ok + 0.20 * min(1.0, avg_len / 10.0) + 0.20 * min(1.0, uniq * 10)

    if std == "ticket_id":
        x = s.astype(str).fillna("").str.strip()
        ok = float((x != "").mean())
        uniq = float(x.nunique() / max(len(x), 1))
        return 0.55 * ok + 0.45 * min(1.0, uniq * 10)

    return 0.0



def detect_column_mapping(
    df_raw: pd.DataFrame,
    dayfirst: bool = True,
) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, List[str]], Dict[str, Any]]:
    """
    Detecta mapeo {col_original -> col_estandar} y devuelve:
      - mapping_detected
      - mapping_reverse (std -> orig)
      - collisions (std -> [candidatos])
      - diagnostics por estándar (scores y candidatos)
    """
    candidates = _detect_candidates(list(df_raw.columns))
    reverse: Dict[str, str] = {}
    collisions: Dict[str, List[str]] = {}
    diagnostics: Dict[str, Any] = {}

    for std, cols in candidates.items():
        diagnostics[std] = {
            "candidates": cols[:],
            "selected": None,
            "scores": {},
        }

        if not cols:
            continue

        if len(cols) == 1:
            reverse[std] = cols[0]
            diagnostics[std]["selected"] = cols[0]
            diagnostics[std]["scores"] = {
                cols[0]: round(_score_candidate(df_raw, cols[0], std, dayfirst=dayfirst), 4)
            }
            continue

        collisions[std] = cols[:]
        scored = [(c, _score_candidate(df_raw, c, std, dayfirst=dayfirst)) for c in cols]
        scored.sort(key=lambda x: x[1], reverse=True)
        reverse[std] = scored[0][0]
        diagnostics[std]["selected"] = scored[0][0]
        diagnostics[std]["scores"] = {c: round(float(s), 4) for c, s in scored}

    mapping: Dict[str, str] = {}
    for std, orig in reverse.items():
        mapping[orig] = std

    return mapping, reverse, collisions, diagnostics


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



def _drop_empty_edge_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Recorta filas completamente vacías al principio/final.
    Ayuda con Excels exportados con basura visual arriba o abajo.
    """
    if df is None or df.empty:
        return df
    mask = ~df.isna().all(axis=1)
    if not mask.any():
        return df.iloc[0:0].copy()
    first = mask.idxmax()
    last = mask[::-1].idxmax()
    return df.loc[first:last].copy()



def load_input_file(path: Path, sheet_name: Optional[str] = None) -> Tuple[pd.DataFrame, dict]:
    """
    Carga un archivo de entrada (xlsx/xls/csv) con heurísticas robustas.
    Devuelve (df, meta_read).
    """
    if not path.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {path}")

    meta_read = {
        "path": str(path),
        "kind": path.suffix.lower(),
        "sheet_used": None,
        "available_sheets": None,
        "csv_sep": None,
        "csv_encoding": None,
        "rows_loaded": None,
        "cols_loaded": None,
    }

    if path.suffix.lower() == ".csv":
        encodings = ["utf-8", "utf-8-sig", "latin-1"]
        for enc in encodings:
            try:
                df = pd.read_csv(path, sep=None, engine="python", encoding=enc)
                df = _drop_empty_edge_rows(df)
                meta_read["csv_encoding"] = enc
                meta_read["csv_sep"] = "auto(engine=python)"
                meta_read["rows_loaded"] = int(len(df))
                meta_read["cols_loaded"] = int(len(df.columns))
                return df, meta_read
            except Exception:
                pass

        df = pd.read_csv(path, sep=";", encoding="latin-1")
        df = _drop_empty_edge_rows(df)
        meta_read["csv_encoding"] = "latin-1"
        meta_read["csv_sep"] = ";"
        meta_read["rows_loaded"] = int(len(df))
        meta_read["cols_loaded"] = int(len(df.columns))
        return df, meta_read

    xl = pd.ExcelFile(path)
    meta_read["available_sheets"] = xl.sheet_names[:]

    if sheet_name and sheet_name in xl.sheet_names:
        df = xl.parse(sheet_name=sheet_name)
        df = _drop_empty_edge_rows(df)
        meta_read["sheet_used"] = sheet_name
        meta_read["rows_loaded"] = int(len(df))
        meta_read["cols_loaded"] = int(len(df.columns))
        return df, meta_read

    # Elegimos la hoja más prometedora: no vacía y con al menos 2 columnas
    best_df = None
    best_sheet = None
    best_score = -1

    for sh in xl.sheet_names:
        df_try = xl.parse(sheet_name=sh)
        df_try = _drop_empty_edge_rows(df_try)
        score = (0 if df_try is None else len(df_try)) + (0 if df_try is None else len(df_try.columns) * 5)
        if df_try is not None and not df_try.empty and len(df_try.columns) >= 2 and score > best_score:
            best_df = df_try
            best_sheet = sh
            best_score = score

    if best_df is None:
        best_sheet = xl.sheet_names[0]
        best_df = _drop_empty_edge_rows(xl.parse(sheet_name=best_sheet))

    meta_read["sheet_used"] = best_sheet
    meta_read["rows_loaded"] = int(len(best_df))
    meta_read["cols_loaded"] = int(len(best_df.columns))
    return best_df, meta_read


# =============================================================================
# Core: normalización y limpieza
# =============================================================================
def normalize_dataframe(df_raw: pd.DataFrame, dayfirst: bool = True) -> Tuple[pd.DataFrame, dict]:
    """
    Renombra columnas al esquema interno.
    Devuelve df normalizado + metadata rica de matching.
    """
    mapping, reverse, collisions, diagnostics = detect_column_mapping(df_raw, dayfirst=dayfirst)
    df2 = df_raw.rename(columns=mapping).copy()

    missing_internal = []
    for col in ESQUEMA_INTERNO:
        if col not in df2.columns:
            df2[col] = pd.NA
            missing_internal.append(col)

    df2 = df2[ESQUEMA_INTERNO].copy()

    meta = {
        "mapping_detected": mapping,
        "mapping_reverse": reverse,
        "mapping_collisions": collisions,
        "mapping_diagnostics": diagnostics,
        "original_columns": list(df_raw.columns),
        "normalized_columns": list(df2.columns),
        "missing_internal_columns_filled": missing_internal,
    }
    return df2, meta


# =============================================================================
# Heurísticas de revenue / granularidad
# =============================================================================
def _infer_price_mode_from_headers(mapping_reverse: Dict[str, str]) -> str:
    """
    Usa el nombre original de la columna mapeada para decidir si es unitario o total.
    Devuelve: line_total | unit_price | unknown
    """
    orig_unit = mapping_reverse.get("precio_unitario", "")
    orig_total = mapping_reverse.get("importe_total", "")

    def n(x: str) -> str:
        return _normalize_col_name(x)

    u = n(orig_unit)
    t = n(orig_total)

    if any(k in t for k in ["total", "importe total", "subtotal", "neto", "amount", "line total"]):
        return "line_total"

    if any(k in u for k in ["unit", "unitario", "pvp", "precio unitario"]):
        return "unit_price"

    return "unknown"



def _corr_is_total_like(df: pd.DataFrame) -> bool:
    """Si precio/importe correlaciona fuerte con cantidad, suele ser total de línea."""
    try:
        cand = df[["cantidad", "_precio_raw"]].dropna()
        cand = cand[(cand["cantidad"] > 0) & (cand["_precio_raw"].notna())]
        if len(cand) < 30:
            return False
        corr = cand["cantidad"].corr(cand["_precio_raw"])
        return bool(corr is not None and corr >= 0.80)
    except Exception:
        return False



def _granularity_label(df: pd.DataFrame) -> str:
    """
    Señal simple de granularidad:
    - exacta si hay ticket_id usable
    - aproximada si no lo hay
    """
    if "ticket_id" in df.columns and df["ticket_id"].notna().any():
        non_empty = df["ticket_id"].astype(str).str.strip().ne("")
        if float(non_empty.mean()) >= 0.80:
            return "exacta"
        if float(non_empty.mean()) >= 0.30:
            return "mixta"
    return "aproximada"



def _metric_exactness_flags(df: pd.DataFrame) -> Dict[str, str]:
    """
    Explica qué métricas son exactas vs aproximadas.
    Esto es material muy útil para app y para venta honesta del producto.
    """
    gran = _granularity_label(df)

    if gran == "exacta":
        return {
            "revenue_total": "exacta",
            "unidades_vendidas": "exacta",
            "operaciones": "exacta",
            "ticket_medio": "exacta",
            "top_productos": "exacta",
            "comparacion_temporal": "exacta",
        }

    if gran == "mixta":
        return {
            "revenue_total": "exacta",
            "unidades_vendidas": "exacta",
            "operaciones": "aproximada",
            "ticket_medio": "aproximada",
            "top_productos": "exacta",
            "comparacion_temporal": "exacta",
        }

    return {
        "revenue_total": "exacta",
        "unidades_vendidas": "exacta",
        "operaciones": "aproximada",
        "ticket_medio": "aproximada",
        "top_productos": "exacta",
        "comparacion_temporal": "exacta",
    }


# =============================================================================
# Limpieza + cálculo de revenue
# =============================================================================
def clean_and_compute_revenue(
    df: pd.DataFrame,
    meta_mapping: dict,
    *,
    allow_negative_revenue: bool = False,
    dayfirst: bool = True,
) -> Tuple[pd.DataFrame, dict]:
    """
    Limpieza + cálculo de revenue robusto.

    Reglas:
      - fecha parseable
      - producto no vacío (placeholder si falta)
      - cantidad > 0
      - revenue calculado con prioridad clara
      - metadata rica sobre cobertura y fiabilidad
    """
    stats: Dict[str, Any] = {
        "rows_in": int(len(df)),
        "rows_dropped_all_nan": 0,
        "rows_dropped_na_core": 0,
        "rows_dropped_rules": 0,
        "rows_with_placeholder_producto": 0,
        "rows_with_ticket_id": 0,
        "ticket_id_coverage": 0.0,
        "price_mode": "unknown",
        "revenue_source": "unknown",
        "revenue_source_confidence": "baja",
        "granularity": "aproximada",
        "metric_exactness": {},
        "allow_negative_revenue": bool(allow_negative_revenue),
        "missing_key_input_columns": [],
        "input_column_coverage": {},
        "sanity": {},
        "warnings": [],
    }

    stats["missing_key_input_columns"] = [
        col for col in ["fecha", "producto", "cantidad"] if col not in df.columns
    ]

    for col in ["fecha", "producto", "cantidad", "precio_unitario", "importe_total", "ticket_id"]:
        if col in df.columns:
            notna_ratio = float(pd.Series(df[col]).notna().mean()) if len(df) else 0.0
            stats["input_column_coverage"][col] = round(notna_ratio, 4)
        else:
            stats["input_column_coverage"][col] = 0.0

    df0 = df.dropna(how="all").copy()
    stats["rows_dropped_all_nan"] = int(len(df) - len(df0))

    # Fecha
    df0["fecha"] = pd.to_datetime(df0["fecha"], errors="coerce", dayfirst=dayfirst)

    # Ticket id
    if "ticket_id" in df0.columns:
        ticket = df0["ticket_id"].astype(str).fillna("").str.strip()
        ticket = ticket.mask(ticket.eq(""), pd.NA)
        # Evitar strings de NA textual
        ticket = ticket.mask(ticket.astype(str).str.lower().isin(["nan", "none", "null", "<na>"]), pd.NA)
        df0["ticket_id"] = ticket

    # Producto
    prod = df0["producto"].astype(str).fillna("").str.strip()
    prod = prod.mask(prod.eq(""), PLACEHOLDER_PRODUCT)
    prod = prod.mask(prod.str.lower().isin(["nan", "none", "null", "<na>"]), PLACEHOLDER_PRODUCT)
    df0["producto"] = prod
    stats["rows_with_placeholder_producto"] = int((df0["producto"] == PLACEHOLDER_PRODUCT).sum())

    # Numéricos
    df0["cantidad"] = parse_number_series(df0["cantidad"]).astype(float)
    df0["precio_unitario"] = parse_number_series(df0["precio_unitario"]).astype(float)
    df0["importe_total"] = parse_number_series(df0["importe_total"]).astype(float)

    before = len(df0)
    df0 = df0.dropna(subset=["fecha", "producto", "cantidad"])
    stats["rows_dropped_na_core"] = int(before - len(df0))

    before2 = len(df0)
    df0 = df0[df0["cantidad"] > 0]
    df0 = df0[df0["producto"].astype(str).str.len() > 0]
    stats["rows_dropped_rules"] = int(before2 - len(df0))

    if df0.empty:
        return df0, stats

    # Granularidad y cobertura ticket
    if "ticket_id" in df0.columns:
        stats["rows_with_ticket_id"] = int(df0["ticket_id"].notna().sum())
        stats["ticket_id_coverage"] = round(float(df0["ticket_id"].notna().mean()), 4)

    reverse = meta_mapping.get("mapping_reverse") or {}
    header_mode = _infer_price_mode_from_headers(reverse)

    df0["_precio_raw"] = df0["importe_total"]
    if df0["_precio_raw"].isna().all() or (df0["_precio_raw"].fillna(0) == 0).all():
        df0["_precio_raw"] = df0["precio_unitario"]

    corr_total_like = _corr_is_total_like(df0)

    if allow_negative_revenue:
        importe_valid = df0["importe_total"].notna()
        unit_valid = df0["precio_unitario"].notna()
    else:
        importe_valid = df0["importe_total"].notna() & (df0["importe_total"] >= 0)
        unit_valid = df0["precio_unitario"].notna() & (df0["precio_unitario"] >= 0)

    has_importe = int(importe_valid.sum())
    has_unit = int(unit_valid.sum())
    coverage_importe = has_importe / max(len(df0), 1)
    coverage_unit = has_unit / max(len(df0), 1)

    stats["input_column_coverage"]["importe_total_valid"] = round(float(coverage_importe), 4)
    stats["input_column_coverage"]["precio_unitario_valid"] = round(float(coverage_unit), 4)

    def _clip_if_needed(x: pd.Series) -> pd.Series:
        return x if allow_negative_revenue else x.clip(lower=0.0)

    # Selección de source más trazable
    if header_mode == "line_total":
        if has_importe > 0:
            df0["revenue"] = _clip_if_needed(df0["importe_total"])
            stats["price_mode"] = "line_total"
            stats["revenue_source"] = "importe_total"
            stats["revenue_source_confidence"] = "alta"
        elif has_unit > 0:
            df0["revenue"] = _clip_if_needed(df0["cantidad"] * df0["precio_unitario"])
            stats["price_mode"] = "unit_price"
            stats["revenue_source"] = "precio_unitario*cantidad"
            stats["revenue_source_confidence"] = "media"
            stats["warnings"].append("header_sugeria_total_pero_se_usó_unitario")
        else:
            raise ValueError("No se pudo calcular revenue: no hay importe_total ni precio_unitario válidos.")
    elif header_mode == "unit_price":
        if has_unit > 0:
            df0["revenue"] = _clip_if_needed(df0["cantidad"] * df0["precio_unitario"])
            stats["price_mode"] = "unit_price"
            stats["revenue_source"] = "precio_unitario*cantidad"
            stats["revenue_source_confidence"] = "alta"
        elif has_importe > 0:
            df0["revenue"] = _clip_if_needed(df0["importe_total"])
            stats["price_mode"] = "line_total"
            stats["revenue_source"] = "importe_total"
            stats["revenue_source_confidence"] = "media"
            stats["warnings"].append("header_sugeria_unitario_pero_se_usó_total")
        else:
            raise ValueError("No se pudo calcular revenue: no hay precio_unitario ni importe_total válidos.")
    else:
        if has_importe > 0 and coverage_importe >= 0.70:
            df0["revenue"] = _clip_if_needed(df0["importe_total"])
            stats["price_mode"] = "line_total"
            stats["revenue_source"] = "importe_total"
            stats["revenue_source_confidence"] = "media-alta"
        elif corr_total_like and has_importe > 0:
            df0["revenue"] = _clip_if_needed(df0["importe_total"])
            stats["price_mode"] = "line_total"
            stats["revenue_source"] = "importe_total"
            stats["revenue_source_confidence"] = "media"
            stats["warnings"].append("revenue_total_inferido_por_correlacion")
        elif has_unit > 0:
            df0["revenue"] = _clip_if_needed(df0["cantidad"] * df0["precio_unitario"])
            stats["price_mode"] = "unit_price"
            stats["revenue_source"] = "precio_unitario*cantidad"
            stats["revenue_source_confidence"] = "media"
        elif has_importe > 0:
            df0["revenue"] = _clip_if_needed(df0["importe_total"])
            stats["price_mode"] = "line_total"
            stats["revenue_source"] = "importe_total"
            stats["revenue_source_confidence"] = "baja-media"
            stats["warnings"].append("revenue_total_usado_por_fallback")
        else:
            raise ValueError("No se pudo calcular revenue: no hay importe_total ni precio_unitario válidos.")

    # Sanity / trazabilidad
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

    stats["granularity"] = _granularity_label(df0)
    stats["metric_exactness"] = _metric_exactness_flags(df0)

    if stats["granularity"] != "exacta":
        stats["warnings"].append("ticket_y_operaciones_aproximadas")

    if stats["rows_with_placeholder_producto"] > 0:
        stats["warnings"].append("hay_filas_sin_producto_util")

    df0["cantidad"] = df0["cantidad"].astype(float)
    df0["precio_unitario"] = df0["precio_unitario"].astype(float)
    df0["importe_total"] = df0["importe_total"].astype(float)
    df0["revenue"] = df0["revenue"].astype(float)

    df0 = df0.sort_values("fecha")
    df0 = df0.drop(columns=["_precio_raw"], errors="ignore")

    return df0, stats


# =============================================================================
# KPIs simples del pipeline
# =============================================================================
def calculate_kpis(df: pd.DataFrame) -> pd.DataFrame:
    """KPIs básicos del negocio con flags de exactitud/aproximación."""
    if df is None or df.empty:
        return pd.DataFrame([{
            "revenue_total": 0.0,
            "ordenes_totales": 0,
            "ordenes_label": "sin_datos",
            "ticket_medio": 0.0,
            "ticket_label": "sin_datos",
            "unidades_vendidas": 0.0,
            "productos_unicos": 0,
            "tiene_ticket_id": False,
            "ticket_id_coverage": 0.0,
            "granularity": "aproximada",
            "fecha_inicio": pd.NaT,
            "fecha_fin": pd.NaT,
        }])

    ticket_cov = 0.0
    tiene_ticket_id = False
    if "ticket_id" in df.columns:
        ticket_cov = float(df["ticket_id"].notna().mean())
        tiene_ticket_id = bool(df["ticket_id"].notna().any())

    granularity = _granularity_label(df)

    if tiene_ticket_id and granularity == "exacta":
        ordenes = int(df["ticket_id"].nunique(dropna=True))
        ticket_medio = float(df.groupby("ticket_id", dropna=True)["revenue"].sum().mean()) if ordenes else 0.0
        ordenes_label = "exacta"
        ticket_label = "exacta"
    else:
        ordenes = int(len(df))
        ticket_medio = float(df["revenue"].mean()) if len(df) else 0.0
        ordenes_label = "aproximada"
        ticket_label = "aproximada"

    return pd.DataFrame([{
        "revenue_total": float(df["revenue"].sum()),
        "ordenes_totales": ordenes,
        "ordenes_label": ordenes_label,
        "ticket_medio": ticket_medio,
        "ticket_label": ticket_label,
        "unidades_vendidas": float(df["cantidad"].sum()),
        "productos_unicos": int(df["producto"].nunique()),
        "tiene_ticket_id": tiene_ticket_id,
        "ticket_id_coverage": round(ticket_cov, 4),
        "granularity": granularity,
        "fecha_inicio": df["fecha"].min(),
        "fecha_fin": df["fecha"].max(),
    }])


# =============================================================================
# Outputs
# =============================================================================
def save_outputs(df_clean: pd.DataFrame, df_kpis: pd.DataFrame, meta: dict, paths: PipelinePaths) -> None:
    """Guarda outputs del pipeline."""
    paths.processed_dir.mkdir(parents=True, exist_ok=True)

    keep_cols = ["fecha", "producto", "cantidad", "revenue", "precio_unitario", "importe_total"]
    if "ticket_id" in df_clean.columns:
        keep_cols.append("ticket_id")

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
    df_norm, meta_mapping = normalize_dataframe(df_raw, dayfirst=paths.dayfirst)

    df_clean, stats = clean_and_compute_revenue(
        df_norm,
        meta_mapping,
        allow_negative_revenue=paths.allow_negative_revenue,
        dayfirst=paths.dayfirst,
    )

    if df_clean is None or df_clean.empty:
        raise ValueError(
            "Tras limpiar, no quedaron filas válidas.\n"
            "Revisa fechas/números, columnas clave o envíame una muestra del Excel/CSV para ajustar reglas."
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
        "pipeline_notes": {
            "revenue_total": "Se considera exacto si revenue se pudo construir con cobertura suficiente.",
            "ordenes_y_ticket": "Son exactos solo cuando ticket_id tiene buena cobertura; si no, se tratan como aproximados.",
            "perfilado_y_recomendaciones": "Pierden precisión si producto tiene muchos placeholders o faltan columnas clave.",
        },
    }

    save_outputs(df_clean, df_kpis, meta, paths)

    print("✅ Pipeline ejecutado correctamente.")
    print(f"📄 Input: {input_path}")
    print(f"🧭 price_mode: {stats.get('price_mode')}")
    print(f"🧮 revenue_source: {stats.get('revenue_source')}")
    print(f"🎯 revenue_source_confidence: {stats.get('revenue_source_confidence')}")
    print(f"📏 granularity: {stats.get('granularity')}")
    print(f"🧾 allow_negative_revenue: {stats.get('allow_negative_revenue')}")
    print(df_kpis.to_string(index=False))


if __name__ == "__main__":
    run_pipeline()