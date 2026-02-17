"""
Pipeline de datos para dashboard automático (MVP robusto).

Objetivo:
- Recibir Excels "en bruto" de clientes (columnas variables y formatos mixtos)
- Normalizar columnas a un esquema interno estándar
- Limpiar, calcular KPIs y guardar resultados en /data/processed
- Generar metadata para trazabilidad (qué detectamos, qué mapeamos, qué descartamos)

Salida:
- data/processed/ventas_limpias.parquet
- data/processed/kpis.parquet
- data/processed/metadata.json
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import re
from typing import Dict, List, Optional, Tuple

import pandas as pd


# ----------------------------
# Config
# ----------------------------
ESQUEMA = ["fecha", "producto", "cantidad", "precio"]  # estándar interno

# Sinónimos: amplia con el tiempo (pero ya cubre la mayoría)
SINONIMOS = {
    "fecha": [
        "fecha", "dia", "día", "fecha venta", "fecha_venta", "fecha de venta",
        "date", "fecha pedido", "fecha_pedido", "created_at", "timestamp"
    ],
    "producto": [
        "producto", "producto/servicio", "servicio", "articulo", "artículo", "item",
        "nombre producto", "nombre_producto", "descripcion", "descripción", "product",
        "concepto", "detalle"
    ],
    "cantidad": [
        "cantidad", "uds", "unidades", "unidad", "qty", "quantity", "n", "numero", "número",
        "cant", "q"
    ],
    # Nota: aquí mezclamos unitario y total (luego detectamos cuál es cuál)
    "precio": [
        "precio", "precio unitario", "precio_unitario", "importe", "total", "importe total",
        "valor", "price", "amount", "importe (€)", "total (€)"
    ],
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


# ----------------------------
# Utils: normalización columnas
# ----------------------------
def _normalize_col_name(name: str) -> str:
    """
    Normaliza nombres de columnas para facilitar matching:
    - minúsculas
    - quita acentos
    - reemplaza _ por espacio
    - colapsa espacios
    """
    name = str(name).strip().lower()
    name = name.replace("_", " ")
    name = re.sub(r"\s+", " ", name)

    # quitar acentos manualmente (simple y suficiente)
    name = (
        name.replace("á", "a").replace("é", "e").replace("í", "i")
        .replace("ó", "o").replace("ú", "u").replace("ñ", "n")
    )
    return name


def _build_synonyms_norm() -> Dict[str, List[str]]:
    """Precalcula sinónimos normalizados."""
    out = {}
    for k, vals in SINONIMOS.items():
        out[k] = [_normalize_col_name(v) for v in vals]
    return out


SYN_NORM = _build_synonyms_norm()


def detect_column_mapping(columns: List[str]) -> Dict[str, str]:
    """
    Detecta mapeo {col_original -> col_estandar}.
    Matching:
      1) exact match vs sinónimos normalizados
      2) si no hay exact, contiene palabra clave (heurística simple)
    """
    cols_norm = {c: _normalize_col_name(c) for c in columns}
    mapping: Dict[str, str] = {}

    # 1) exact match
    for col_orig, col_norm in cols_norm.items():
        for std, opts_norm in SYN_NORM.items():
            if col_norm in opts_norm:
                mapping[col_orig] = std
                break

    # 2) contains match (si no se mapeó aún)
    # Ej: "importe (€)" contiene "importe"
    keywords = {
        "fecha": ["fecha", "dia", "date", "created"],
        "producto": ["producto", "servicio", "articulo", "item", "descripcion", "concepto", "detalle", "product"],
        "cantidad": ["cantidad", "uds", "unidades", "qty", "quantity", "numero", "cant"],
        "precio": ["precio", "importe", "total", "amount", "valor", "€", "eur"],
    }

    for col_orig, col_norm in cols_norm.items():
        if col_orig in mapping:
            continue
        for std, keys in keywords.items():
            if any(k in col_norm for k in keys):
                mapping[col_orig] = std
                break

    return mapping


# ----------------------------
# Utils: parseo números ES/mixto
# ----------------------------
_CURRENCY_RE = re.compile(r"[€$£]|eur|euro", flags=re.IGNORECASE)


def parse_number_series(s: pd.Series) -> pd.Series:
    """
    Convierte una serie a numérico soportando:
      - "12,50"
      - "1.234,56"
      - "€ 12,50"
      - "1,234.56" (a veces exportan en US)
    Estrategia:
      - eliminar símbolos moneda y espacios
      - detectar patrón dominante (coma decimal vs punto decimal)
      - convertir
    """
    if s is None or s.empty:
        return pd.to_numeric(s, errors="coerce")

    x = s.astype(str).str.strip()
    x = x.str.replace(_CURRENCY_RE, "", regex=True)
    x = x.str.replace(r"\s+", "", regex=True)

    # Heurística: si hay muchas comas y también puntos -> probablemente ES (1.234,56)
    sample = x.head(200).tolist()
    has_comma = sum("," in v for v in sample)
    has_dot = sum("." in v for v in sample)

    # Caso típico ES: miles con '.' y decimales con ','
    if has_comma > 0 and has_dot > 0:
        x = x.str.replace(".", "", regex=False)      # quita miles
        x = x.str.replace(",", ".", regex=False)     # coma decimal -> punto
        return pd.to_numeric(x, errors="coerce")

    # Caso: solo comas -> asume coma decimal
    if has_comma > 0 and has_dot == 0:
        x = x.str.replace(",", ".", regex=False)
        return pd.to_numeric(x, errors="coerce")

    # Caso: solo puntos o nada -> to_numeric directo
    return pd.to_numeric(x, errors="coerce")


# ----------------------------
# IO Excel
# ----------------------------
def find_latest_excel(input_dir: Path) -> Path:
    """Devuelve el Excel más reciente en input_dir (xlsx/xls/csv)."""
    if not input_dir.exists():
        raise FileNotFoundError(f"No existe la carpeta de entrada: {input_dir}")

    files = []
    for ext in ("*.xlsx", "*.xls", "*.csv"):
        files.extend(input_dir.glob(ext))

    if not files:
        raise FileNotFoundError(f"No hay archivos .xlsx/.xls/.csv en: {input_dir}")

    files = sorted(files, key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0]


def load_input_file(path: Path, sheet_name: Optional[str] = None) -> pd.DataFrame:
    """
    Carga un archivo de entrada (xlsx/xls/csv).
    Para Excel: lee sheet_name si se indica, si no la primera hoja con datos.
    """
    if not path.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {path}")

    if path.suffix.lower() == ".csv":
        # CSV: intenta separadores comunes
        try:
            return pd.read_csv(path)
        except Exception:
            return pd.read_csv(path, sep=";")

    # Excel
    xl = pd.ExcelFile(path)
    if sheet_name and sheet_name in xl.sheet_names:
        return xl.parse(sheet_name=sheet_name)

    # Si no se especifica hoja: elegir la primera que tenga columnas razonables
    for sh in xl.sheet_names:
        df_try = xl.parse(sheet_name=sh)
        if df_try is not None and not df_try.empty and len(df_try.columns) >= 2:
            return df_try

    # fallback
    return xl.parse(sheet_name=xl.sheet_names[0])


# ----------------------------
# Core pipeline steps
# ----------------------------
def normalize_dataframe(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Renombra columnas al esquema estándar.
    Devuelve: (df_normalizado, mapping_detectado)
    """
    mapping = detect_column_mapping(list(df.columns))
    df2 = df.rename(columns=mapping).copy()

    missing = [c for c in ESQUEMA if c not in df2.columns]
    if missing:
        # Error útil y claro
        raise ValueError(
            "El Excel no tiene columnas suficientes para procesarlo.\n"
            f"Faltan: {missing}\n"
            f"Columnas originales: {list(df.columns)}\n"
            f"Columnas tras mapeo: {list(df2.columns)}\n"
            f"Mapeo detectado: {mapping}\n"
            "Sugerencia: renombra columnas o dime cómo se llaman y lo adaptamos."
        )

    df2 = df2[ESQUEMA].copy()
    return df2, mapping


def clean_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Limpieza + cálculo revenue robusto.
    También intenta detectar si 'precio' parece total de línea.
    Devuelve: (df_limpio, stats)
    """
    stats = {
        "rows_in": float(len(df)),
        "rows_dropped_na": 0.0,
        "rows_dropped_rules": 0.0,
        "precio_as_line_total": 0.0,  # 1.0 si detectamos que precio es total, 0.0 si unitario
    }

    df = df.dropna(how="all").copy()

    # fecha
    df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")

    # producto
    df["producto"] = df["producto"].astype(str).str.strip()

    # cantidad / precio numérico con soporte ES
    df["cantidad"] = parse_number_series(df["cantidad"])
    df["precio"] = parse_number_series(df["precio"])

    before = len(df)
    df = df.dropna(subset=["fecha", "producto", "cantidad", "precio"])
    stats["rows_dropped_na"] = float(before - len(df))

    # reglas mínimas
    before2 = len(df)
    df = df[df["cantidad"] > 0]
    df = df[df["precio"] >= 0]
    df = df[df["producto"].str.len() > 0]
    stats["rows_dropped_rules"] = float(before2 - len(df))

    if df.empty:
        return df, stats

    # Detectar si "precio" parece total de línea:
    # Heurística: si hay muchas filas con cantidad != 1 y el "precio" parece grande
    # y además (precio / cantidad) da valores raros (muchos decimales o muy bajos),
    # entonces probablemente es total.
    # Esta heurística es conservadora: si duda, asume unitario.
    cand = df[df["cantidad"] != 1].copy()
    if len(cand) >= max(20, int(0.1 * len(df))):
        ratio = (cand["precio"] / cand["cantidad"]).replace([pd.NA, pd.NaT, float("inf")], pd.NA).dropna()
        # si el ratio tiene una dispersión extrema o muchos ratios muy pequeños, huele a total
        if not ratio.empty:
            q10, q90 = ratio.quantile(0.10), ratio.quantile(0.90)
            # condición conservadora
            if q10 <= 1 and q90 <= 10:
                stats["precio_as_line_total"] = 1.0

    if stats["precio_as_line_total"] == 1.0:
        # precio ya es total por línea
        df["revenue"] = df["precio"]
        # opcional: guardar unitario estimado
        df["precio_unitario_est"] = df["precio"] / df["cantidad"]
    else:
        # precio unitario
        df["revenue"] = df["cantidad"] * df["precio"]

    return df, stats


def calculate_kpis(df: pd.DataFrame) -> pd.DataFrame:
    """KPIs básicos del negocio."""
    if df is None or df.empty:
        kpis = {
            "revenue_total": 0.0,
            "ordenes_totales": 0,
            "ticket_medio": 0.0,
            "unidades_vendidas": 0.0,
            "productos_unicos": 0,
            "fecha_inicio": pd.NaT,
            "fecha_fin": pd.NaT,
        }
        return pd.DataFrame([kpis])

    kpis = {
        "revenue_total": float(df["revenue"].sum()),
        "ordenes_totales": int(len(df)),
        "ticket_medio": float(df["revenue"].mean()) if len(df) else 0.0,
        "unidades_vendidas": float(df["cantidad"].sum()),
        "productos_unicos": int(df["producto"].nunique()),
        "fecha_inicio": df["fecha"].min(),
        "fecha_fin": df["fecha"].max(),
    }
    return pd.DataFrame([kpis])


def save_outputs(df_clean: pd.DataFrame, df_kpis: pd.DataFrame, meta: dict, paths: PipelinePaths) -> None:
    """Guarda outputs del pipeline."""
    paths.processed_dir.mkdir(parents=True, exist_ok=True)

    df_clean.to_parquet(paths.output_clean, index=False)
    df_kpis.to_parquet(paths.output_kpis, index=False)

    # metadata trazable
    paths.output_meta.parent.mkdir(parents=True, exist_ok=True)
    with open(paths.output_meta, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2, default=str)


def run_pipeline(paths: PipelinePaths = PipelinePaths(), sheet_name: Optional[str] = None) -> None:
    """Ejecuta pipeline completo."""
    # input
    input_path = paths.input_file or find_latest_excel(paths.input_dir)
    df_raw = load_input_file(input_path, sheet_name=sheet_name)

    # normalize
    df_norm, mapping = normalize_dataframe(df_raw)

    # clean + revenue
    df_clean, stats = clean_data(df_norm)

    if df_clean.empty:
        raise ValueError(
            "Tras limpiar, no quedaron filas válidas.\n"
            "Revisa fechas/números o envíame una muestra del Excel para ajustar reglas."
        )

    # kpis
    df_kpis = calculate_kpis(df_clean)

    # metadata
    meta = {
        "input_file": str(input_path),
        "rows_raw": int(len(df_raw)),
        "rows_norm": int(len(df_norm)),
        "rows_clean": int(len(df_clean)),
        "mapping_detected": mapping,
        "stats": stats,
        "revenue_mode": "line_total" if stats.get("precio_as_line_total", 0.0) == 1.0 else "unit_price",
        "date_min": str(df_clean["fecha"].min()),
        "date_max": str(df_clean["fecha"].max()),
        "columns_out": list(df_clean.columns),
    }

    # save
    save_outputs(df_clean, df_kpis, meta, paths)

    print("✅ Pipeline ejecutado correctamente.")
    print(f"📄 Input: {input_path}")
    print(f"🧭 Revenue mode: {meta['revenue_mode']}")
    print(df_kpis.to_string(index=False))


if __name__ == "__main__":
    run_pipeline()