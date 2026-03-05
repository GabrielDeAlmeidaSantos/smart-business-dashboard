# app/analytics/profile.py
from __future__ import annotations

from dataclasses import dataclass
import re
import unicodedata
import pandas as pd


@dataclass(frozen=True)
class BusinessProfile:
    """Perfil inferido del negocio (heurística explicable, conservadora)."""
    business_type: str           # 'servicios' | 'retail' | 'restauracion' | 'desconocido'
    subtype: str                 # 'peluqueria_estetica', 'bar_restaurante', etc.
    confidence: float            # 0..1
    evidence: list[str]          # razones explicables


_PLACEHOLDER_PRODUCTS = {"-", "SIN_PRODUCTO", "sin_producto", "nan", "none", ""}


def _strip_accents(s: str) -> str:
    """Quita diacríticos de forma estándar (Unicode NFKD)."""
    s = str(s)
    nfkd = unicodedata.normalize("NFKD", s)  # :contentReference[oaicite:1]{index=1}
    return "".join(ch for ch in nfkd if not unicodedata.combining(ch))


def _norm_text(s: str) -> str:
    s = _strip_accents(s).lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s


# Tokens/regex (reglas de arranque, pero más cubrientes)
# Nota: patrones SIN acentos (porque normalizamos texto).
_SERVICIOS = [
    r"\bcorte\b", r"\btinte\b", r"\bpeinado\b", r"\bmechas\b", r"\bbarba\b",
    r"\btratamiento\b", r"\bmanicura\b", r"\bpedicura\b", r"\bcejas\b",
    r"\bunas\b", r"\bdepilacion\b", r"\bmasaje\b", r"\bfacial\b",
    r"\blifting\b", r"\bcoloracion\b",
]
_RESTAURACION = [
    r"\bcafe\b", r"\bcerveza\b", r"\btapa\b", r"\bmenu\b", r"\bbocadillo\b",
    r"\brefresco\b", r"\bracion\b", r"\bdesayuno\b", r"\bagua\b", r"\bvino\b",
    r"\bhamburguesa\b", r"\bpizza\b", r"\bcopa\b",
]
_RETAIL = [
    r"\btalla\b", r"\bcolor\b", r"\bmodelo\b", r"\bsku\b", r"\breferencia\b", r"\bref\b",
    r"\be an\b", r"\bean\b", r"\bcodigo\b", r"\bbarcode\b", r"\bpack\b",
]


def _top_product_names(df: pd.DataFrame, top_k: int) -> list[str]:
    """Top productos robusto (por revenue si existe; si no por frecuencia). Filtra placeholders."""
    if df is None or df.empty or "producto" not in df.columns:
        return []

    s = df["producto"].astype(str).fillna("").str.strip()
    s_norm = s.map(_norm_text)
    mask = ~s_norm.isin({x.lower() for x in _PLACEHOLDER_PRODUCTS})
    df2 = df.loc[mask].copy()

    if df2.empty:
        return []

    if "revenue" in df2.columns:
        top = (
            df2.assign(_p=df2["producto"].astype(str))
            .groupby("_p", as_index=False)["revenue"]
            .sum()
            .sort_values("revenue", ascending=False)
            .head(int(top_k))["_p"]
            .tolist()
        )
        return top

    return df2["producto"].astype(str).value_counts().head(int(top_k)).index.tolist()


def infer_business_profile(df: pd.DataFrame, top_k_products: int = 30) -> BusinessProfile:
    """Inferencia heurística basada en nombres de producto/servicio (conservadora)."""
    if df is None or df.empty:
        return BusinessProfile("desconocido", "desconocido", 0.0, ["dataset vacío"])

    if "producto" not in df.columns:
        return BusinessProfile("desconocido", "desconocido", 0.0, ["falta columna 'producto'"])

    top_names = _top_product_names(df, top_k_products)
    if not top_names:
        return BusinessProfile("desconocido", "desconocido", 0.2, ["no hay productos válidos para inferir"])

    # Normaliza texto global y también lista normalizada por item
    names_norm = [_norm_text(x) for x in top_names]
    text = " | ".join(names_norm)

    def score(patterns: list[str]) -> tuple[int, list[str]]:
        """
        Score por nº de NOMBRES top que matchean al menos un patrón.
        Esto reduce falsos positivos vs contar solo patrones.
        """
        matched_names = []
        for raw, n in zip(top_names, names_norm):
            if any(re.search(p, n) for p in patterns):
                matched_names.append(str(raw))
        # hits = nº productos que matchean
        return len(matched_names), matched_names[:4]

    s_serv, ex_serv = score(_SERVICIOS)
    s_rest, ex_rest = score(_RESTAURACION)
    s_ret, ex_ret = score(_RETAIL)

    scores = {"servicios": s_serv, "restauracion": s_rest, "retail": s_ret}
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    best, best_score = ranked[0]
    second, second_score = ranked[1]

    if best_score == 0:
        return BusinessProfile(
            "desconocido",
            "desconocido",
            0.30,
            ["sin matches claros en top productos (perfilado débil)"],
        )

    margin = best_score - second_score
    ambiguous = margin <= 1

    # Confianza conservadora: depende de nº de productos con match y margen
    # (cap 0.80)
    base = 0.45 + 0.06 * best_score
    if ambiguous:
        base -= 0.20
    conf = float(max(0.35, min(base, 0.80)))

    if best == "servicios":
        subtype = "peluqueria_estetica"
        evidence = [
            f"productos con señales de servicios: {best_score} (margen vs 2º: {margin})",
            f"ejemplos: {', '.join(ex_serv) if ex_serv else '—'}",
        ]
    elif best == "restauracion":
        subtype = "bar_restaurante"
        evidence = [
            f"productos con señales de restauración: {best_score} (margen vs 2º: {margin})",
            f"ejemplos: {', '.join(ex_rest) if ex_rest else '—'}",
        ]
    else:
        subtype = "tienda_retail"
        evidence = [
            f"productos con señales de retail: {best_score} (margen vs 2º: {margin})",
            f"ejemplos: {', '.join(ex_ret) if ex_ret else '—'}",
        ]

    if ambiguous and conf < 0.55:
        return BusinessProfile(
            "desconocido",
            "mixto_o_ambiguo",
            conf,
            evidence + ["clasificación ambigua: señales mezcladas"],
        )

    return BusinessProfile(best, subtype, conf, evidence)


def detect_business_profile(df: pd.DataFrame, top_k_products: int = 30) -> BusinessProfile:
    """Export estable usado por la app."""
    return infer_business_profile(df=df, top_k_products=top_k_products)