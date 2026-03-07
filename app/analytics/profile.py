# app/analytics/profile.py
from __future__ import annotations

from dataclasses import dataclass
import re
import unicodedata
import pandas as pd


@dataclass(frozen=True)
class BusinessProfile:
    """Perfil inferido del negocio (heurística explicable y conservadora)."""
    business_type: str   # "servicios" | "retail" | "restauracion" | "unknown"
    subtype: str         # "peluqueria_estetica" | "ferreteria" | "tienda_general" | "bar_restaurante" | "unknown"
    confidence: float    # 0..1
    evidence: list[str]  # razones explicables


_PLACEHOLDER_PRODUCTS = {"-", "SIN_PRODUCTO", "sin_producto", "nan", "none", ""}


def _strip_accents(s: str) -> str:
    """Quita diacríticos de forma estándar (Unicode NFKD)."""
    s = str(s)
    nfkd = unicodedata.normalize("NFKD", s)
    return "".join(ch for ch in nfkd if not unicodedata.combining(ch))


def _norm_text(s: str) -> str:
    s = _strip_accents(s).lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s


# ----------------------------
# Patrones
# ----------------------------
_SERVICIOS = [
    r"\bcorte\b",
    r"\btinte\b",
    r"\bpeinado\b",
    r"\bmechas\b",
    r"\bbarba\b",
    r"\btratamiento\b",
    r"\bmanicura\b",
    r"\bpedicura\b",
    r"\bcejas\b",
    r"\bunas\b",
    r"\bdepilacion\b",
    r"\bmasaje\b",
    r"\bfacial\b",
    r"\blifting\b",
    r"\bcoloracion\b",
]

_RESTAURACION = [
    r"\bcafe\b",
    r"\bcerveza\b",
    r"\btapa\b",
    r"\bmenu\b",
    r"\bbocadillo\b",
    r"\brefresco\b",
    r"\bracion\b",
    r"\bdesayuno\b",
    r"\bagua\b",
    r"\bvino\b",
    r"\bhamburguesa\b",
    r"\bpizza\b",
    r"\bcopa\b",
]

# Retail general
_RETAIL = [
    r"\btalla\b",
    r"\bcolor\b",
    r"\bmodelo\b",
    r"\bsku\b",
    r"\breferencia\b",
    r"\bref\b",
    r"\bean\b",
    r"\bcodigo\b",
    r"\bbarcode\b",
    r"\bpack\b",
]

# Ferretería / bricolaje / suministros
_FERRETERIA = [
    r"\btornillo\b",
    r"\btuerca\b",
    r"\barandela\b",
    r"\btaco\b",
    r"\bbroca\b",
    r"\bmartillo\b",
    r"\bdestornillador\b",
    r"\bllave inglesa\b",
    r"\ballen\b",
    r"\bsierra\b",
    r"\btaladro\b",
    r"\bcinta americana\b",
    r"\bcinta aislante\b",
    r"\bsilicona\b",
    r"\badhesivo\b",
    r"\bcola\b",
    r"\bpintura\b",
    r"\bbarniz\b",
    r"\brodillo\b",
    r"\bbrocha\b",
    r"\bcable\b",
    r"\benchufe\b",
    r"\bombilla\b",
    r"\bportalamp\b",
    r"\btuberia\b",
    r"\bgrifo\b",
    r"\bfontaneria\b",
    r"\bescuadra\b",
    r"\bbisagra\b",
    r"\bcerrojo\b",
    r"\bcandado\b",
    r"\bremache\b",
    r"\babrasivo\b",
    r"\blija\b",
]

# Tienda general / comercio no especializado
_TIENDA_GENERAL_HINTS = [
    r"\bcamiseta\b",
    r"\bpantalon\b",
    r"\bvestido\b",
    r"\bzapato\b",
    r"\bbolso\b",
    r"\baccesorio\b",
    r"\bcrema\b",
    r"\bserum\b",
    r"\bmaquillaje\b",
    r"\bchampu\b",
    r"\bgel\b",
    r"\bperfume\b",
    r"\bjuguete\b",
    r"\bregalo\b",
    r"\bpapeleria\b",
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


def _score_patterns(top_names: list[str], names_norm: list[str], patterns: list[str]) -> tuple[int, list[str]]:
    """
    Score por número de nombres top que matchean al menos un patrón.
    Reduce falsos positivos frente a contar solo patrones.
    """
    matched_names: list[str] = []
    for raw, n in zip(top_names, names_norm):
        if any(re.search(p, n) for p in patterns):
            matched_names.append(str(raw))
    return len(matched_names), matched_names[:4]


def infer_business_profile(df: pd.DataFrame, top_k_products: int = 30) -> BusinessProfile:
    """Inferencia heurística basada en nombres de producto/servicio."""
    if df is None or df.empty:
        return BusinessProfile("unknown", "unknown", 0.0, ["dataset vacío"])

    if "producto" not in df.columns:
        return BusinessProfile("unknown", "unknown", 0.0, ["falta columna 'producto'"])

    top_names = _top_product_names(df, top_k_products)
    if not top_names:
        return BusinessProfile("unknown", "unknown", 0.2, ["no hay productos válidos para inferir"])

    names_norm = [_norm_text(x) for x in top_names]

    s_serv, ex_serv = _score_patterns(top_names, names_norm, _SERVICIOS)
    s_rest, ex_rest = _score_patterns(top_names, names_norm, _RESTAURACION)
    s_ret, ex_ret = _score_patterns(top_names, names_norm, _RETAIL)
    s_ferr, ex_ferr = _score_patterns(top_names, names_norm, _FERRETERIA)
    s_tienda, ex_tienda = _score_patterns(top_names, names_norm, _TIENDA_GENERAL_HINTS)

    # Score principal por tipo de negocio
    scores = {
        "servicios": s_serv,
        "restauracion": s_rest,
        "retail": max(s_ret, s_ferr, s_tienda),
    }
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    best, best_score = ranked[0]
    second, second_score = ranked[1]

    if best_score == 0:
        return BusinessProfile(
            "unknown",
            "unknown",
            0.30,
            ["sin matches claros en top productos (perfilado débil)"],
        )

    margin = best_score - second_score
    ambiguous = margin <= 1

    # Confianza conservadora
    base = 0.45 + 0.06 * best_score
    if ambiguous:
        base -= 0.20
    conf = float(max(0.35, min(base, 0.85)))

    # Subtipo
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
        # Dentro de retail distinguimos mejor el subtipo
        retail_candidates = {
            "ferreteria": s_ferr,
            "tienda_general": max(s_tienda, s_ret),
        }
        retail_ranked = sorted(retail_candidates.items(), key=lambda x: x[1], reverse=True)
        retail_best, retail_best_score = retail_ranked[0]
        retail_second_score = retail_ranked[1][1]
        retail_margin = retail_best_score - retail_second_score

        if retail_best_score == 0:
            subtype = "tienda_general"
            ex_best = ex_ret or ex_tienda
        else:
            subtype = retail_best
            ex_best = ex_ferr if retail_best == "ferreteria" else (ex_tienda or ex_ret)

        evidence = [
            f"productos con señales de retail: {best_score} (margen vs 2º: {margin})",
            f"subtipo retail: {subtype} (margen interno: {retail_margin})",
            f"ejemplos: {', '.join(ex_best) if ex_best else '—'}",
        ]

    if ambiguous and conf < 0.55:
        return BusinessProfile(
            best,
            "unknown",
            conf,
            evidence + ["subtipo ambiguo: señales mezcladas"],
        )

    return BusinessProfile(best, subtype, conf, evidence)


def detect_business_profile(df: pd.DataFrame, top_k_products: int = 30) -> BusinessProfile:
    """Export estable usado por la app."""
    return infer_business_profile(df=df, top_k_products=top_k_products)