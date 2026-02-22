# analytics/profile.py
from __future__ import annotations

from dataclasses import dataclass
import re
import pandas as pd


@dataclass(frozen=True)
class BusinessProfile:
    """Perfil inferido del negocio (heurística explicable, conservadora)."""
    business_type: str           # 'servicios' | 'retail' | 'restauracion' | 'desconocido'
    subtype: str                 # 'peluqueria_estetica', 'bar_restaurante', etc.
    confidence: float            # 0..1
    evidence: list[str]          # razones explicables


# Tokens/regex (mantén esto simple: son reglas de arranque)
_SERVICIOS = [
    r"\bcorte\b", r"\btinte\b", r"\bpeinado\b", r"\bmechas\b", r"\bbarba\b",
    r"\btratamiento\b", r"\bmanicura\b", r"\bpedicura\b", r"\bcejas\b",
]
_RESTAURACION = [
    r"\bcafe\b", r"\bcafé\b", r"\bcerveza\b", r"\btapa\b", r"\bmenu\b", r"\bmenú\b",
    r"\bbocadillo\b", r"\brefresco\b", r"\bracion\b", r"\bración\b",
]
_RETAIL = [
    r"\btalla\b", r"\bcolor\b", r"\bmodelo\b", r"\bsku\b",
]


def _top_product_names(df: pd.DataFrame, top_k: int) -> list[str]:
    """Obtiene top productos de forma robusta (por revenue si existe, si no por frecuencia)."""
    if df is None or df.empty or "producto" not in df.columns:
        return []

    s = df["producto"].astype(str)

    if "revenue" in df.columns:
        top = (
            df.assign(_p=s)
            .groupby("_p", as_index=False)["revenue"]
            .sum()
            .sort_values("revenue", ascending=False)
            .head(int(top_k))["_p"]
            .tolist()
        )
        return top

    # Fallback: frecuencia
    return s.value_counts().head(int(top_k)).index.tolist()


def infer_business_profile(df: pd.DataFrame, top_k_products: int = 30) -> BusinessProfile:
    """Inferencia heurística basada en nombres de producto/servicio (conservadora)."""
    if df is None or df.empty:
        return BusinessProfile("desconocido", "desconocido", 0.0, ["dataset vacío"])

    if "producto" not in df.columns:
        return BusinessProfile("desconocido", "desconocido", 0.0, ["falta columna 'producto'"])

    top_names = _top_product_names(df, top_k_products)
    if not top_names:
        return BusinessProfile("desconocido", "desconocido", 0.2, ["no hay productos válidos para inferir"])

    text = " | ".join(top_names).lower()

    def score(patterns: list[str]) -> tuple[int, list[str]]:
        hits: list[str] = []
        for p in patterns:
            if re.search(p, text):
                hits.append(p)
        return len(hits), hits

    s_serv, hits_serv = score(_SERVICIOS)
    s_rest, hits_rest = score(_RESTAURACION)
    s_ret, hits_ret = score(_RETAIL)

    scores = {"servicios": s_serv, "restauracion": s_rest, "retail": s_ret}
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    best, best_score = ranked[0]
    second, second_score = ranked[1]

    # Si no hay matches claros: desconocido
    if best_score == 0:
        return BusinessProfile(
            "desconocido",
            "desconocido",
            0.30,
            ["sin matches claros en productos/servicios (perfilado débil)"],
        )

    # Ambigüedad: si el segundo está muy cerca, bajamos confianza fuerte
    margin = best_score - second_score
    ambiguous = margin <= 1  # regla simple y conservadora

    # Confianza conservadora: depende de #hits y del margen
    # (Máximo 0.80 porque esto es solo texto, no ML)
    base = 0.45 + 0.08 * best_score
    if ambiguous:
        base -= 0.20
    conf = float(max(0.35, min(base, 0.80)))

    # Evidencia útil: enseña ejemplos reales (top_names) en vez de regex
    # Extraemos algunos tokens “humanos” detectados
    def examples_for(patterns: list[str]) -> list[str]:
        ex = []
        for name in top_names:
            n = str(name).lower()
            if any(re.search(p, n) for p in patterns):
                ex.append(str(name))
            if len(ex) >= 4:
                break
        return ex

    if best == "servicios":
        subtype = "peluqueria_estetica"
        ex = examples_for(_SERVICIOS)
        evidence = [
            f"matches servicios: {best_score} (margen vs 2º: {margin})",
            f"ejemplos: {', '.join(ex) if ex else '—'}",
        ]
    elif best == "restauracion":
        subtype = "bar_restaurante"
        ex = examples_for(_RESTAURACION)
        evidence = [
            f"matches restauración: {best_score} (margen vs 2º: {margin})",
            f"ejemplos: {', '.join(ex) if ex else '—'}",
        ]
    else:
        subtype = "tienda_retail"
        ex = examples_for(_RETAIL)
        evidence = [
            f"matches retail: {best_score} (margen vs 2º: {margin})",
            f"ejemplos: {', '.join(ex) if ex else '—'}",
        ]

    # Si es ambiguo, no mientas: etiqueta como desconocido/mixto si quieres
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