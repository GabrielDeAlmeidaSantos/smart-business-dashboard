from __future__ import annotations

from dataclasses import dataclass
import re
import unicodedata

import pandas as pd


# =============================================================================
# OBJETIVO DE ESTA VERSIÓN
# - Mejorar la detección del tipo de negocio y subtipo.
# - Reducir ambigüedad entre servicios, retail y restauración.
# - Activar el catálogo vertical correcto con más fiabilidad.
#
# PROBLEMAS DE LA VERSIÓN ORIGINAL
# 1) Dependía demasiado del top de productos sin ponderar bien calidad de señal.
# 2) Retail podía absorber señales ambiguas con demasiado facilidad.
# 3) No distinguía bien entre:
#    - un servicio real,
#    - un producto cosmético vendido en retail,
#    - restauración con vocabulario genérico.
# 4) La confianza era demasiado lineal y poco sensible a mezcla/conflicto.
#
# EN ESTA VERSIÓN
# - usamos evidencia ponderada por revenue/frecuencia;
# - diferenciamos señales fuertes y débiles;
# - medimos conflicto entre verticales;
# - hacemos subtipo más conservador cuando la señal no es suficientemente limpia.
# =============================================================================


@dataclass(frozen=True)
class BusinessProfile:
    """Perfil inferido del negocio (heurística explicable y conservadora)."""

    business_type: str   # "servicios" | "retail" | "restauracion" | "unknown"
    subtype: str         # "peluqueria_estetica" | "ferreteria" | "tienda_general" | "bar_restaurante" | "unknown"
    confidence: float    # 0..1
    evidence: list[str]  # razones explicables


_PLACEHOLDER_PRODUCTS = {"-", "SIN_PRODUCTO", "sin_producto", "nan", "none", ""}


# =============================================================================
# Normalización
# =============================================================================
def _strip_accents(s: str) -> str:
    """Quita diacríticos de forma estándar (Unicode NFKD)."""
    s = str(s)
    nfkd = unicodedata.normalize("NFKD", s)
    return "".join(ch for ch in nfkd if not unicodedata.combining(ch))



def _norm_text(s: str) -> str:
    s = _strip_accents(s).lower().strip()
    s = re.sub(r"[^a-z0-9\s\-_/+]", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s


# =============================================================================
# Patrones
# =============================================================================
# Nota:
# - separo señales fuertes y genéricas.
# - algunas palabras antes daban falsos positivos entre verticales.
# - por ejemplo "tratamiento", "gel", "crema" o "pack" pueden existir en más de un contexto.

_SERVICIOS_STRONG = [
    r"\bcorte\b",
    r"\btinte\b",
    r"\bpeinado\b",
    r"\bmechas\b",
    r"\bbarba\b",
    r"\bmanicura\b",
    r"\bpedicura\b",
    r"\bcejas\b",
    r"\bunas\b",
    r"\bdepilacion\b",
    r"\bmasaje\b",
    r"\bfacial\b",
    r"\blifting\b",
    r"\bcoloracion\b",
    r"\blavado\b",
    r"\bsecado\b",
    r"\balisado\b",
    r"\bkeratina\b",
    r"\bretoque\b",
]

_SERVICIOS_SOFT = [
    r"\btratamiento\b",
    r"\bservicio\b",
    r"\bsesion\b",
    r"\bbono\b",
    r"\bdiagnostico\b",
]

_RESTAURACION_STRONG = [
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
    r"\bcaña\b",
    r"\bpincho\b",
    r"\bcombinado\b",
    r"\bpostre\b",
]

_RESTAURACION_SOFT = [
    r"\bextra\b",
    r"\bpara llevar\b",
    r"\bbebida\b",
    r"\bacompanamiento\b",
]

_RETAIL_GENERIC_STRONG = [
    r"\bsku\b",
    r"\breferencia\b",
    r"\bref\b",
    r"\bean\b",
    r"\bcodigo\b",
    r"\bbarcode\b",
    r"\btalla\b",
    r"\bmodelo\b",
]

_RETAIL_GENERIC_SOFT = [
    r"\bcolor\b",
    r"\bpack\b",
    r"\bunidades?\b",
]

_FERRETERIA_STRONG = [
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

_TIENDA_GENERAL_STRONG = [
    r"\bcamiseta\b",
    r"\bpantalon\b",
    r"\bvestido\b",
    r"\bzapato\b",
    r"\bbolso\b",
    r"\baccesorio\b",
    r"\bmaquillaje\b",
    r"\bperfume\b",
    r"\bjuguete\b",
    r"\bregalo\b",
    r"\bpapeleria\b",
]

_TIENDA_GENERAL_SOFT = [
    r"\bcrema\b",
    r"\bserum\b",
    r"\bchampu\b",
    r"\bgel\b",
]


# =============================================================================
# Helpers internos
# =============================================================================
def _valid_product_mask(s: pd.Series) -> pd.Series:
    s = s.astype(str).fillna("").str.strip()
    s_norm = s.map(_norm_text)
    placeholders = {_norm_text(x) for x in _PLACEHOLDER_PRODUCTS}
    return ~s_norm.isin(placeholders)



def _prepare_product_rows(df: pd.DataFrame, top_k: int) -> pd.DataFrame:
    """
    Prepara una tabla agregada por producto con peso explicable.

    Peso usado:
    - si hay revenue: peso por revenue agregado
    - si no: peso por frecuencia

    Además guardamos share para tener señal relativa, no solo conteo bruto.
    """
    if df is None or df.empty or "producto" not in df.columns:
        return pd.DataFrame(columns=["producto", "producto_norm", "weight", "share"])

    mask = _valid_product_mask(df["producto"])
    df2 = df.loc[mask].copy()
    if df2.empty:
        return pd.DataFrame(columns=["producto", "producto_norm", "weight", "share"])

    df2["producto"] = df2["producto"].astype(str).str.strip()
    df2["producto_norm"] = df2["producto"].map(_norm_text)

    if "revenue" in df2.columns:
        grp = (
            df2.groupby(["producto", "producto_norm"], as_index=False)["revenue"]
            .sum()
            .rename(columns={"revenue": "weight"})
            .sort_values("weight", ascending=False)
            .head(int(top_k))
        )
    else:
        grp = (
            df2.groupby(["producto", "producto_norm"], as_index=False)
            .size()
            .rename(columns={"size": "weight"})
            .sort_values("weight", ascending=False)
            .head(int(top_k))
        )

    total = float(grp["weight"].sum()) if not grp.empty else 0.0
    grp["share"] = (grp["weight"] / total) if total > 0 else 0.0
    return grp



def _score_weighted_patterns(
    top_rows: pd.DataFrame,
    strong_patterns: list[str],
    soft_patterns: list[str] | None = None,
) -> dict:
    """
    Calcula señal ponderada por producto.

    Lógica:
    - strong match: suma 1.00 * share del producto
    - soft match: suma 0.50 * share del producto
    - devolvemos también ejemplos para evidencia explicable

    Esto reduce falsos positivos y evita que un único patrón genérico mande demasiado.
    """
    soft_patterns = soft_patterns or []
    score = 0.0
    strong_hits: list[str] = []
    soft_hits: list[str] = []
    raw_hits = 0

    if top_rows is None or top_rows.empty:
        return {
            "score": 0.0,
            "raw_hits": 0,
            "strong_hits": [],
            "soft_hits": [],
        }

    for _, row in top_rows.iterrows():
        raw = str(row["producto"])
        norm = str(row["producto_norm"])
        share = float(row["share"])

        strong = any(re.search(p, norm) for p in strong_patterns)
        soft = any(re.search(p, norm) for p in soft_patterns) if not strong else False

        if strong:
            score += 1.00 * share
            raw_hits += 1
            if raw not in strong_hits:
                strong_hits.append(raw)
        elif soft:
            score += 0.50 * share
            raw_hits += 1
            if raw not in soft_hits:
                soft_hits.append(raw)

    return {
        "score": float(score),
        "raw_hits": int(raw_hits),
        "strong_hits": strong_hits[:5],
        "soft_hits": soft_hits[:5],
    }



def _describe_hits(label: str, payload: dict) -> str:
    strong = payload.get("strong_hits") or []
    soft = payload.get("soft_hits") or []
    parts: list[str] = []
    if strong:
        parts.append(f"{label} fuertes: {', '.join(strong[:3])}")
    if soft:
        parts.append(f"{label} suaves: {', '.join(soft[:2])}")
    return " | ".join(parts) if parts else f"sin ejemplos claros de {label}"



def _clamp(x: float, low: float, high: float) -> float:
    return max(low, min(high, float(x)))


# =============================================================================
# Inferencia principal
# =============================================================================
def infer_business_profile(df: pd.DataFrame, top_k_products: int = 30) -> BusinessProfile:
    """
    Inferencia heurística más robusta basada en nombres de producto/servicio.

    Filosofía:
    - explicable;
    - conservadora;
    - prioriza acertar vertical principal antes que inventar subtipo.
    """
    if df is None or df.empty:
        return BusinessProfile("unknown", "unknown", 0.0, ["dataset vacío"])

    if "producto" not in df.columns:
        return BusinessProfile("unknown", "unknown", 0.0, ["falta columna 'producto'"])

    top_rows = _prepare_product_rows(df, top_k_products)
    if top_rows.empty:
        return BusinessProfile("unknown", "unknown", 0.20, ["no hay productos válidos para inferir"])

    # Señales base por vertical / subtipo
    sig_services = _score_weighted_patterns(top_rows, _SERVICIOS_STRONG, _SERVICIOS_SOFT)
    sig_rest = _score_weighted_patterns(top_rows, _RESTAURACION_STRONG, _RESTAURACION_SOFT)
    sig_retail_generic = _score_weighted_patterns(top_rows, _RETAIL_GENERIC_STRONG, _RETAIL_GENERIC_SOFT)
    sig_ferr = _score_weighted_patterns(top_rows, _FERRETERIA_STRONG, [])
    sig_store = _score_weighted_patterns(top_rows, _TIENDA_GENERAL_STRONG, _TIENDA_GENERAL_SOFT)

    # Retail se compone de:
    # - señal ferretería
    # - señal tienda general
    # - algo de genérico retail, pero con peso limitado para no comerse todo
    retail_score = max(
        sig_ferr["score"],
        sig_store["score"],
        sig_retail_generic["score"] * 0.75,
    )

    type_scores = {
        "servicios": sig_services["score"],
        "restauracion": sig_rest["score"],
        "retail": retail_score,
    }

    ranked = sorted(type_scores.items(), key=lambda x: x[1], reverse=True)
    best_type, best_score = ranked[0]
    second_type, second_score = ranked[1]
    third_type, third_score = ranked[2]

    if best_score <= 0:
        return BusinessProfile(
            "unknown",
            "unknown",
            0.30,
            ["sin matches claros en top productos (perfilado débil)"],
        )

    margin_1_2 = best_score - second_score
    spread = best_score - third_score

    # Ambigüedad real: la segunda señal está demasiado cerca.
    ambiguous_type = margin_1_2 < 0.12

    evidence: list[str] = [
        f"score servicios={sig_services['score']:.2f} | restauracion={sig_rest['score']:.2f} | retail={retail_score:.2f}",
        f"margen principal={margin_1_2:.2f} frente a {second_type}",
    ]

    # Casos conflictivos habituales a vigilar.
    if sig_services["score"] > 0 and sig_store["score"] > 0:
        evidence.append("hay mezcla entre señales de servicio y tienda general")
    if sig_rest["score"] > 0 and sig_retail_generic["score"] > 0:
        evidence.append("hay mezcla entre señales de restauración y retail genérico")

    # ---------------------------------------------------------------------
    # Decisión del tipo principal
    # ---------------------------------------------------------------------
    # Regla conservadora:
    # si el mejor score es muy débil o muy mezclado, preferimos unknown antes que activar mal el catálogo.
    if best_score < 0.16 and ambiguous_type:
        return BusinessProfile(
            "unknown",
            "unknown",
            0.38,
            evidence + ["señal demasiado débil y mezclada para asignar vertical con seguridad"],
        )

    # Confianza del tipo
    type_conf = 0.45 + (best_score * 0.70) + (margin_1_2 * 0.60)
    if ambiguous_type:
        type_conf -= 0.14
    type_conf = _clamp(type_conf, 0.35, 0.92)

    # ---------------------------------------------------------------------
    # Subtipo por vertical
    # ---------------------------------------------------------------------
    subtype = "unknown"
    subtype_conf = 0.0

    if best_type == "servicios":
        # En esta app solo tenemos peluquería/estética como subtipo de servicios.
        # Pero no lo disparamos con alegría si la señal es floja.
        subtype = "peluqueria_estetica"
        subtype_conf = _clamp(0.50 + sig_services["score"] * 0.80, 0.45, 0.92)
        evidence.append(_describe_hits("servicios", sig_services))

        # Si hay demasiada mezcla con tienda/cosmética retail, dejamos subtipo en unknown.
        if sig_store["score"] >= max(0.12, sig_services["score"] * 0.80):
            subtype = "unknown"
            subtype_conf = min(subtype_conf, 0.50)
            evidence.append("subtipo de servicios degradado por mezcla fuerte con tienda general")

    elif best_type == "restauracion":
        subtype = "bar_restaurante"
        subtype_conf = _clamp(0.52 + sig_rest["score"] * 0.80, 0.45, 0.92)
        evidence.append(_describe_hits("restauracion", sig_rest))

        # Si la señal de restauración es real pero muy mezclada con retail genérico, degradamos subtipo.
        if sig_retail_generic["score"] >= max(0.10, sig_rest["score"] * 0.85):
            subtype = "unknown"
            subtype_conf = min(subtype_conf, 0.52)
            evidence.append("subtipo de restauración degradado por mezcla fuerte con retail genérico")

    elif best_type == "retail":
        retail_candidates = {
            "ferreteria": sig_ferr["score"],
            "tienda_general": max(sig_store["score"], sig_retail_generic["score"] * 0.80),
        }
        retail_ranked = sorted(retail_candidates.items(), key=lambda x: x[1], reverse=True)
        retail_best, retail_best_score = retail_ranked[0]
        retail_second, retail_second_score = retail_ranked[1]
        retail_margin = retail_best_score - retail_second_score

        evidence.append(
            f"subscore retail -> ferreteria={retail_candidates['ferreteria']:.2f} | tienda_general={retail_candidates['tienda_general']:.2f}"
        )

        if retail_best_score <= 0:
            subtype = "unknown"
            subtype_conf = 0.40
            evidence.append("retail detectado, pero subtipo sin señal clara")
        else:
            subtype = retail_best
            subtype_conf = _clamp(0.46 + retail_best_score * 0.75 + retail_margin * 0.40, 0.40, 0.90)

            if subtype == "ferreteria":
                evidence.append(_describe_hits("ferreteria", sig_ferr))
            else:
                evidence.append(_describe_hits("tienda_general", sig_store))
                if sig_retail_generic["score"] > 0:
                    evidence.append(_describe_hits("retail_generico", sig_retail_generic))

            # Si el margen interno es pobre, no fingimos subtipo fuerte.
            if retail_margin < 0.08:
                subtype = "unknown"
                subtype_conf = min(subtype_conf, 0.52)
                evidence.append("subtipo retail degradado por empate interno")

    # ---------------------------------------------------------------------
    # Decisión final de subtipo y confianza global
    # ---------------------------------------------------------------------
    # Importante: si el tipo es razonable pero subtipo no está suficientemente claro,
    # devolvemos subtipo unknown antes que activar mal un vertical específico.
    if subtype != "unknown" and subtype_conf < 0.50:
        subtype = "unknown"
        evidence.append("subtipo degradado por confianza insuficiente")

    final_conf = type_conf
    if subtype == "unknown":
        final_conf = min(final_conf, 0.68)
    else:
        final_conf = _clamp((type_conf * 0.65) + (subtype_conf * 0.35), 0.40, 0.93)

    # Freno final: conflicto fuerte entre dos verticales principales.
    if best_score > 0 and second_score / best_score >= 0.90:
        final_conf = min(final_conf, 0.56)
        evidence.append("conflicto alto entre vertical principal y secundaria")
        if subtype != "unknown":
            subtype = "unknown"
            evidence.append("subtipo eliminado por conflicto alto de vertical")

    return BusinessProfile(best_type, subtype, round(float(final_conf), 4), evidence)


# =============================================================================
# Export estable
# =============================================================================
def detect_business_profile(df: pd.DataFrame, top_k_products: int = 30) -> BusinessProfile:
    """Export estable usado por la app."""
    return infer_business_profile(df=df, top_k_products=top_k_products)