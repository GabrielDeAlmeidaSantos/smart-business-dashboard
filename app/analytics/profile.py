from __future__ import annotations

from dataclasses import dataclass
import re
import pandas as pd


@dataclass(frozen=True)
class BusinessProfile:
    """
    Perfil inferido del negocio.
    """
    business_type: str           # 'servicios' | 'retail' | 'restauracion' | 'desconocido'
    subtype: str                 # opcional: 'peluqueria', 'moda', etc.
    confidence: float            # 0..1
    evidence: list[str]          # razones explicables


_SERVICIOS = [
    r"\bcorte\b", r"\btinte\b", r"\bpeinado\b", r"\bmechas\b", r"\bbarba\b",
    r"\btratamiento\b", r"\bmanicura\b", r"\bpedicura\b", r"\bcejas\b",
]
_RESTAURACION = [
    r"\bcafe\b", r"\bcerveza\b", r"\btapa\b", r"\bmenu\b", r"\bbocadillo\b",
    r"\brefresco\b", r"\bracion\b",
]
_RETAIL = [
    r"\btalla\b", r"\bcolor\b", r"\bmodelo\b", r"\bsku\b",
]


def infer_business_profile(df: pd.DataFrame, top_k_products: int = 30) -> BusinessProfile:
    """
    Inferencia heurística del tipo de negocio basada en nombres de producto/servicio.

    Ventaja:
        - No alucina.
        - Es explicable.
        - Arranca sin dataset etiquetado.

    Luego, cuando tengas muchos clientes, se reemplaza por un clasificador ML.
    """
    if df.empty:
        return BusinessProfile("desconocido", "desconocido", 0.0, ["dataset vacío"])

    top_names = (
        df.groupby("producto", as_index=False)["revenue"]
        .sum()
        .sort_values("revenue", ascending=False)
        .head(top_k_products)["producto"]
        .astype(str)
        .tolist()
    )
    text = " | ".join(top_names).lower()

    def score(patterns: list[str]) -> tuple[int, list[str]]:
        hits = []
        c = 0
        for p in patterns:
            if re.search(p, text):
                c += 1
                hits.append(p)
        return c, hits

    s_serv, hits_serv = score(_SERVICIOS)
    s_rest, hits_rest = score(_RESTAURACION)
    s_ret, hits_ret = score(_RETAIL)

    # Regla simple: el mayor gana
    scores = {"servicios": s_serv, "restauracion": s_rest, "retail": s_ret}
    best = max(scores, key=scores.get)
    best_score = scores[best]

    if best_score == 0:
        return BusinessProfile("desconocido", "desconocido", 0.35, ["sin matches claros en productos"])

    # Confianza proporcional y limitada
    conf = min(0.55 + 0.1 * best_score, 0.9)

    evidence = []
    if best == "servicios":
        evidence = [f"matches servicios: {len(hits_serv)}", "nombres tipo corte/tinte/tratamiento"]
        subtype = "peluqueria_estetica"
    elif best == "restauracion":
        evidence = [f"matches restauración: {len(hits_rest)}", "nombres tipo menú/café/tapa"]
        subtype = "bar_restaurante"
    else:
        evidence = [f"matches retail: {len(hits_ret)}", "nombres tipo talla/color/modelo"]
        subtype = "tienda_retail"

    return BusinessProfile(best, subtype, float(conf), evidence)