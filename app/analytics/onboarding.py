from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

from app.analytics.data_quality import (
    DataQualitySummary,
    assess_data_quality,
    should_block_dashboard,
)


# =============================================================================
# OBJETIVO DEL MÓDULO
# - Convertir la carga inicial en un proceso defendible comercialmente.
# - Dejar claro qué necesita un negocio para empezar.
# - Decidir si un negocio está:
#     * no preparado,
#     * preparado con limitaciones,
#     * listo para usar.
# - Dar soporte a una experiencia de onboarding más seria para Owner/Admin.
#
# IDEA CLAVE
# Hasta ahora el sistema procesa datos y muestra resultados, pero faltaba una capa
# que traduzca eso a lenguaje de implantación real:
# - qué archivo trajeron,
# - qué detectamos,
# - qué falta,
# - si ya podemos usar el sistema con credibilidad.
# =============================================================================


# =============================================================================
# Estados del onboarding
# =============================================================================
ONBOARDING_NOT_READY = "not_ready"
ONBOARDING_READY_WITH_LIMITATIONS = "ready_with_limitations"
ONBOARDING_READY = "ready"


# =============================================================================
# Modelos
# =============================================================================
@dataclass(frozen=True)
class OnboardingRequirement:
    """Requisito o condición del setup inicial."""

    code: str
    label: str
    status: str  # ok | warning | missing | blocked
    message: str
    value: Any | None = None


@dataclass(frozen=True)
class OnboardingSummary:
    """
    Resumen final del estado de onboarding.

    readiness_status:
        not_ready | ready_with_limitations | ready
    readiness_score:
        Score 0..100 del setup inicial.
    commercial_status:
        Texto corto defendible para conversación comercial.
    """

    readiness_status: str
    readiness_score: int
    commercial_status: str
    owner_message: str
    admin_message: str
    requirements: list[OnboardingRequirement]
    quality: dict[str, Any]
    stats: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["requirements"] = [asdict(x) for x in self.requirements]
        return out


# =============================================================================
# Helpers
# =============================================================================
def _req(code: str, label: str, status: str, message: str, value: Any | None = None) -> OnboardingRequirement:
    return OnboardingRequirement(
        code=code,
        label=label,
        status=status,
        message=message,
        value=value,
    )



def _safe_meta_stats(meta: dict | None) -> dict[str, Any]:
    meta = meta or {}
    stats = meta.get("stats") or {}
    return stats if isinstance(stats, dict) else {}



def _safe_mapping(meta: dict | None) -> dict[str, Any]:
    meta = meta or {}
    mapping = meta.get("mapping") or {}
    return mapping if isinstance(mapping, dict) else {}



def _has_usable_ticket_id(meta_stats: dict[str, Any], quality: DataQualitySummary) -> bool:
    # Preferimos la señal de quality si existe ticket_coverage.
    ticket_cov = quality.stats.get("ticket_coverage", 0.0)
    try:
        return float(ticket_cov) >= 0.70
    except Exception:
        return False



def _metric_exactness(meta_stats: dict[str, Any]) -> dict[str, str]:
    value = meta_stats.get("metric_exactness") or {}
    return value if isinstance(value, dict) else {}


# =============================================================================
# Motor principal
# =============================================================================
def assess_onboarding(
    df,
    meta: dict | None = None,
) -> OnboardingSummary:
    """
    Evalúa si el negocio ya está listo para usar el sistema.

    Enfoque:
    - usa data_quality como base;
    - añade lectura comercial/operativa del setup;
    - devuelve un estado accionable.
    """
    meta = meta or {}
    quality = assess_data_quality(df=df, meta=meta)
    meta_stats = _safe_meta_stats(meta)
    mapping = _safe_mapping(meta)
    exactness = _metric_exactness(meta_stats)

    requirements: list[OnboardingRequirement] = []

    # -------------------------------------------------------------------------
    # 1) Archivo procesable / calidad mínima
    # -------------------------------------------------------------------------
    if should_block_dashboard(quality):
        requirements.append(
            _req(
                "quality_base",
                "Calidad mínima del archivo",
                "blocked",
                "El archivo no supera la calidad mínima para usar el sistema con credibilidad.",
                value=quality.read_confidence_score,
            )
        )
    elif quality.action == "warn":
        requirements.append(
            _req(
                "quality_base",
                "Calidad mínima del archivo",
                "warning",
                "El archivo es usable, pero el setup inicial tiene limitaciones que conviene revisar.",
                value=quality.read_confidence_score,
            )
        )
    else:
        requirements.append(
            _req(
                "quality_base",
                "Calidad mínima del archivo",
                "ok",
                "La lectura del archivo es suficientemente sólida para empezar.",
                value=quality.read_confidence_score,
            )
        )

    # -------------------------------------------------------------------------
    # 2) Estructura básica disponible
    # -------------------------------------------------------------------------
    missing_required = quality.stats.get("missing_required_columns", []) or []
    if missing_required:
        requirements.append(
            _req(
                "core_structure",
                "Columnas base necesarias",
                "blocked",
                f"Faltan columnas base para operar bien: {', '.join(missing_required)}.",
                value=missing_required,
            )
        )
    else:
        requirements.append(
            _req(
                "core_structure",
                "Columnas base necesarias",
                "ok",
                "Existen las columnas mínimas para calcular ingresos, producto, cantidad y fechas.",
            )
        )

    # -------------------------------------------------------------------------
    # 3) Nivel de exactitud operativa
    # -------------------------------------------------------------------------
    ticket_usable = _has_usable_ticket_id(meta_stats, quality)
    if ticket_usable:
        requirements.append(
            _req(
                "operational_exactness",
                "Exactitud de tickets y operaciones",
                "ok",
                "El archivo permite calcular tickets y operaciones con buena exactitud.",
                value=quality.stats.get("ticket_coverage"),
            )
        )
    else:
        requirements.append(
            _req(
                "operational_exactness",
                "Exactitud de tickets y operaciones",
                "warning",
                "El sistema puede arrancar, pero algunas métricas de ticket y operación serán aproximadas.",
                value=quality.stats.get("ticket_coverage"),
            )
        )

    # -------------------------------------------------------------------------
    # 4) Calidad del detalle de producto/servicio
    # -------------------------------------------------------------------------
    placeholder_ratio = float(quality.stats.get("producto_placeholder_ratio", 0.0) or 0.0)
    if placeholder_ratio >= 0.70:
        requirements.append(
            _req(
                "product_detail",
                "Detalle de producto o servicio",
                "blocked",
                "La mayoría de las filas no identifican bien el producto o servicio. El motor perdería mucha precisión.",
                value=placeholder_ratio,
            )
        )
    elif placeholder_ratio >= 0.25:
        requirements.append(
            _req(
                "product_detail",
                "Detalle de producto o servicio",
                "warning",
                "El sistema puede arrancar, pero el detalle de mix y recomendaciones será menos preciso de lo deseable.",
                value=placeholder_ratio,
            )
        )
    else:
        requirements.append(
            _req(
                "product_detail",
                "Detalle de producto o servicio",
                "ok",
                "El detalle de producto o servicio es suficientemente útil para análisis y recomendaciones.",
                value=placeholder_ratio,
            )
        )

    # -------------------------------------------------------------------------
    # 5) Trazabilidad de revenue
    # -------------------------------------------------------------------------
    revenue_source = meta_stats.get("revenue_source")
    revenue_conf = meta_stats.get("revenue_source_confidence")
    if not revenue_source:
        requirements.append(
            _req(
                "revenue_traceability",
                "Trazabilidad del ingreso",
                "warning",
                "No queda del todo claro cómo se reconstruyó el ingreso. Conviene revisar la normalización.",
            )
        )
    else:
        status = "ok" if str(revenue_conf).lower() in {"alta", "media-alta", "high", "medium-high"} else "warning"
        msg = (
            f"El ingreso se ha reconstruido desde '{revenue_source}' con confianza {revenue_conf}."
            if revenue_conf
            else f"El ingreso se ha reconstruido desde '{revenue_source}'."
        )
        requirements.append(
            _req(
                "revenue_traceability",
                "Trazabilidad del ingreso",
                status,
                msg,
                value=revenue_source,
            )
        )

    # -------------------------------------------------------------------------
    # 6) Colisiones de mapping
    # -------------------------------------------------------------------------
    collisions = mapping.get("mapping_collisions") or {}
    collisions_count = len(collisions) if isinstance(collisions, dict) else 0
    if collisions_count > 0:
        requirements.append(
            _req(
                "mapping_stability",
                "Estabilidad del mapeo",
                "warning",
                "Hubo colisiones al detectar columnas. El sistema eligió la mejor opción, pero conviene revisar ese archivo.",
                value=collisions_count,
            )
        )
    else:
        requirements.append(
            _req(
                "mapping_stability",
                "Estabilidad del mapeo",
                "ok",
                "No se detectaron colisiones relevantes en el mapeo de columnas.",
                value=0,
            )
        )

    # -------------------------------------------------------------------------
    # Scoring final
    # -------------------------------------------------------------------------
    blocked_count = sum(1 for r in requirements if r.status == "blocked")
    warning_count = sum(1 for r in requirements if r.status == "warning")
    ok_count = sum(1 for r in requirements if r.status == "ok")

    score = 100
    score -= blocked_count * 35
    score -= warning_count * 10
    score = max(0, min(100, score))

    if blocked_count > 0:
        readiness_status = ONBOARDING_NOT_READY
        commercial_status = "No listo para usar"
        owner_message = (
            "Todavía no conviene arrancar con este archivo. Antes hay que corregir la base para que el sistema no dé lecturas engañosas."
        )
        admin_message = (
            f"Onboarding bloqueado: {blocked_count} requisito(s) bloqueantes, {warning_count} con warning, {ok_count} correctos."
        )
    elif warning_count > 0:
        readiness_status = ONBOARDING_READY_WITH_LIMITATIONS
        commercial_status = "Listo con limitaciones"
        owner_message = (
            "El negocio ya puede empezar a usar el sistema, pero algunas métricas o recomendaciones tendrán limitaciones que conviene conocer."
        )
        admin_message = (
            f"Onboarding usable con límites: {warning_count} requisito(s) con warning, {ok_count} correctos."
        )
    else:
        readiness_status = ONBOARDING_READY
        commercial_status = "Listo para usar"
        owner_message = (
            "El negocio ya tiene una base suficientemente sólida para usar el sistema con una lectura defendible."
        )
        admin_message = (
            f"Onboarding correcto: {ok_count} requisito(s) cubiertos, sin bloqueos relevantes."
        )

    stats = {
        "requirements_total": len(requirements),
        "requirements_ok": ok_count,
        "requirements_warning": warning_count,
        "requirements_blocked": blocked_count,
        "ticket_usable": ticket_usable,
        "metric_exactness": exactness,
    }

    return OnboardingSummary(
        readiness_status=readiness_status,
        readiness_score=int(score),
        commercial_status=commercial_status,
        owner_message=owner_message,
        admin_message=admin_message,
        requirements=requirements,
        quality=quality.to_dict(),
        stats=stats,
    )


# =============================================================================
# Helpers de UI / uso
# =============================================================================
def is_business_ready(summary: OnboardingSummary) -> bool:
    return summary.readiness_status in {ONBOARDING_READY, ONBOARDING_READY_WITH_LIMITATIONS}



def is_business_fully_ready(summary: OnboardingSummary) -> bool:
    return summary.readiness_status == ONBOARDING_READY



def onboarding_owner_caption(summary: OnboardingSummary) -> str:
    return (
        f"Estado inicial: {summary.commercial_status}. "
        f"{summary.owner_message} "
        f"Preparación estimada: {summary.readiness_score}/100."
    )



def onboarding_admin_lines(summary: OnboardingSummary) -> list[str]:
    lines = [summary.admin_message]
    for req in summary.requirements:
        lines.append(f"- {req.status.upper()}: {req.label} -> {req.message}")
    return lines