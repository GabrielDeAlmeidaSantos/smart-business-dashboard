from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import pandas as pd


# =============================================================================
# OBJETIVO DEL MÓDULO
# - Medir si el dataset cargado es suficientemente fiable para enseñar métricas.
# - Detectar problemas típicos de lectura / normalización / estructura.
# - Exponer señales claras para Owner y Admin.
# - Permitir bloquear o advertir antes de mostrar números absurdos.
#
# ENFOQUE
# Este módulo NO intenta arreglar el dataset. Intenta responder cuatro preguntas:
# 1) ¿Se puede leer?
# 2) ¿Se puede confiar mínimamente en él?
# 3) ¿Qué problemas hay?
# 4) ¿Debemos bloquear, advertir o dejar pasar?
# =============================================================================


# =============================================================================
# Severidad y decisiones
# =============================================================================
SEVERITY_INFO = "info"
SEVERITY_WARNING = "warning"
SEVERITY_CRITICAL = "critical"

ACTION_OK = "ok"
ACTION_WARN = "warn"
ACTION_BLOCK = "block"


# =============================================================================
# Modelos
# =============================================================================
@dataclass(frozen=True)
class QualityIssue:
    """Problema detectado durante la validación de calidad."""

    code: str
    severity: str
    title: str
    message: str
    metric_key: str | None = None
    value: float | int | str | None = None
    threshold: float | int | str | None = None


@dataclass(frozen=True)
class DataQualitySummary:
    """
    Resumen final listo para UI, reglas y debug.

    read_confidence_score:
        Score 0..100 que resume calidad global de lectura.
    action:
        ok | warn | block
    owner_label:
        etiqueta simple para Owner.
    owner_message:
        texto corto y entendible.
    admin_message:
        resumen algo más técnico.
    """

    read_confidence_score: int
    action: str
    owner_label: str
    owner_message: str
    admin_message: str
    issues: list[QualityIssue]
    stats: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["issues"] = [asdict(x) for x in self.issues]
        return out


# =============================================================================
# Helpers internos
# =============================================================================
def _safe_series(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(dtype="object")
    return df[col]



def _ratio(num: int | float, den: int | float) -> float:
    try:
        den = float(den)
        if den <= 0:
            return 0.0
        return float(num) / den
    except Exception:
        return 0.0



def _to_numeric_ratio(s: pd.Series) -> float:
    if s is None or len(s) == 0:
        return 0.0
    try:
        coerced = pd.to_numeric(s, errors="coerce")
        return float(coerced.notna().mean())
    except Exception:
        return 0.0



def _missing_ratio(s: pd.Series) -> float:
    if s is None or len(s) == 0:
        return 1.0
    try:
        return float(s.isna().mean())
    except Exception:
        return 1.0



def _placeholder_ratio_producto(s: pd.Series) -> float:
    if s is None or len(s) == 0:
        return 1.0
    try:
        s2 = s.astype(str).fillna("").str.strip().str.lower()
        placeholders = {"", "-", "sin_producto", "nan", "none", "null"}
        return float(s2.isin(placeholders).mean())
    except Exception:
        return 1.0



def _negative_ratio(s: pd.Series) -> float:
    if s is None or len(s) == 0:
        return 0.0
    try:
        x = pd.to_numeric(s, errors="coerce")
        valid = x.dropna()
        if valid.empty:
            return 0.0
        return float((valid < 0).mean())
    except Exception:
        return 0.0



def _zero_ratio(s: pd.Series) -> float:
    if s is None or len(s) == 0:
        return 0.0
    try:
        x = pd.to_numeric(s, errors="coerce")
        valid = x.dropna()
        if valid.empty:
            return 0.0
        return float((valid == 0).mean())
    except Exception:
        return 0.0



def _duplicate_ratio(df: pd.DataFrame, cols: list[str]) -> float:
    usable = [c for c in cols if c in df.columns]
    if df.empty or not usable:
        return 0.0
    try:
        return float(df.duplicated(subset=usable).mean())
    except Exception:
        return 0.0



def _date_parse_ratio(s: pd.Series) -> float:
    if s is None or len(s) == 0:
        return 0.0
    try:
        parsed = pd.to_datetime(s, errors="coerce")
        return float(parsed.notna().mean())
    except Exception:
        return 0.0



def _date_span_days(s: pd.Series) -> int:
    if s is None or len(s) == 0:
        return 0
    try:
        parsed = pd.to_datetime(s, errors="coerce").dropna()
        if parsed.empty:
            return 0
        return int((parsed.max() - parsed.min()).days) + 1
    except Exception:
        return 0



def _build_issue(
    code: str,
    severity: str,
    title: str,
    message: str,
    *,
    metric_key: str | None = None,
    value: float | int | str | None = None,
    threshold: float | int | str | None = None,
) -> QualityIssue:
    return QualityIssue(
        code=code,
        severity=severity,
        title=title,
        message=message,
        metric_key=metric_key,
        value=value,
        threshold=threshold,
    )


# =============================================================================
# Reglas principales
# =============================================================================
def assess_data_quality(
    df: pd.DataFrame,
    meta: dict | None = None,
) -> DataQualitySummary:
    """
    Evalúa la calidad del dataset procesado.

    Reglas de diseño:
    - si faltan columnas estructurales, bloquear;
    - si la lectura parece muy débil, advertir fuerte o bloquear;
    - si hay señales raras pero aún aprovechables, advertir;
    - devolver stats reutilizables por app, owner y admin.
    """
    meta = meta or {}
    issues: list[QualityIssue] = []
    stats: dict[str, Any] = {}

    if df is None or df.empty:
        issues.append(
            _build_issue(
                "empty_dataset",
                SEVERITY_CRITICAL,
                "Dataset vacío",
                "No hay registros procesados para analizar.",
            )
        )
        return _finalize_summary(issues=issues, stats={"rows": 0})

    rows = int(len(df))
    cols = list(df.columns)
    stats["rows"] = rows
    stats["columns"] = cols

    required_cols = ["fecha", "producto", "cantidad", "revenue"]
    missing_required = [c for c in required_cols if c not in df.columns]
    stats["missing_required_columns"] = missing_required

    if missing_required:
        issues.append(
            _build_issue(
                "missing_required_columns",
                SEVERITY_CRITICAL,
                "Faltan columnas clave",
                f"Faltan columnas necesarias para un análisis fiable: {', '.join(missing_required)}.",
                value=", ".join(missing_required),
            )
        )

    s_fecha = _safe_series(df, "fecha")
    s_producto = _safe_series(df, "producto")
    s_cantidad = _safe_series(df, "cantidad")
    s_revenue = _safe_series(df, "revenue")
    s_precio = _safe_series(df, "precio_unitario")
    s_ticket = _safe_series(df, "ticket_id")

    fecha_parse_ratio = _date_parse_ratio(s_fecha)
    producto_missing_ratio = _missing_ratio(s_producto)
    producto_placeholder_ratio = _placeholder_ratio_producto(s_producto)
    cantidad_numeric_ratio = _to_numeric_ratio(s_cantidad)
    revenue_numeric_ratio = _to_numeric_ratio(s_revenue)
    precio_numeric_ratio = _to_numeric_ratio(s_precio) if not s_precio.empty else 0.0
    revenue_negative_ratio = _negative_ratio(s_revenue)
    revenue_zero_ratio = _zero_ratio(s_revenue)
    cantidad_negative_ratio = _negative_ratio(s_cantidad)
    duplicate_ratio = _duplicate_ratio(df, ["fecha", "producto", "cantidad", "revenue"])
    date_span_days = _date_span_days(s_fecha)
    ticket_coverage = 1.0 - _missing_ratio(s_ticket) if not s_ticket.empty else 0.0

    stats.update(
        {
            "fecha_parse_ratio": round(fecha_parse_ratio, 4),
            "producto_missing_ratio": round(producto_missing_ratio, 4),
            "producto_placeholder_ratio": round(producto_placeholder_ratio, 4),
            "cantidad_numeric_ratio": round(cantidad_numeric_ratio, 4),
            "revenue_numeric_ratio": round(revenue_numeric_ratio, 4),
            "precio_numeric_ratio": round(precio_numeric_ratio, 4),
            "revenue_negative_ratio": round(revenue_negative_ratio, 4),
            "revenue_zero_ratio": round(revenue_zero_ratio, 4),
            "cantidad_negative_ratio": round(cantidad_negative_ratio, 4),
            "duplicate_ratio": round(duplicate_ratio, 4),
            "date_span_days": int(date_span_days),
            "ticket_coverage": round(ticket_coverage, 4),
        }
    )

    # -------------------------------------------------------------------------
    # Reglas críticas
    # -------------------------------------------------------------------------
    if fecha_parse_ratio < 0.80:
        issues.append(
            _build_issue(
                "fecha_parse_critical",
                SEVERITY_CRITICAL,
                "Fechas poco fiables",
                "Una parte demasiado alta de las fechas no se ha interpretado bien. Mostrar evolución temporal sería engañoso.",
                metric_key="fecha_parse_ratio",
                value=round(fecha_parse_ratio, 3),
                threshold=0.80,
            )
        )

    if revenue_numeric_ratio < 0.85:
        issues.append(
            _build_issue(
                "revenue_numeric_critical",
                SEVERITY_CRITICAL,
                "Ingresos poco fiables",
                "La columna de ingresos no se ha leído bien en demasiados registros. Las métricas principales perderían credibilidad.",
                metric_key="revenue_numeric_ratio",
                value=round(revenue_numeric_ratio, 3),
                threshold=0.85,
            )
        )

    if producto_placeholder_ratio >= 0.70:
        issues.append(
            _build_issue(
                "producto_placeholder_critical",
                SEVERITY_CRITICAL,
                "Demasiados productos sin identificar",
                "La mayoría de las filas no tienen un nombre útil de producto o servicio. Las recomendaciones por negocio serían poco fiables.",
                metric_key="producto_placeholder_ratio",
                value=round(producto_placeholder_ratio, 3),
                threshold=0.70,
            )
        )

    # -------------------------------------------------------------------------
    # Reglas warning
    # -------------------------------------------------------------------------
    if rows < 15:
        issues.append(
            _build_issue(
                "few_rows",
                SEVERITY_WARNING,
                "Pocos registros",
                "Hay muy pocos registros para sacar conclusiones consistentes. El resumen puede servir como orientación, no como lectura sólida.",
                metric_key="rows",
                value=rows,
                threshold=15,
            )
        )

    if 0.80 <= fecha_parse_ratio < 0.95:
        issues.append(
            _build_issue(
                "fecha_parse_warning",
                SEVERITY_WARNING,
                "Fechas con lectura parcial",
                "Las fechas se han leído razonablemente, pero hay parte del periodo que podría estar incompleto o mal interpretado.",
                metric_key="fecha_parse_ratio",
                value=round(fecha_parse_ratio, 3),
                threshold=0.95,
            )
        )

    if 0.85 <= revenue_numeric_ratio < 0.97:
        issues.append(
            _build_issue(
                "revenue_numeric_warning",
                SEVERITY_WARNING,
                "Ingresos con lectura parcial",
                "La columna de ingresos parece usable, pero no totalmente limpia. Conviene interpretar los totales con algo de prudencia.",
                metric_key="revenue_numeric_ratio",
                value=round(revenue_numeric_ratio, 3),
                threshold=0.97,
            )
        )

    if cantidad_numeric_ratio < 0.90:
        issues.append(
            _build_issue(
                "cantidad_numeric_warning",
                SEVERITY_WARNING,
                "Cantidad poco consistente",
                "La columna de cantidad tiene demasiados valores no numéricos. Algunas métricas por unidades pueden ser poco fiables.",
                metric_key="cantidad_numeric_ratio",
                value=round(cantidad_numeric_ratio, 3),
                threshold=0.90,
            )
        )

    if 0.25 <= producto_placeholder_ratio < 0.70:
        issues.append(
            _build_issue(
                "producto_placeholder_warning",
                SEVERITY_WARNING,
                "Producto o servicio incompleto en muchas filas",
                "Hay bastantes filas sin nombre de producto o servicio útil. El detalle de mix y recomendaciones pierde precisión.",
                metric_key="producto_placeholder_ratio",
                value=round(producto_placeholder_ratio, 3),
                threshold=0.25,
            )
        )

    if revenue_negative_ratio > 0.10:
        issues.append(
            _build_issue(
                "negative_revenue_warning",
                SEVERITY_WARNING,
                "Muchos ingresos negativos",
                "Hay demasiados importes negativos para un flujo comercial normal. Puede haber devoluciones, abonos o errores de lectura no controlados.",
                metric_key="revenue_negative_ratio",
                value=round(revenue_negative_ratio, 3),
                threshold=0.10,
            )
        )

    if revenue_zero_ratio > 0.30:
        issues.append(
            _build_issue(
                "zero_revenue_warning",
                SEVERITY_WARNING,
                "Muchos ingresos a cero",
                "Una parte alta de las filas tiene ingresos a cero. Eso puede distorsionar ticket medio, ranking de productos y recomendaciones.",
                metric_key="revenue_zero_ratio",
                value=round(revenue_zero_ratio, 3),
                threshold=0.30,
            )
        )

    if cantidad_negative_ratio > 0.05:
        issues.append(
            _build_issue(
                "negative_quantity_warning",
                SEVERITY_WARNING,
                "Cantidades negativas frecuentes",
                "Se detectan cantidades negativas en demasiadas filas. Puede haber devoluciones o una estructura no prevista por el sistema actual.",
                metric_key="cantidad_negative_ratio",
                value=round(cantidad_negative_ratio, 3),
                threshold=0.05,
            )
        )

    if duplicate_ratio > 0.20:
        issues.append(
            _build_issue(
                "duplicate_ratio_warning",
                SEVERITY_WARNING,
                "Posibles duplicados altos",
                "Hay muchas filas repetidas en campos clave. Conviene revisar si el archivo trae duplicados o una exportación inestable.",
                metric_key="duplicate_ratio",
                value=round(duplicate_ratio, 3),
                threshold=0.20,
            )
        )

    if date_span_days <= 1:
        issues.append(
            _build_issue(
                "date_span_low",
                SEVERITY_WARNING,
                "Periodo demasiado corto",
                "El archivo cubre muy pocos días. La comparativa y la detección de patrones temporales será limitada.",
                metric_key="date_span_days",
                value=date_span_days,
                threshold=2,
            )
        )

    # -------------------------------------------------------------------------
    # Señales informativas útiles
    # -------------------------------------------------------------------------
    if ticket_coverage == 0.0:
        issues.append(
            _build_issue(
                "ticket_missing_info",
                SEVERITY_INFO,
                "Sin ticket_id",
                "No hay identificador de ticket. Algunas métricas de ticket y operación serán aproximadas.",
                metric_key="ticket_coverage",
                value=round(ticket_coverage, 3),
            )
        )
    elif ticket_coverage < 0.70:
        issues.append(
            _build_issue(
                "ticket_partial_info",
                SEVERITY_INFO,
                "Cobertura parcial de ticket_id",
                "El identificador de ticket existe, pero no cubre suficiente parte del archivo. Algunas métricas seguirán siendo aproximadas.",
                metric_key="ticket_coverage",
                value=round(ticket_coverage, 3),
                threshold=0.70,
            )
        )

    # Meta/pipeline si existe
    if meta:
        stats_meta = meta.get("stats") or {}
        mapping = meta.get("mapping") or {}
        mapping_collisions = mapping.get("mapping_collisions") or {}
        placeholders_meta = int(stats_meta.get("placeholders_producto", 0) or 0)
        collisions_count = (
            len(mapping_collisions) if isinstance(mapping_collisions, dict) else 0
        )

        stats["meta_placeholders_producto"] = placeholders_meta
        stats["meta_mapping_collisions_count"] = collisions_count
        stats["meta_revenue_source"] = stats_meta.get("revenue_source")
        stats["meta_price_mode"] = stats_meta.get("price_mode")

        if collisions_count > 0:
            issues.append(
                _build_issue(
                    "mapping_collisions_warning",
                    SEVERITY_WARNING,
                    "Colisiones en el mapeo",
                    "Se detectaron colisiones al normalizar columnas. El archivo se pudo procesar, pero conviene revisar la lectura.",
                    metric_key="meta_mapping_collisions_count",
                    value=collisions_count,
                    threshold=0,
                )
            )

    return _finalize_summary(issues=issues, stats=stats)


# =============================================================================
# Finalización del resumen
# =============================================================================
def _finalize_summary(
    issues: list[QualityIssue],
    stats: dict[str, Any],
) -> DataQualitySummary:
    critical_count = sum(1 for x in issues if x.severity == SEVERITY_CRITICAL)
    warning_count = sum(1 for x in issues if x.severity == SEVERITY_WARNING)
    info_count = sum(1 for x in issues if x.severity == SEVERITY_INFO)

    score = 100
    score -= critical_count * 35
    score -= warning_count * 12
    score -= info_count * 3
    score = max(0, min(100, score))

    if critical_count > 0:
        action = ACTION_BLOCK
        owner_label = "bloqueada"
        owner_message = (
            "Los datos cargados no son suficientemente fiables para mostrar métricas con credibilidad."
        )
        admin_message = (
            f"Calidad bloqueante: {critical_count} problema(s) críticos, "
            f"{warning_count} warning(s), {info_count} info(s)."
        )
    elif warning_count > 0:
        action = ACTION_WARN
        if score >= 80:
            owner_label = "media-alta"
            owner_message = (
                "La lectura es útil, pero hay detalles del archivo que conviene interpretar con prudencia."
            )
        elif score >= 60:
            owner_label = "media"
            owner_message = (
                "La lectura permite orientarse, pero algunas métricas o recomendaciones pueden perder precisión."
            )
        else:
            owner_label = "media-baja"
            owner_message = (
                "La lectura es aprovechable, pero tiene suficientes problemas como para revisar el archivo antes de confiar del todo."
            )
        admin_message = (
            f"Calidad con advertencias: {warning_count} warning(s), {info_count} info(s), score={score}/100."
        )
    else:
        action = ACTION_OK
        if score >= 95:
            owner_label = "alta"
            owner_message = "La lectura del periodo es sólida y permite una interpretación bastante fiable."
        else:
            owner_label = "alta-media"
            owner_message = "La lectura del periodo es buena, con pequeñas limitaciones no bloqueantes."
        admin_message = f"Calidad correcta: sin warnings críticos, score={score}/100."

    return DataQualitySummary(
        read_confidence_score=int(score),
        action=action,
        owner_label=owner_label,
        owner_message=owner_message,
        admin_message=admin_message,
        issues=issues,
        stats=stats,
    )


# =============================================================================
# Helpers de UI / integración
# =============================================================================
def should_block_dashboard(summary: DataQualitySummary) -> bool:
    """True si no deberíamos mostrar KPIs ni recomendaciones principales."""
    return summary.action == ACTION_BLOCK



def should_warn_dashboard(summary: DataQualitySummary) -> bool:
    """True si conviene mostrar warning visible antes del contenido."""
    return summary.action == ACTION_WARN



def owner_quality_caption(summary: DataQualitySummary) -> str:
    """Texto corto listo para Owner."""
    return (
        f"Calidad de lectura: {summary.owner_label}. "
        f"{summary.owner_message} "
        f"Confianza estimada: {summary.read_confidence_score}/100."
    )



def admin_quality_lines(summary: DataQualitySummary) -> list[str]:
    """Líneas explicables para Admin/debug."""
    lines = [summary.admin_message]
    for issue in summary.issues:
        metric = f" [{issue.metric_key}]" if issue.metric_key else ""
        lines.append(f"- {issue.severity.upper()}: {issue.title}{metric} -> {issue.message}")
    return lines