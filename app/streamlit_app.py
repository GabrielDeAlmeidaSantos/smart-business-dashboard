from __future__ import annotations

from datetime import datetime
from pathlib import Path
import json

import pandas as pd
import plotly.express as px
import streamlit as st

from app.analytics.schema import ensure_schema, safe_date_str
from app.analytics.kpis import (
    eur,
    pct_str,
    compute_kpis,
    ingresos_por_dia_semana,
    build_serie_tiempo,
)
from app.analytics.profile import detect_business_profile
from app.analytics.history_store import HistoryStore
from app.analytics.recommender import build_recommendation_plan

# Pipeline (para uploader)
from src.pipeline import PipelinePaths, run_pipeline


# =============================================================================
# OBJETIVO DE ESTA VERSIÓN
# - Simplificar Owner: menos ruido, menos texto visible y menos detalle técnico.
# - Evitar mostrar comparativas engañosas si no existe periodo anterior equivalente.
# - Quitar score del Owner y dejar detalles técnicos en expander/Admin.
# - Reforzar el bloque de aprendizaje reciente + siguiente foco.
# - Proteger la UI antes de procesar para no enseñar estados/KPIs incoherentes.
# =============================================================================


# ----------------------------
# Data loading
# ----------------------------
@st.cache_data(show_spinner=False)
def load_data(path: str) -> pd.DataFrame:
    """Carga datos ya procesados (parquet)."""
    return pd.read_parquet(path)


@st.cache_data(show_spinner=False)
def load_metadata(path: str) -> dict:
    """Carga metadata del pipeline (json)."""
    p = Path(path)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


# ----------------------------
# Auth (simple)
# ----------------------------
def is_authorized(client_id: str, code: str) -> bool:
    """Permite acceso de escritura usando secrets por client_id."""
    try:
        expected = st.secrets["clients"][client_id]
        return str(code).strip() == str(expected).strip()
    except Exception:
        return False


def available_client_ids() -> list[str]:
    """Lista de clientes conocidos en secrets, si existe."""
    try:
        clients = st.secrets.get("clients", {})
        if isinstance(clients, dict) and clients:
            return sorted([str(k) for k in clients.keys()])
    except Exception:
        pass
    return []


# ----------------------------
# Charts
# ----------------------------
def fig_ingresos_tiempo(serie: pd.DataFrame, agrupacion: str):
    """Figura: ingresos en el tiempo."""
    serie = serie.copy()

    if agrupacion == "Mes":
        tickformat, dtick = "%Y-%m", "M1"
        serie["hover_fecha"] = serie["periodo"].dt.strftime("%Y-%m")
    elif agrupacion == "Semana":
        tickformat, dtick = "%d %b", 7 * 24 * 60 * 60 * 1000
        serie["hover_fecha"] = serie["periodo"].dt.strftime("%d %b %Y")
    else:
        tickformat, dtick = "%d %b", None
        serie["hover_fecha"] = serie["periodo"].dt.strftime("%d %b %Y")

    serie["hover_eur"] = serie["ingresos"].apply(eur)

    fig = px.line(
        serie,
        x="periodo",
        y="ingresos",
        title="Ingresos en el tiempo",
        markers=True,
        custom_data=["hover_fecha", "hover_eur"],
    )
    fig.update_traces(
        hovertemplate="<b>%{customdata[0]}</b><br>Ingresos: %{customdata[1]}<extra></extra>"
    )
    fig.update_yaxes(title_text="Ingresos (€)")
    fig.update_xaxes(title_text="Periodo", tickformat=tickformat)
    if dtick is not None:
        fig.update_xaxes(dtick=dtick)
    return fig


def fig_top_servicios(df_f: pd.DataFrame, top_n: int):
    """Figura: top productos/servicios por ingresos + tabla top."""
    top = (
        df_f.groupby("producto", as_index=False)["revenue"]
        .sum()
        .sort_values("revenue", ascending=False)
        .head(int(top_n))
        .rename(columns={"revenue": "ingresos"})
    )

    top_total = float(df_f["revenue"].sum()) if not df_f.empty else 0.0
    top["pct"] = (top["ingresos"] / top_total * 100) if top_total else 0.0
    top["hover_eur"] = top["ingresos"].apply(eur)

    fig = px.bar(
        top,
        x="ingresos",
        y="producto",
        orientation="h",
        title=f"Top {top_n} productos/servicios por ingresos",
        custom_data=["hover_eur", "pct"],
    )
    fig.update_traces(
        hovertemplate="<b>%{y}</b><br>Ingresos: %{customdata[0]}<br>Peso: %{customdata[1]:.1f}%<extra></extra>"
    )
    fig.update_xaxes(title_text="Ingresos (€)")
    fig.update_yaxes(title_text="Producto/Servicio", autorange="reversed")
    return fig, top


def fig_ingresos_dia_semana(ingresos_dia: pd.DataFrame):
    """Figura: ingresos por día de semana."""
    ingresos_dia = ingresos_dia.copy()
    ingresos_dia["hover_eur"] = ingresos_dia["ingresos"].apply(eur)

    fig = px.bar(
        ingresos_dia,
        x="dia_nombre",
        y="ingresos",
        title="Ingresos por día de la semana",
        custom_data=["hover_eur"],
    )
    fig.update_traces(
        hovertemplate="<b>%{x}</b><br>Ingresos: %{customdata[0]}<extra></extra>"
    )
    fig.update_yaxes(title_text="Ingresos (€)")
    fig.update_xaxes(title_text="Día")
    return fig


# ----------------------------
# Text helpers (Owner)
# ----------------------------
def has_equivalent_previous_period(df_prev: pd.DataFrame) -> bool:
    """
    Consideramos comparación válida si existe un bloque anterior con datos reales.
    No basta con calcular KPIs: debe haber registros en el periodo equivalente anterior.
    """
    return not df_prev.empty



def comparison_caption(has_prev: bool) -> str:
    """Copy más honesto para la línea de comparación de KPIs."""
    if has_prev:
        return "Comparado con el periodo anterior de la misma duración."
    return (
        "Todavía no hay un periodo anterior equivalente con datos suficientes. "
        "Mostramos el periodo actual sin comparación para evitar conclusiones engañosas."
    )



def comparison_fallback_text(has_prev: bool) -> str:
    """Texto breve para el resumen ejecutivo cuando no hay base comparable."""
    if has_prev:
        return ""
    return (
        "Aún no hay base suficiente para comparar este rango con otro equivalente anterior. "
        "La lectura de hoy se centra en el estado actual del negocio."
    )



def build_owner_summary(
    kpis_act,
    kpis_prev,
    has_prev_period: bool,
    peor_dia: dict,
    mejor_dia: dict,
    top_item: str,
) -> list[str]:
    """
    Genera frases simples para que Owner entienda rápido qué está pasando.
    Máximo 3 mensajes visibles para mantener foco.
    """
    frases: list[str] = []

    if has_prev_period and kpis_prev.ingresos > 0:
        delta_pct = ((kpis_act.ingresos - kpis_prev.ingresos) / kpis_prev.ingresos) * 100
        if delta_pct >= 5:
            frases.append(f"Los ingresos suben un {delta_pct:.1f}% frente al periodo anterior equivalente.")
        elif delta_pct <= -5:
            frases.append(f"Los ingresos bajan un {abs(delta_pct):.1f}% frente al periodo anterior equivalente.")
        else:
            frases.append("Los ingresos están estables frente al periodo anterior equivalente.")
    else:
        frases.append("No comparamos contra otro periodo porque aún no hay referencia equivalente fiable.")

    frases.append(
        f"El día con peor rendimiento es {peor_dia['dia_nombre']} y el mejor es {mejor_dia['dia_nombre']}."
    )
    frases.append(
        f"La prioridad más clara es reforzar {peor_dia['dia_nombre']} usando como apoyo lo que ya funciona con {top_item}."
    )

    return frases



def top_item_share_pct(top_df: pd.DataFrame, df_f: pd.DataFrame) -> float:
    """Peso porcentual del top 1 sobre ingresos totales."""
    if top_df.empty or df_f.empty:
        return 0.0
    total = float(df_f["revenue"].sum()) if not df_f.empty else 0.0
    if total <= 0:
        return 0.0
    top_value = float(top_df.iloc[0]["ingresos"])
    return (top_value / total) * 100.0



def owner_metric_labels(granularity: str) -> tuple[str, str]:
    """Etiquetas más entendibles para Owner."""
    if granularity == "ticket":
        return ("Tickets", "Importe medio por ticket")
    return ("Operaciones registradas", "Importe medio por operación (aprox.)")



def dataset_readiness_label(meta: dict, granularity: str) -> tuple[str, str]:
    """Etiqueta simple de calidad de lectura para Owner."""
    stats = meta.get("stats") or {}
    placeholders = int(stats.get("placeholders_producto", 0) or 0)
    collisions = stats.get("mapping_collisions_count")
    if collisions is None:
        mapping = meta.get("mapping") or {}
        mapping_collisions = mapping.get("mapping_collisions") or {}
        collisions = len(mapping_collisions) if isinstance(mapping_collisions, dict) else 0

    if granularity == "ticket" and placeholders == 0 and int(collisions) == 0:
        return (
            "alta",
            "Lectura sólida del periodo. Las recomendaciones tienen buena base para este rango.",
        )
    if granularity == "ticket":
        return (
            "media",
            "Lectura correcta, aunque conviene revisar algunos detalles del archivo cargado.",
        )
    return (
        "media",
        "Lectura útil, pero parte de las estimaciones se ajustan con prudencia por el detalle disponible.",
    )



def kpi_delta_or_none(current_value, previous_value, formatter=str):
    """
    Devuelve delta legible solo si existe base comparable.
    Así evitamos mostrar None, 0 engañoso o deltas sin contexto.
    """
    if previous_value is None or previous_value == 0:
        return None
    delta_abs = current_value - previous_value
    return f"{formatter(delta_abs)} ({pct_str(delta_abs, previous_value)})"



def render_learning_block(history: HistoryStore, period_key: str):
    """
    Bloque más útil para Owner:
    - qué aprendimos recientemente
    - cuál debería ser el siguiente foco
    """
    st.divider()
    st.markdown("## Aprendizaje reciente y siguiente foco")
    st.caption("Resumen de lo último probado y hacia dónde conviene mirar ahora.")

    learnings = history.get_recent_learnings(limit=5)
    period_data = history.get_period(period_key) or {}
    items = period_data.get("items") or []

    pending_items = [
        it for it in items if it.get("status") in {"planned", "active", "already_doing"}
    ]
    last_useful_learning = next(
        (
            row
            for row in learnings
            if row.get("outcome") in {"improved", "not_improved", "inconclusive"}
        ),
        None,
    )

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("### Lo último aprendido")
        if not learnings:
            st.info("Todavía no hay aprendizajes guardados.")
        else:
            for row in learnings[:3]:
                with st.container(border=True):
                    st.write(f"**Acción:** {row.get('insight_id', '-')}")
                    outcome_map = {
                        "improved": "Mejoró",
                        "not_improved": "No mejoró",
                        "inconclusive": "Inconcluso",
                        "unknown": "Sin revisar",
                    }
                    st.caption(f"Resultado: {outcome_map.get(row.get('outcome', 'unknown'), 'Sin revisar')}")
                    if row.get("learning_note"):
                        st.write(row.get("learning_note"))
                    elif row.get("note"):
                        st.write(row.get("note"))
                    else:
                        st.caption("Sin nota de aprendizaje guardada.")

    with c2:
        st.markdown("### Siguiente foco")
        if pending_items:
            next_item = pending_items[0]
            with st.container(border=True):
                st.write(f"**Acción siguiente:** {next_item.get('insight_id', '-')}")
                st.caption(f"Estado actual: {next_item.get('status', 'planned')}")
                if next_item.get("primary_metric_label"):
                    st.write(f"**Qué mirar:** {next_item.get('primary_metric_label')}")
                if next_item.get("note"):
                    st.caption(f"Nota: {next_item.get('note')}")
                else:
                    st.caption("Todavía no hay observaciones añadidas a esta acción.")
        elif last_useful_learning is not None:
            with st.container(border=True):
                st.write("**Siguiente foco:** repetir o escalar lo que ya mostró señal.")
                st.caption(
                    f"Última referencia útil: {last_useful_learning.get('insight_id', '-')}, "
                    f"resultado: {last_useful_learning.get('outcome', 'unknown')}"
                )
                if last_useful_learning.get("learning_note"):
                    st.write(last_useful_learning.get("learning_note"))
        else:
            st.info("Aún no hay suficiente seguimiento para sugerir el siguiente foco con historial.")


# ----------------------------
# Configuración app
# ----------------------------
st.set_page_config(page_title="Resumen del Negocio", layout="wide")
st.title("Resumen del Negocio (ventas)")


# ----------------------------
# Sidebar (cliente + auth + dataset + uploader)
# ----------------------------
with st.sidebar:
    st.header("Controles")

    modo_admin = st.toggle("Modo admin", value=False)
    st.caption("Sube datos, filtra fechas y mira el resumen en segundos.")
    st.divider()

    # Cliente
    known = available_client_ids()
    if known:
        client_id = st.selectbox("Cliente (ID)", options=known, index=0)
    else:
        client_id = st.text_input(
            "Cliente (ID)",
            value="default",
            help="Identificador del cliente (carpeta en data/clients/).",
        )

    # Auth
    access_code = st.text_input(
        "Código del negocio",
        type="password",
        help="Activa guardado de plan y seguimiento para este negocio.",
    )
    authorized = is_authorized(client_id, access_code)

    autoguardar = st.toggle(
        "Autoguardar plan del periodo",
        value=True,
        help="Guarda el plan automáticamente si aún no existe para este periodo. No sobrescribe el seguimiento.",
    )

    if not authorized:
        st.info("Sin código: puedes ver el dashboard, pero no se guarda seguimiento.")

    st.divider()

    # Dataset selector
    st.subheader("Datos")
    dataset_mode = st.radio(
        "Fuente",
        options=["Cliente", "Global (demo)"],
        index=0,
        help="Cliente usa data/clients/{id}/processed. Global usa data/processed.",
    )

    # Uploader (solo modo Cliente)
    if dataset_mode == "Cliente":
        st.caption("Sube CSV/XLSX del cliente. Se guarda y se procesa.")
        uploaded = st.file_uploader(
            "Subir archivo",
            type=["csv", "xlsx", "xls"],
            accept_multiple_files=False,
        )

        if uploaded is not None:
            ext = Path(uploaded.name).suffix.lower()
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            input_dir = Path("data/clients") / client_id / "input"
            input_dir.mkdir(parents=True, exist_ok=True)
            saved_path = input_dir / f"upload_{ts}{ext}"
            saved_path.write_bytes(uploaded.getbuffer())

            st.success(f"Archivo guardado: {saved_path.name}")

            if st.button("Procesar ahora", use_container_width=True):
                try:
                    paths = PipelinePaths.for_client(client_id)
                    paths = PipelinePaths(
                        input_dir=paths.input_dir,
                        processed_dir=paths.processed_dir,
                        input_file=saved_path,
                        output_clean=paths.output_clean,
                        output_kpis=paths.output_kpis,
                        output_meta=paths.output_meta,
                        allow_negative_revenue=False,
                        dayfirst=True,
                    )
                    run_pipeline(paths)
                    st.cache_data.clear()
                    st.success("Procesado correctamente ✅")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error procesando: {e}")

    if modo_admin:
        st.divider()
        st.subheader("Admin")
        st.caption("Si prefieres consola:")
        st.code("python src/pipeline.py", language="bash")
        st.caption("Aviso: si no hay ticket_id, las métricas de ticket son aproximadas.")


# ----------------------------
# Resolver rutas de dataset
# ----------------------------
if dataset_mode == "Cliente":
    RUTA_DATOS = Path("data/clients") / client_id / "processed" / "ventas_limpias.parquet"
    RUTA_META = Path("data/clients") / client_id / "processed" / "metadata.json"
else:
    RUTA_DATOS = Path("data/processed/ventas_limpias.parquet")
    RUTA_META = Path("data/processed/metadata.json")


# ----------------------------
# Estado previo a carga
# ----------------------------
if not RUTA_DATOS.exists():
    st.warning(
        "No hay datos procesados todavía.\n\n"
        "- En modo Cliente: sube un archivo y pulsa **Procesar ahora**.\n"
        "- En modo Global (demo): usa la fuente demo para ver un ejemplo completo."
    )
    st.stop()


# ----------------------------
# Carga de datos
# ----------------------------
try:
    df = load_data(str(RUTA_DATOS))
    df = ensure_schema(df)
except Exception as e:
    st.error(str(e))
    st.stop()

if df.empty:
    st.warning("El dataset procesado está vacío tras normalizar. Revisa el archivo o el pipeline.")
    st.stop()

st.caption(
    f"Datos: {safe_date_str(df['fecha'].min())} → {safe_date_str(df['fecha'].max())}"
    f"  |  Registros: {len(df):,}".replace(",", ".")
)

meta = load_metadata(str(RUTA_META)) if RUTA_META.exists() else {}


# ----------------------------
# Tabs (Owner / Detalle / Admin)
# ----------------------------
tab_owner, tab_detalle, tab_admin = st.tabs(["Owner", "Detalle", "Admin"])


# ----------------------------
# Filtros (Owner + Detalle)
# ----------------------------
min_f, max_f = df["fecha"].min().date(), df["fecha"].max().date()

with tab_owner:
    c_f1, c_f2, c_f3 = st.columns([2, 2, 1])
    with c_f1:
        rango = st.date_input(
            "Rango de fechas",
            value=(min_f, max_f),
            min_value=min_f,
            max_value=max_f,
        )
    with c_f2:
        agrupacion = st.selectbox("Agrupar ingresos por", ["Mes", "Semana", "Día"], index=0)
    with c_f3:
        top_n = st.selectbox("Top productos/servicios", [5, 10, 15, 20], index=0)

if not isinstance(rango, (list, tuple)) or len(rango) != 2:
    st.warning("Selecciona un rango de fechas válido (inicio y fin).")
    st.stop()

inicio = pd.to_datetime(rango[0])
fin = pd.to_datetime(rango[1]) + pd.Timedelta(days=1)
end_inclusive = pd.to_datetime(rango[1])

df_f = df[(df["fecha"] >= inicio) & (df["fecha"] < fin)].copy()
if df_f.empty:
    st.warning("No hay registros en ese rango de fechas.")
    st.stop()

period_key = f"{inicio.date().isoformat()}_{end_inclusive.date().isoformat()}"


# ----------------------------
# KPIs + comparativa
# ----------------------------
duracion = fin - inicio
inicio_prev = inicio - duracion
fin_prev = inicio
df_prev = df[(df["fecha"] >= inicio_prev) & (df["fecha"] < fin_prev)].copy()

kpis_act = compute_kpis(df_f)
kpis_prev = compute_kpis(df_prev)
has_prev_period = has_equivalent_previous_period(df_prev)

# Solo calculamos deltas para uso interno; en Owner se muestran solo si la comparación es válida.
delta_ing = kpis_act.ingresos - kpis_prev.ingresos
delta_ven = kpis_act.ventas - kpis_prev.ventas
delta_tic = kpis_act.ticket - kpis_prev.ticket
delta_uni = kpis_act.unidades - kpis_prev.unidades


# ----------------------------
# Perfil detectado (solo para lógica interna)
# ----------------------------
profile = detect_business_profile(df_f)


# ----------------------------
# Día semana + base simulador
# ----------------------------
ingresos_dia, info_dias = ingresos_por_dia_semana(df_f)
peor_dia = info_dias["peor"]
mejor_dia = info_dias["mejor"]
gap = info_dias["gap"]
ventas_peor_raw = int(info_dias["ventas_peor"])

# Modo conservador si NO hay ticket_id
conservative_factor = 1.0 if info_dias.get("granularity") == "ticket" else 0.5
ventas_peor = int(max(1, round(ventas_peor_raw * conservative_factor)))

# Top para textos e insights
fig_top, top_df = fig_top_servicios(df_f, int(top_n))
top_item = top_df.iloc[0]["producto"] if not top_df.empty else "tu servicio principal"

# Store histórico
history_path = Path("data/clients") / client_id / "history.json"
history_path.parent.mkdir(parents=True, exist_ok=True)
history = HistoryStore(client_id=client_id, path=history_path)


# ----------------------------
# Owner tab
# ----------------------------
with tab_owner:
    # Resumen ejecutivo
    st.markdown("## Qué está pasando en el negocio")
    st.caption("Lectura rápida del periodo para entender dónde estás y qué conviene mover primero.")

    owner_summary = build_owner_summary(
        kpis_act=kpis_act,
        kpis_prev=kpis_prev,
        has_prev_period=has_prev_period,
        peor_dia=peor_dia,
        mejor_dia=mejor_dia,
        top_item=top_item,
    )
    for frase in owner_summary:
        st.write(f"- {frase}")

    fallback_note = comparison_fallback_text(has_prev_period)
    if fallback_note:
        st.info(fallback_note)

    if meta:
        calidad_lectura, lectura_msg = dataset_readiness_label(
            meta=meta,
            granularity=info_dias.get("granularity", "row"),
        )
        st.caption(f"Lectura del periodo: {calidad_lectura}. {lectura_msg}")

    st.divider()

    # KPIs
    st.markdown("## Resumen del periodo")
    st.caption(comparison_caption(has_prev_period))

    ops_label, avg_label = owner_metric_labels(info_dias.get("granularity", "row"))

    k1, k2, k3, k4 = st.columns(4)
    k1.metric(
        "Ingresos",
        eur(kpis_act.ingresos),
        delta=kpi_delta_or_none(kpis_act.ingresos, kpis_prev.ingresos, formatter=eur)
        if has_prev_period
        else None,
    )
    k2.metric(
        ops_label,
        f"{kpis_act.ventas}",
        delta=kpi_delta_or_none(kpis_act.ventas, kpis_prev.ventas, formatter=lambda x: f"{int(x)}")
        if has_prev_period
        else None,
    )
    k3.metric(
        avg_label,
        eur(kpis_act.ticket),
        delta=kpi_delta_or_none(kpis_act.ticket, kpis_prev.ticket, formatter=eur)
        if has_prev_period
        else None,
    )
    k4.metric(
        "Unidades",
        f"{int(kpis_act.unidades)}",
        delta=kpi_delta_or_none(
            kpis_act.unidades,
            kpis_prev.unidades,
            formatter=lambda x: f"{int(x)}",
        )
        if has_prev_period
        else None,
    )

    st.divider()

    # Oportunidad principal
    st.markdown("## La oportunidad más clara ahora")
    st.caption("Estimación orientativa basada en mejorar el día flojo sin depender de descuentos.")

    cS1, cS2 = st.columns([2, 1])
    with cS1:
        subida_eur = st.slider(
            f"Mejora de {avg_label} en el día flojo (€)",
            min_value=0,
            max_value=10,
            value=3,
            step=1,
        )
    with cS2:
        horizonte = st.selectbox(
            "Horizonte de referencia",
            ["1× rango", "3× rango", "12× rango"],
            index=1,
        )

    h_mult = {"1× rango": 1, "3× rango": 3, "12× rango": 12}[horizonte]

    impacto_rango = float(ventas_peor * subida_eur)
    impacto_horizonte = float(impacto_rango * h_mult)

    days_in_range = max(int((fin - inicio).days), 1)
    impacto_por_dia = impacto_rango / days_in_range
    impacto_anualizado = impacto_por_dia * 365

    hero1, hero2, hero3 = st.columns([2, 1, 1])
    with hero1:
        st.metric("Impacto orientativo", eur(impacto_horizonte))
        gran = info_dias.get("granularity", "row")
        if gran != "ticket":
            st.caption("Estimación ajustada de forma prudente al nivel de detalle disponible.")
        st.caption(
            f"Día flojo: {peor_dia['dia_nombre']} · Base: {ventas_peor_raw} "
            f"{('tickets' if gran == 'ticket' else 'registros')} (ajustado a {ventas_peor})."
        )
    with hero2:
        st.metric("Impacto en este rango", eur(impacto_rango))
        st.caption(f"Referencia usada: {horizonte}")
    with hero3:
        st.metric("Promedio diario", eur(impacto_por_dia))
        with st.expander("Ver anualizado", expanded=False):
            st.metric("Anualizado", eur(impacto_anualizado))

    st.info(
        f"Si {peor_dia['dia_nombre']} mejora +{subida_eur} € por operación, "
        f"el impacto orientativo sería {eur(impacto_horizonte)} tomando como referencia {horizonte}."
    )

    st.divider()

    # Contexto para recomendaciones
    delta_ing_pct = 0.0
    if has_prev_period and kpis_prev.ingresos and float(kpis_prev.ingresos) != 0.0:
        delta_ing_pct = (
            (float(kpis_act.ingresos) - float(kpis_prev.ingresos)) / float(kpis_prev.ingresos)
        ) * 100.0

    gran = info_dias.get("granularity", "row")
    granularity_label = "alta" if gran == "ticket" else "media"
    top_share_pct = top_item_share_pct(top_df, df_f)

    recommendation_ctx = {
        "worst_day": peor_dia["dia_nombre"],
        "best_day": mejor_dia["dia_nombre"],
        "worst_day_gap_eur": float(gap),
        "ventas_peor_dia": int(ventas_peor),
        "top_item": top_item,
        "slider_subida_ticket_eur": float(subida_eur),
        "ticket_label": avg_label,
        "delta_ing_pct": float(delta_ing_pct),
        "top_share_pct": float(top_share_pct),
        "granularity_label": granularity_label,
    }

    recommendation_plan = build_recommendation_plan(
        ctx=recommendation_ctx,
        business_type=profile.business_type,
        subtype=profile.subtype,
        history=history,
        client_id=client_id,
        period_key=period_key,
        top_k=3,
    )

    if autoguardar and authorized and (history.get_period(period_key) is None):
        history.upsert_period_plan(
            period_key=period_key,
            insight_ids=[x.card.insight_id for x in recommendation_plan.top_cards],
            meta={
                "generated_at": datetime.now().isoformat(timespec="seconds"),
                "business_type": profile.business_type,
                "subtype": profile.subtype,
                "horizon_label": str(horizonte),
                "horizon_multiplier": int(h_mult),
            },
        )

    def _render_action_card(
        rec_result,
        horizonte_label: str,
        authorized_: bool,
        history_: HistoryStore | None,
        period_key_: str,
        show_admin_details: bool = False,
    ):
        """
        Render de una recomendación orientada a experimento.
        En Owner solo mostramos lo esencial. Score y razones técnicas quedan ocultos.
        """
        card = rec_result.card
        low, high = card.estimated_impact_eur

        st.markdown(f"### {card.title}")
        st.caption(
            f"Impacto orientativo ({horizonte_label}): {eur(low)} – {eur(high)} · "
            f"Tiempo: {card.time_to_apply_min} min"
        )

        st.write(f"**Por qué ahora:** {card.why_now}")
        st.write(f"**Qué hacer:** {card.action}")
        st.caption(f"Dónde aplicarlo: {card.where_to_apply}")

        with st.expander("Ver más contexto", expanded=False):
            st.write(f"**Objetivo:** {card.goal}")
            st.write(f"**Hipótesis:** {card.hypothesis}")
            st.write(f"**Qué mediremos:** {card.primary_metric_label}")

            if card.secondary_metric_labels:
                st.write("**Métricas secundarias:** " + ", ".join(card.secondary_metric_labels))

            st.write(f"**Qué consideramos éxito:** {card.success_rule_text}")
            st.write(f"**Si ya lo haces:** {card.if_already_doing}")
            st.write(f"**Estrategia sugerida:** {card.strategy}")
            st.write(f"**Esfuerzo:** {card.effort}")
            st.write(f"**Prioridad:** {card.priority_label}")
            st.write(f"**Confianza:** {card.confidence_label}")

            if show_admin_details:
                st.caption(f"Score interno: {rec_result.score:.2f}")
                st.caption(f"Motivo del ranking: {rec_result.reason}")

        row1_col1, row1_col2 = st.columns(2)
        done = row1_col1.button(
            "✅ Hecho",
            use_container_width=True,
            key=f"done_{period_key_}_{card.insight_id}",
        )
        skip = row1_col2.button(
            "↩️ Saltar",
            use_container_width=True,
            key=f"skip_{period_key_}_{card.insight_id}",
        )

        row2_col1, row2_col2 = st.columns(2)
        later = row2_col1.button(
            "🕒 Luego",
            use_container_width=True,
            key=f"later_{period_key_}_{card.insight_id}",
        )
        already = row2_col2.button(
            "🔁 Ya lo hacemos",
            use_container_width=True,
            key=f"already_{period_key_}_{card.insight_id}",
        )

        st.session_state.setdefault("fb", {})
        key = f"{period_key_}:{card.insight_id}"
        st.session_state["fb"].setdefault(key, {"ask_outcome": False})

        if done:
            st.session_state["fb"][key]["ask_outcome"] = True
            if authorized_ and history_:
                history_.update_item(
                    period_key=period_key_,
                    insight_id=card.insight_id,
                    status="done",
                    outcome="unknown",
                    primary_metric_key=card.primary_metric_key,
                    primary_metric_label=card.primary_metric_label,
                    create_if_missing=True,
                )
                st.toast("Guardado ✅", icon="✅")
            else:
                st.warning("Para guardar seguimiento, introduce el código del negocio.")

        if skip:
            st.session_state["fb"][key]["ask_outcome"] = False
            if authorized_ and history_:
                history_.update_item(
                    period_key=period_key_,
                    insight_id=card.insight_id,
                    status="skipped",
                    outcome="unknown",
                    create_if_missing=True,
                )
                st.toast("Guardado ✅", icon="✅")
            else:
                st.warning("Para guardar seguimiento, introduce el código del negocio.")

        if later:
            st.session_state["fb"][key]["ask_outcome"] = False
            if authorized_ and history_:
                history_.update_item(
                    period_key=period_key_,
                    insight_id=card.insight_id,
                    status="active",
                    outcome="unknown",
                    primary_metric_key=card.primary_metric_key,
                    primary_metric_label=card.primary_metric_label,
                    create_if_missing=True,
                )
                st.toast("Guardado ✅", icon="✅")
            else:
                st.warning("Para guardar seguimiento, introduce el código del negocio.")

        if already:
            st.session_state["fb"][key]["ask_outcome"] = False
            if authorized_ and history_:
                history_.update_item(
                    period_key=period_key_,
                    insight_id=card.insight_id,
                    status="already_doing",
                    outcome="unknown",
                    note="Ya lo hacen en el negocio",
                    primary_metric_key=card.primary_metric_key,
                    primary_metric_label=card.primary_metric_label,
                    create_if_missing=True,
                )
                st.toast("Guardado ✅", icon="✅")
            else:
                st.warning("Para guardar seguimiento, introduce el código del negocio.")

        if st.session_state["fb"][key].get("ask_outcome"):
            st.divider()
            st.markdown("**¿Qué pasó al probarlo?**")

            o1, o2, o3 = st.columns(3)
            yes = o1.button(
                "👍 Mejoró",
                use_container_width=True,
                key=f"out_yes_{period_key_}_{card.insight_id}",
            )
            no = o2.button(
                "👎 No mejoró",
                use_container_width=True,
                key=f"out_no_{period_key_}_{card.insight_id}",
            )
            inc = o3.button(
                "🤷 Inconcluso",
                use_container_width=True,
                key=f"out_inc_{period_key_}_{card.insight_id}",
            )

            note = st.text_input(
                "Aprendizaje / nota (opcional)",
                key=f"note_{period_key_}_{card.insight_id}",
                placeholder="Ej: funcionó mejor por la tarde, pero no fue constante...",
            )

            chosen = None
            if yes:
                chosen = "improved"
            elif no:
                chosen = "not_improved"
            elif inc:
                chosen = "inconclusive"

            if chosen is not None:
                st.session_state["fb"][key]["ask_outcome"] = False
                if authorized_ and history_:
                    history_.update_item(
                        period_key=period_key_,
                        insight_id=card.insight_id,
                        status="done",
                        outcome=chosen,
                        note=note,
                        learning_note=note,
                        reviewed_at=datetime.now().isoformat(timespec="seconds"),
                        primary_metric_key=card.primary_metric_key,
                        primary_metric_label=card.primary_metric_label,
                        create_if_missing=True,
                    )
                    st.toast("Resultado guardado ✅", icon="✅")
                else:
                    st.warning("Para guardar seguimiento, introduce el código del negocio.")

    # Acciones del día
    st.markdown("## Acciones recomendadas para hoy")
    st.caption("Priorizadas por impacto potencial, facilidad de aplicación y contexto del negocio.")

    if not authorized:
        st.caption("🔒 Sin código puedes ver recomendaciones, pero no guardar progreso.")

    if not recommendation_plan.top_cards:
        st.info("Todavía no hay recomendaciones activas para este periodo con las reglas actuales.")
    else:
        action_cols = st.columns(len(recommendation_plan.top_cards))
        for col, rec_result in zip(action_cols, recommendation_plan.top_cards):
            with col:
                with st.container(border=True):
                    _render_action_card(
                        rec_result=rec_result,
                        horizonte_label=str(horizonte),
                        authorized_=authorized,
                        history_=history if authorized else None,
                        period_key_=period_key,
                        show_admin_details=False,
                    )

    if authorized:
        render_learning_block(history=history, period_key=period_key)

        # Seguimiento
        st.divider()
        st.markdown("## Seguimiento de acciones")
        st.caption("Estado guardado para este periodo.")

        period_data = history.get_period(period_key)
        if period_data is None:
            st.info("Todavía no hay plan guardado para este periodo.")
        else:
            items = period_data.get("items") or []
            if not items:
                st.info("El plan existe, pero todavía no hay acciones registradas.")
            else:
                rows = [
                    {
                        "insight_id": it.get("insight_id", "-"),
                        "estado": it.get("status", "planned"),
                        "resultado": it.get("outcome", "unknown"),
                        "métrica_principal": it.get("primary_metric_label", ""),
                        "actualizado": it.get("updated_at", "-"),
                        "nota": it.get("note", ""),
                        "aprendizaje": it.get("learning_note", ""),
                    }
                    for it in items
                ]
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ----------------------------
# Detalle tab
# ----------------------------
with tab_detalle:
    st.markdown("## Visualización (detalle)")
    serie = build_serie_tiempo(df_f, agrupacion)
    st.plotly_chart(fig_ingresos_tiempo(serie, agrupacion), use_container_width=True)

    fig_top, _ = fig_top_servicios(df_f, int(top_n))
    st.plotly_chart(fig_top, use_container_width=True)

    st.subheader("Días fuertes y días flojos")
    st.plotly_chart(fig_ingresos_dia_semana(ingresos_dia), use_container_width=True)

    cA, cB, cC = st.columns(3)
    with cA:
        st.metric("Día más flojo (ingresos)", eur(peor_dia["ingresos"]))
        st.caption(f"Día: {peor_dia['dia_nombre']}")
    with cB:
        st.metric("Mejor día (ingresos)", eur(mejor_dia["ingresos"]))
        st.caption(f"Día: {mejor_dia['dia_nombre']}")
    with cC:
        st.metric("Diferencia (mejor - flojo)", eur(gap))
        st.caption(f"{mejor_dia['dia_nombre']} vs {peor_dia['dia_nombre']}")

    with st.expander("Ver detalle de registros", expanded=False):
        st.dataframe(df_f.sort_values("fecha", ascending=False), use_container_width=True)


# ----------------------------
# Admin tab
# ----------------------------
with tab_admin:
    if not modo_admin:
        st.info("Activa **Modo admin** en la barra lateral para ver debug.")
    else:
        st.markdown("## Admin · Metadata del pipeline (debug)")
        if not meta:
            st.warning("No se encontró metadata.json para este dataset.")
        else:
            stats = meta.get("stats") or {}
            mapping = meta.get("mapping") or {}
            mapping_reverse = mapping.get("mapping_reverse") or {}
            mapping_collisions = mapping.get("mapping_collisions") or {}

            cM1, cM2, cM3 = st.columns(3)
            with cM1:
                st.metric("price_mode", str(stats.get("price_mode", "-")))
                st.metric("revenue_source", str(stats.get("revenue_source", "-")))
            with cM2:
                st.metric("rows_raw", int(meta.get("rows_raw", 0)))
                st.metric("rows_clean", int(meta.get("rows_clean", 0)))
            with cM3:
                st.metric("date_min", str(meta.get("date_min", "-")))
                st.metric("date_max", str(meta.get("date_max", "-")))

            st.divider()
            st.markdown("**Calidad de lectura / normalización**")
            placeholders = int(stats.get("placeholders_producto", 0) or 0)
            cQ1, cQ2, cQ3 = st.columns(3)
            with cQ1:
                st.metric("Fuente revenue", str(stats.get("revenue_source", "-")))
            with cQ2:
                st.metric("Placeholders producto", f"{placeholders}")
            with cQ3:
                st.metric(
                    "Colisiones mapping",
                    f"{len(mapping_collisions) if isinstance(mapping_collisions, dict) else 0}",
                )

            sanity = stats.get("sanity") or {}
            if sanity:
                s_rev = sanity.get("revenue") or {}
                s_qty = sanity.get("cantidad") or {}

                st.markdown("**Sanity (percentiles)**")
                s1, s2 = st.columns(2)
                with s1:
                    st.write(
                        f"Revenue p50: **{eur(s_rev.get('p50', 0.0) or 0.0)}**  |  "
                        f"p95: **{eur(s_rev.get('p95', 0.0) or 0.0)}**"
                    )
                    st.caption(
                        f"min={eur(s_rev.get('min', 0.0) or 0.0)} · "
                        f"max={eur(s_rev.get('max', 0.0) or 0.0)}"
                    )
                with s2:
                    st.write(
                        f"Cantidad p50: **{(s_qty.get('p50', 0.0) or 0.0):.2f}**  |  "
                        f"p95: **{(s_qty.get('p95', 0.0) or 0.0):.2f}**"
                    )
                    st.caption(
                        f"min={(s_qty.get('min', 0.0) or 0.0):.2f} · "
                        f"max={(s_qty.get('max', 0.0) or 0.0):.2f}"
                    )

            if mapping_reverse:
                st.markdown("**Columnas detectadas (original → estándar)**")
                original_to_std = {str(orig): str(std) for std, orig in mapping_reverse.items()}
                st.json(original_to_std, expanded=False)

            if isinstance(mapping_collisions, dict) and mapping_collisions:
                st.markdown("**Colisiones detectadas (revisar)**")
                st.json(mapping_collisions, expanded=False)

        st.divider()
        st.markdown("## Admin · Recomendaciones activadas")
        st.caption("Vista completa del motor actual")

        try:
            for rec_result in recommendation_plan.all_cards[:15]:
                with st.container(border=True):
                    _render_action_card(
                        rec_result=rec_result,
                        horizonte_label=str(horizonte),
                        authorized_=False,
                        history_=None,
                        period_key_=f"admin_preview_{period_key}",
                        show_admin_details=True,
                    )
        except Exception:
            st.info("Las recomendaciones se muestran cuando se genera el Owner.")