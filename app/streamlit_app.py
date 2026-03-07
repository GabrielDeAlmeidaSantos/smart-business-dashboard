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
from app.analytics.data_quality import (
    assess_data_quality,
    should_block_dashboard,
    should_warn_dashboard,
    owner_quality_caption,
    admin_quality_lines,
)
from app.analytics.onboarding import (
    assess_onboarding,
    onboarding_owner_caption,
    onboarding_admin_lines,
)
from app.analytics.learning import (
    build_learning_summary,
    learning_owner_caption,
    pick_next_best_learning_action,
)

# Pipeline (para uploader)
from src.pipeline import PipelinePaths, run_pipeline


# =============================================================================
# OBJETIVO DE ESTA VERSIÓN
# - Integrar calidad de datos, onboarding y aprendizaje de forma real.
# - Simplificar Owner: menos ruido, menos texto visible y menos detalle técnico.
# - Evitar comparativas engañosas si no existe periodo anterior equivalente.
# - Mostrar mejor qué es exacto y qué es aproximado.
# - Hacer que las recomendaciones se perciban como pruebas operativas, no solo consejos.
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
    return not df_prev.empty



def comparison_caption(has_prev: bool) -> str:
    if has_prev:
        return "Comparado con el periodo anterior de la misma duración."
    return (
        "Todavía no hay un periodo anterior equivalente con datos suficientes. "
        "Mostramos el periodo actual sin comparación para evitar conclusiones engañosas."
    )



def build_owner_summary(
    kpis_act,
    kpis_prev,
    has_prev_period: bool,
    peor_dia: dict,
    mejor_dia: dict,
    top_item: str,
) -> list[str]:
    """Máximo 3 mensajes visibles para mantener foco."""
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
    if top_df.empty or df_f.empty:
        return 0.0
    total = float(df_f["revenue"].sum()) if not df_f.empty else 0.0
    if total <= 0:
        return 0.0
    top_value = float(top_df.iloc[0]["ingresos"])
    return (top_value / total) * 100.0



def owner_metric_labels(granularity: str) -> tuple[str, str]:
    if granularity == "ticket":
        return ("Tickets", "Importe medio por ticket")
    return ("Operaciones registradas", "Importe medio por operación (aprox.)")



def exactness_caption(meta: dict) -> str:
    stats = meta.get("stats") or {}
    metric_exactness = stats.get("metric_exactness") or {}
    if not metric_exactness:
        return "No hay detalle suficiente sobre exactitud de métricas en este archivo."

    ticket_state = metric_exactness.get("ticket_medio", "aproximada")
    ops_state = metric_exactness.get("operaciones", "aproximada")
    revenue_state = metric_exactness.get("revenue_total", "exacta")

    return (
        f"Ingresos: {revenue_state}. "
        f"Operaciones: {ops_state}. "
        f"Ticket medio: {ticket_state}."
    )



def kpi_delta_or_none(current_value, previous_value, formatter=str):
    if previous_value is None or previous_value == 0:
        return None
    delta_abs = current_value - previous_value
    return f"{formatter(delta_abs)} ({pct_str(delta_abs, previous_value)})"



def render_learning_block(history: HistoryStore):
    st.divider()
    st.markdown("## Aprendizaje reciente y siguiente foco")
    st.caption("Resumen de lo último probado y hacia dónde conviene mirar ahora.")

    learning_summary = build_learning_summary(history_store=history, recent_limit=6, next_focus_limit=3)
    next_best = pick_next_best_learning_action(learning_summary)

    st.caption(learning_owner_caption(learning_summary))

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("### Lo último aprendido")
        if not learning_summary.insights:
            st.info("Todavía no hay suficiente aprendizaje guardado.")
        else:
            shown = 0
            for ins in learning_summary.insights:
                if ins.category not in {"win", "loss", "inconclusive", "habit"}:
                    continue
                with st.container(border=True):
                    st.write(f"**{ins.title}**")
                    st.caption(f"Confianza: {ins.confidence_label}")
                    st.write(ins.summary)
                    if ins.metric_label:
                        st.caption(f"Métrica: {ins.metric_label}")
                shown += 1
                if shown >= 3:
                    break
            if shown == 0:
                st.info("Aún no hay aprendizajes cerrados con suficiente claridad.")

    with c2:
        st.markdown("### Siguiente foco")
        if next_best is None:
            st.info("Aún no hay suficiente seguimiento para sugerir el siguiente foco con historial.")
        else:
            with st.container(border=True):
                st.write(f"**{next_best.title}**")
                st.caption(f"Tipo: {next_best.category} · Confianza: {next_best.confidence_label}")
                st.write(next_best.recommended_next_step)
                if next_best.evidence:
                    st.caption(next_best.evidence)


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

    known = available_client_ids()
    if known:
        client_id = st.selectbox("Cliente (ID)", options=known, index=0)
    else:
        client_id = st.text_input(
            "Cliente (ID)",
            value="default",
            help="Identificador del cliente (carpeta en data/clients/).",
        )

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

    st.subheader("Datos")
    dataset_mode = st.radio(
        "Fuente",
        options=["Cliente", "Global (demo)"],
        index=0,
        help="Cliente usa data/clients/{id}/processed. Global usa data/processed.",
    )

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
        st.caption("Si no hay ticket_id, parte de las métricas serán aproximadas.")


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

meta = load_metadata(str(RUTA_META)) if RUTA_META.exists() else {}
quality = assess_data_quality(df=df, meta=meta)
onboarding = assess_onboarding(df=df, meta=meta)

st.caption(
    f"Datos: {safe_date_str(df['fecha'].min())} → {safe_date_str(df['fecha'].max())}"
    f"  |  Registros: {len(df):,}".replace(",", ".")
)

if should_block_dashboard(quality):
    st.error(owner_quality_caption(quality))
    st.info(onboarding_owner_caption(onboarding))
    with st.expander("Ver detalle técnico", expanded=False):
        for line in admin_quality_lines(quality):
            st.write(line)
        for line in onboarding_admin_lines(onboarding):
            st.write(line)
    st.stop()

if should_warn_dashboard(quality):
    st.warning(owner_quality_caption(quality))

st.caption(onboarding_owner_caption(onboarding))
st.caption(exactness_caption(meta))


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


# ----------------------------
# Perfil detectado
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

conservative_factor = 1.0 if info_dias.get("granularity") == "ticket" else 0.5
ventas_peor = int(max(1, round(ventas_peor_raw * conservative_factor)))

fig_top, top_df = fig_top_servicios(df_f, int(top_n))
top_item = top_df.iloc[0]["producto"] if not top_df.empty else "tu servicio principal"

history_path = Path("data/clients") / client_id / "history.json"
history_path.parent.mkdir(parents=True, exist_ok=True)
history = HistoryStore(client_id=client_id, path=history_path)


# ----------------------------
# Owner tab
# ----------------------------
with tab_owner:
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

    st.divider()

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
        delta=kpi_delta_or_none(kpis_act.unidades, kpis_prev.unidades, formatter=lambda x: f"{int(x)}")
        if has_prev_period
        else None,
    )

    st.divider()

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
                "vertical_strength": recommendation_plan.meta.get("vertical_strength"),
            },
            item_payloads=recommendation_plan.top_history_payloads(),
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
        exp = rec_result.experiment
        low, high = card.estimated_impact_eur

        st.markdown(f"### {card.title}")
        st.caption(
            f"Impacto orientativo ({horizonte_label}): {eur(low)} – {eur(high)} · "
            f"Tiempo: {card.time_to_apply_min} min"
        )

        st.write(f"**Por qué ahora:** {card.why_now}")
        st.write(f"**Qué hacer:** {card.action}")
        st.caption(f"Dónde aplicarlo: {exp.where_to_apply}")

        with st.expander("Ver más contexto", expanded=False):
            st.write(f"**Hipótesis:** {exp.hypothesis}")
            st.write(f"**Qué mediremos:** {exp.primary_metric.label}")
            if exp.secondary_metrics:
                st.write("**Métricas secundarias:** " + ", ".join(x.label for x in exp.secondary_metrics))
            st.write(f"**Qué consideramos éxito:** {exp.success_rule_text}")
            st.write(f"**Ventana de revisión:** {exp.review_window_label}")
            if exp.implementation_steps:
                st.write("**Checklist:**")
                for step in exp.implementation_steps:
                    st.caption(f"- {step}")
            st.write(f"**Esfuerzo:** {exp.context.effort}")
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

        base_payload = {
            "primary_metric_key": exp.primary_metric.key,
            "primary_metric_label": exp.primary_metric.label,
            "hypothesis": exp.hypothesis,
            "where_to_apply": exp.where_to_apply,
            "success_rule_text": exp.success_rule_text,
            "review_window_label": exp.review_window_label,
            "duration_days": exp.duration_days,
        }

        if done:
            st.session_state["fb"][key]["ask_outcome"] = True
            if authorized_ and history_:
                history_.update_item(
                    period_key=period_key_,
                    insight_id=card.insight_id,
                    status="done",
                    outcome="unknown",
                    create_if_missing=True,
                    **base_payload,
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
                    **base_payload,
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
                    create_if_missing=True,
                    **base_payload,
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
                    create_if_missing=True,
                    **base_payload,
                )
                st.toast("Guardado ✅", icon="✅")
            else:
                st.warning("Para guardar seguimiento, introduce el código del negocio.")

        if st.session_state["fb"][key].get("ask_outcome"):
            st.divider()
            st.markdown("**¿Qué pasó al probarlo?**")

            o1, o2, o3 = st.columns(3)
            yes = o1.button("👍 Mejoró", use_container_width=True, key=f"out_yes_{period_key_}_{card.insight_id}")
            no = o2.button("👎 No mejoró", use_container_width=True, key=f"out_no_{period_key_}_{card.insight_id}")
            inc = o3.button("🤷 Inconcluso", use_container_width=True, key=f"out_inc_{period_key_}_{card.insight_id}")

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
                        create_if_missing=True,
                        **base_payload,
                    )
                    st.toast("Resultado guardado ✅", icon="✅")
                else:
                    st.warning("Para guardar seguimiento, introduce el código del negocio.")

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
        render_learning_block(history=history)

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
                        "revisión": it.get("review_due_at", ""),
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
        st.markdown("## Admin · Calidad y onboarding")
        for line in admin_quality_lines(quality):
            st.write(line)
        st.divider()
        for line in onboarding_admin_lines(onboarding):
            st.write(line)

        st.divider()
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
            placeholders = int(stats.get("rows_with_placeholder_producto", 0) or stats.get("placeholders_producto", 0) or 0)
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

            if stats.get("metric_exactness"):
                st.json(stats.get("metric_exactness"), expanded=False)

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