from pathlib import Path
from datetime import datetime

import pandas as pd
import plotly.express as px
import streamlit as st

from analytics.schema import ensure_schema
from analytics.features import compute_core_features
from analytics.profile import infer_business_profile
from analytics.insights import generate_insights, Insight
from analytics.history import HistoryStore
from analytics.recommender import rank_insights, pick_plan
from analytics.impact import estimate_extra_revenue_for_worst_day


# ------------------------------------
# Utilidades UI
# ------------------------------------
def eur(x: float) -> str:
    """Formato € estilo ES (63.539,00 €)."""
    try:
        x = float(x)
    except Exception:
        x = 0.0
    s = f"{x:,.2f}"
    s = s.replace(",", "X").replace(".", ",").replace("X", ".")
    return f"{s} €"


def eur_day_from_annual(annual: float) -> str:
    """Convierte un impacto anual a estimación €/día (aprox.)."""
    try:
        annual = float(annual)
    except Exception:
        annual = 0.0
    return eur(annual / 365.0)


def safe_date_str(dt) -> str:
    try:
        return pd.to_datetime(dt).date().isoformat()
    except Exception:
        return "-"


def pct_str(delta: float, base: float) -> str:
    """Devuelve '+3.2%' o '-1.1%' (si base=0 => '')."""
    if base is None or base == 0:
        return ""
    p = (delta / base) * 100.0
    sign = "+" if p >= 0 else ""
    return f"{sign}{p:.1f}%"


def compute_kpis(df_: pd.DataFrame) -> dict:
    """
    KPIs básicos.

    IMPORTANTE:
    - Asume 1 fila = 1 ticket/venta.
    - Si tu dataset son líneas de ticket, el cálculo debe agrupar por ticket_id.
    """
    if df_ is None or df_.empty:
        return {"ingresos": 0.0, "ventas": 0, "ticket": 0.0, "unidades": 0}

    ingresos = float(df_["revenue"].sum())
    ventas = int(len(df_))
    ticket = float(df_["revenue"].mean()) if ventas else 0.0
    unidades = float(df_["cantidad"].sum())
    return {"ingresos": ingresos, "ventas": ventas, "ticket": ticket, "unidades": unidades}


@st.cache_data(show_spinner=False)
def load_data(path: str) -> pd.DataFrame:
    """Carga datos ya procesados (parquet)."""
    return pd.read_parquet(path)


def compute_ingresos_por_dia(df_f: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Ingresos por día de semana + info (mejor/peor/gap/ventas_peor_rango)."""
    mapa_dias = {0: "Lunes", 1: "Martes", 2: "Miércoles", 3: "Jueves", 4: "Viernes", 5: "Sábado", 6: "Domingo"}

    df_sem = df_f.copy()
    df_sem["dia_semana"] = df_sem["fecha"].dt.dayofweek
    df_sem["dia_nombre"] = df_sem["dia_semana"].map(mapa_dias)

    ingresos_dia = (
        df_sem.groupby(["dia_semana", "dia_nombre"], as_index=False)["revenue"]
        .sum()
        .sort_values("dia_semana")
        .rename(columns={"revenue": "ingresos"})
    )

    peor = ingresos_dia.sort_values("ingresos").iloc[0]
    mejor = ingresos_dia.sort_values("ingresos", ascending=False).iloc[0]
    gap = float(mejor["ingresos"] - peor["ingresos"])

    ventas_por_dia = df_sem.groupby("dia_nombre", as_index=False).size().rename(columns={"size": "ventas"})
    ventas_peor_rango = int(ventas_por_dia.loc[ventas_por_dia["dia_nombre"] == peor["dia_nombre"], "ventas"].iloc[0])

    info = {
        "df_sem": df_sem,
        "ingresos_dia": ingresos_dia,
        "peor": peor,
        "mejor": mejor,
        "gap": gap,
        "ventas_peor_rango": ventas_peor_rango,
    }
    return ingresos_dia, info


def build_serie_tiempo(df_f: pd.DataFrame, agrupacion: str) -> pd.DataFrame:
    """Serie temporal agregada por día/semana/mes."""
    df_temp = df_f.copy()
    if agrupacion == "Día":
        df_temp["periodo"] = df_temp["fecha"].dt.floor("D")
    elif agrupacion == "Semana":
        df_temp["periodo"] = df_temp["fecha"].dt.to_period("W-MON").apply(lambda r: r.start_time)
    else:
        df_temp["periodo"] = df_temp["fecha"].dt.to_period("M").dt.to_timestamp()

    serie = (
        df_temp.groupby("periodo", as_index=False)["revenue"]
        .sum()
        .sort_values("periodo")
        .rename(columns={"revenue": "ingresos"})
    )
    return serie


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
    fig.update_traces(hovertemplate="<b>%{customdata[0]}</b><br>Ingresos: %{customdata[1]}<extra></extra>")
    fig.update_yaxes(title_text="Ingresos (€)")
    fig.update_xaxes(title_text="Periodo", tickformat=tickformat)
    if dtick is not None:
        fig.update_xaxes(dtick=dtick)
    return fig


def fig_top_servicios(df_f: pd.DataFrame, top_n: int):
    """Figura: top servicios por ingresos + tabla top."""
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
        title=f"Top {top_n} servicios por ingresos",
        custom_data=["hover_eur", "pct"],
    )
    fig.update_traces(
        hovertemplate="<b>%{y}</b><br>Ingresos: %{customdata[0]}<br>Peso: %{customdata[1]:.1f}%<extra></extra>"
    )
    fig.update_xaxes(title_text="Ingresos (€)")
    fig.update_yaxes(title_text="Servicio", autorange="reversed")
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
    fig.update_traces(hovertemplate="<b>%{x}</b><br>Ingresos: %{customdata[0]}<extra></extra>")
    fig.update_yaxes(title_text="Ingresos (€)")
    fig.update_xaxes(title_text="Día")
    return fig


def format_impact_range(rng: tuple[float, float]) -> str:
    low, high = rng
    if low <= 0 and high <= 0:
        return "—"
    if abs(high - low) < 1e-6:
        return eur(low)
    return f"{eur(low)} – {eur(high)}"


# ----------------------------
# Configuración
# ----------------------------
st.set_page_config(page_title="Resumen del Negocio", layout="wide")
st.title("Resumen del Negocio (ventas)")

RUTA_DATOS = Path("data/processed/ventas_limpias.parquet")

# ----------------------------
# Sidebar
# ----------------------------
with st.sidebar:
    st.header("Controles")
    modo_admin = st.toggle("Modo admin", value=False, key="modo_admin_v7")
    st.caption("Filtra fechas y mira el resumen en segundos.")
    st.divider()

    client_id = st.text_input("Cliente (ID)", value="default", help="Separa historial por cliente.")
    st.caption(f"Historial: data/clients/{client_id}/history.json")
    st.divider()

    if modo_admin:
        st.subheader("Flujo (admin)")
        st.caption("1) Sube/actualiza el Excel en `data/input/`")
        st.caption("2) Ejecuta el pipeline para regenerar el dashboard:")
        st.code("python src/pipeline.py", language="bash")


# ----------------------------
# Carga de datos
# ----------------------------
if not RUTA_DATOS.exists():
    st.warning("No hay datos procesados aún. Ejecuta primero: `python src/pipeline.py`")
    st.stop()

try:
    df = load_data(str(RUTA_DATOS))
    df = ensure_schema(df)
except Exception as e:
    st.error(str(e))
    st.stop()

if df.empty:
    st.warning("El dataset procesado está vacío tras normalizar fechas.")
    st.stop()

st.caption(
    f"Datos procesados: {safe_date_str(df['fecha'].min())} → {safe_date_str(df['fecha'].max())}"
    f"  |  Registros: {len(df):,}".replace(",", ".")
)

# ----------------------------
# Filtros
# ----------------------------
min_f, max_f = df["fecha"].min().date(), df["fecha"].max().date()

c_f1, c_f2, c_f3 = st.columns([2, 2, 1])
with c_f1:
    rango = st.date_input(
        "Rango de fechas",
        value=(min_f, max_f),
        min_value=min_f,
        max_value=max_f,
        key="rango_v7",
    )
with c_f2:
    agrupacion = st.selectbox("Agrupar ingresos por", ["Mes", "Semana", "Día"], index=0, key="agrupacion_v7")
with c_f3:
    top_n = st.selectbox("Top servicios", [5, 10, 15, 20], index=0, key="top_n_v7")

if not isinstance(rango, (list, tuple)) or len(rango) != 2:
    st.warning("Selecciona un rango de fechas válido (inicio y fin).")
    st.stop()

inicio = pd.to_datetime(rango[0])
fin = pd.to_datetime(rango[1]) + pd.Timedelta(days=1)
df_f = df[(df["fecha"] >= inicio) & (df["fecha"] < fin)].copy()

if df_f.empty:
    st.warning("No hay registros en ese rango de fechas.")
    st.stop()

# ----------------------------
# Comparativa periodo anterior
# ----------------------------
duracion = fin - inicio
inicio_prev = inicio - duracion
fin_prev = inicio
df_prev = df[(df["fecha"] >= inicio_prev) & (df["fecha"] < fin_prev)].copy()

kpis_act = compute_kpis(df_f)
kpis_prev = compute_kpis(df_prev)

delta_ing = kpis_act["ingresos"] - kpis_prev["ingresos"]
delta_ven = kpis_act["ventas"] - kpis_prev["ventas"]
delta_tic = kpis_act["ticket"] - kpis_prev["ticket"]
delta_uni = kpis_act["unidades"] - kpis_prev["unidades"]

st.caption("📌 Flechas verdes y rojas = comparación con el **periodo anterior** de la misma duración.")

# ----------------------------
# Base: días, perfil, core, historial
# ----------------------------
ingresos_dia, info_dias = compute_ingresos_por_dia(df_f)
peor_dia = info_dias["peor"]
mejor_dia = info_dias["mejor"]
gap = info_dias["gap"]
ventas_peor_rango = info_dias["ventas_peor_rango"]

profile = infer_business_profile(df_f)
core = compute_core_features(df_f)

history_path = Path(f"data/clients/{client_id}/history.json")
history = HistoryStore(history_path)

period_key = df_f["fecha"].max().strftime("%Y-%m")

# ==========================================================
# Perfil detectado
# ==========================================================
st.markdown("## Perfil detectado")
p1, p2, p3 = st.columns([2, 1, 1])

with p1:
    st.write(f"**Tipo:** {profile.business_type} | **Subtipo:** {profile.subtype}")
    st.caption("Evidencia: " + " · ".join(profile.evidence) if profile.evidence else "Evidencia: -")
with p2:
    st.metric("Confianza", f"{int(profile.confidence * 100)}%")
with p3:
    st.caption("Si no es correcto, ajusta datos o reglas del perfilado.")

st.divider()

# ==========================================================
# Oportunidad + Simulador (impacto corregido)
# ==========================================================
st.markdown("## Oportunidad detectada")
st.caption("No asume subir precios: asume vender un extra/pack para elevar el ticket.")

cS1, cS2 = st.columns([2, 1])
with cS1:
    subida_eur = st.slider(
        "Mejora de ticket medio en el día flojo (€)",
        min_value=0,
        max_value=10,
        value=3,
        step=1,
        key="subida_ticket_v7",
    )
with cS2:
    horizonte = st.selectbox("Horizonte", ["Mes", "Trimestre", "Año"], index=1, key="horizonte_v7")

impacto_h = estimate_extra_revenue_for_worst_day(df_f, core.peor_dia_nombre, float(subida_eur), horizonte)
impacto_mes = estimate_extra_revenue_for_worst_day(df_f, core.peor_dia_nombre, float(subida_eur), "Mes")
impacto_anual = estimate_extra_revenue_for_worst_day(df_f, core.peor_dia_nombre, float(subida_eur), "Año")

hero1, hero2, hero3 = st.columns([2, 1, 1])
with hero1:
    st.metric("Impacto estimado (escenario)", eur(impacto_h), delta="sin subir precios (pack/extra/upsell)")
    st.caption(
        f"Día flojo: **{core.peor_dia_nombre}** | "
        f"Tickets en rango (día flojo): **{ventas_peor_rango}** | "
        f"Ingresos día flojo (rango): **{eur(float(peor_dia['ingresos']))}**"
    )
with hero2:
    st.metric("Equivalente mensual", eur(impacto_mes))
    st.metric("Equivalente anual", eur(impacto_anual))
with hero3:
    st.metric("Equivalente €/día", eur_day_from_annual(impacto_anual))
    st.caption("Para que se sienta real en caja.")

st.info(
    f"Si el **{core.peor_dia_nombre}** sube el ticket medio **+{subida_eur}€**, "
    f"el impacto estimado sería **{eur(impacto_h)}** en **{horizonte.lower()}**."
)

st.divider()

# ==========================================================
# Motor: insights con impacto/dificultad/tiempo + ranking
# ==========================================================
insights = generate_insights(
    df_f=df_f,
    core=core,
    business_type=profile.business_type,
    uplift_eur_per_ticket=float(subida_eur),
    horizon=horizonte,
)

ranked = rank_insights(insights, history)
plan = pick_plan(ranked, k=3)

# Guardar plan del periodo (upsert) en admin (no automático)
# -> evita escribir historial sin querer.
if modo_admin:
    st.subheader("Admin: guardar plan del periodo")
    if st.button(f"Guardar/actualizar plan {period_key}"):
        history.upsert_period_plan(
            period_key=period_key,
            insight_ids=[x.insight.id for x in plan],
            meta={"generated_at": datetime.now().isoformat(), "business_type": profile.business_type},
        )
        st.success("Plan guardado/actualizado en historial.")

st.divider()

# ==========================================================
# Radar (cliente: sin score; admin: muestra score)
# ==========================================================
st.markdown("### Radar de oportunidades (priorizado)")

for r in ranked[:6]:
    ins = r.insight
    icon = "💰" if ins.impact_type == "revenue" else ("🛡️" if ins.impact_type == "risk" else "⏱️")

    c1, c2, c3, c4 = st.columns([2.2, 1.1, 1.0, 1.0])
    with c1:
        st.write(f"**{icon} {ins.title}**")
        st.caption(ins.evidence)
    with c2:
        st.write("**Impacto**")
        st.write(format_impact_range(ins.estimated_impact_eur))
    with c3:
        st.write("**Dificultad**")
        st.write(ins.effort.capitalize())
    with c4:
        st.write("**Tiempo**")
        st.write(f"{ins.time_to_apply_min} min")

    st.write(f"**Acción:** {ins.action_hint}")

    if modo_admin:
        st.caption(f"debug → score={r.score:.2f} | {r.reason} | kpi={ins.kpi_target}")

    st.divider()

# ==========================================================
# Resumen ejecutivo
# ==========================================================
st.markdown("### Resumen ejecutivo (en 10 segundos)")
cE1, cE2, cE3 = st.columns(3)
with cE1:
    st.metric("Día flojo", core.peor_dia_nombre)
    st.caption(f"Ingresos: {eur(float(peor_dia['ingresos']))}")
with cE2:
    st.metric("Mejor día", core.mejor_dia_nombre)
    st.caption(f"Ingresos: {eur(float(mejor_dia['ingresos']))}")
with cE3:
    st.metric("Gap real entre días", eur(core.gap_mejor_peor))
    st.caption(f"{core.mejor_dia_nombre} vs {core.peor_dia_nombre}")

st.divider()

# ==========================================================
# Plan dinámico (3 acciones) + Seguimiento
# ==========================================================
st.subheader("Plan de mejora (3 acciones) — dinámico")

cols = st.columns(3)
for i, (col, r) in enumerate(zip(cols, plan), start=1):
    ins = r.insight
    with col:
        st.success(f"{i}. {ins.title}")
        st.caption(ins.evidence)
        st.write(f"**Impacto:** {format_impact_range(ins.estimated_impact_eur)}")
        st.write(f"**Dificultad:** {ins.effort.capitalize()}  |  **Tiempo:** {ins.time_to_apply_min} min")
        st.write(f"**Acción:** {ins.action_hint}")

st.divider()

# Seguimiento (solo si existe plan guardado del periodo)
st.subheader("Seguimiento (para que no se repita y el sistema aprenda)")
st.caption("Marca si se aplicó y si funcionó. Esto afecta el ranking futuro.")

data_hist = history.load()
recs = data_hist.get("recommendations") or []
this_period = next((r for r in recs if r.get("period") == period_key), None)

if this_period is None:
    st.warning(
        f"No hay plan guardado para {period_key}. "
        "En modo admin, pulsa **Guardar/actualizar plan** para habilitar seguimiento."
    )
else:
    items = this_period.get("items") or []
    # Mapa insight_id -> Insight (para mostrar nombres)
    by_id = {ins.id: ins for ins in insights}

    for it in items:
        iid = str(it.get("insight_id"))
        ins = by_id.get(iid)

        title = ins.title if ins else iid
        st.markdown(f"**{title}**")

        c1, c2, c3 = st.columns([1, 1, 2])

        applied_key = f"applied_{period_key}_{iid}"
        outcome_key = f"outcome_{period_key}_{iid}"
        notes_key = f"notes_{period_key}_{iid}"

        applied_default = bool(it.get("applied", False))
        outcome_default = str(it.get("outcome", "unknown"))
        notes_default = str(it.get("notes", ""))

        with c1:
            applied = st.toggle("Aplicado", value=applied_default, key=applied_key)
        with c2:
            outcome = st.selectbox(
                "Resultado",
                ["unknown", "improved", "not_improved"],
                index=["unknown", "improved", "not_improved"].index(outcome_default)
                if outcome_default in ["unknown", "improved", "not_improved"]
                else 0,
                key=outcome_key,
                help="improved = mejoró el KPI objetivo; not_improved = no mejoró.",
            )
        with c3:
            notes = st.text_input("Notas (opcional)", value=notes_default, key=notes_key)

        if st.button(f"Guardar seguimiento: {title}", key=f"save_{period_key}_{iid}"):
            history.update_item(
                period_key=period_key,
                insight_id=iid,
                applied=applied,
                outcome=outcome,
                notes=notes,
            )
            st.success("Seguimiento guardado.")
        st.divider()

# ----------------------------
# KPIs principales
# ----------------------------
st.subheader("Contexto del rango (KPIs)")
k1, k2, k3, k4 = st.columns(4)

k1.metric(
    "Ingresos",
    eur(kpis_act["ingresos"]),
    delta=f"{eur(delta_ing)} ({pct_str(delta_ing, kpis_prev['ingresos'])})" if kpis_prev["ingresos"] else None,
)
k2.metric(
    "Ventas (tickets)",
    f"{kpis_act['ventas']}",
    delta=f"{delta_ven} ({pct_str(delta_ven, kpis_prev['ventas'])})" if kpis_prev["ventas"] else None,
)
k3.metric(
    "Ticket medio",
    eur(kpis_act["ticket"]),
    delta=f"{eur(delta_tic)} ({pct_str(delta_tic, kpis_prev['ticket'])})" if kpis_prev["ticket"] else None,
)
k4.metric(
    "Unidades",
    f"{int(kpis_act['unidades'])}",
    delta=f"{int(delta_uni)} ({pct_str(delta_uni, kpis_prev['unidades'])})" if kpis_prev["unidades"] else None,
)

st.divider()

# ----------------------------
# Visualización (soporte)
# ----------------------------
st.subheader("Visualización (soporte)")
serie = build_serie_tiempo(df_f, agrupacion)
st.plotly_chart(fig_ingresos_tiempo(serie, agrupacion), use_container_width=True)

fig_top, _ = fig_top_servicios(df_f, int(top_n))
st.plotly_chart(fig_top, use_container_width=True)

st.subheader("Días fuertes y días flojos")
st.plotly_chart(fig_ingresos_dia_semana(ingresos_dia), use_container_width=True)

cA, cB, cC = st.columns(3)
with cA:
    st.metric("Día más flojo (ingresos)", eur(float(peor_dia["ingresos"])))
    st.caption(f"Día: {core.peor_dia_nombre}")
with cB:
    st.metric("Mejor día (ingresos)", eur(float(mejor_dia["ingresos"])))
    st.caption(f"Día: {core.mejor_dia_nombre}")
with cC:
    st.metric("Diferencia (mejor - flojo)", eur(core.gap_mejor_peor))
    st.caption(f"{core.mejor_dia_nombre} vs {core.peor_dia_nombre}")

st.divider()

with st.expander("Ver detalle (opcional)"):
    st.dataframe(df_f.sort_values("fecha", ascending=False), use_container_width=True)