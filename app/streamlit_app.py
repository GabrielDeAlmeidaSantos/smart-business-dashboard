from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

# ------------------------------------
# Formato de dinero (ES)
# ------------------------------------
def eur(x: float) -> str:
    s = f"{float(x):,.2f}"
    s = s.replace(",", "X").replace(".", ",").replace("X", ".")
    return f"{s} €"


def safe_date_str(dt) -> str:
    try:
        return pd.to_datetime(dt).date().isoformat()
    except Exception:
        return "-"


def compute_kpis(df_: pd.DataFrame) -> dict:
    """KPIs básicos sin suposiciones."""
    if df_ is None or df_.empty:
        return {"ingresos": 0.0, "ventas": 0, "ticket": 0.0, "unidades": 0}

    ingresos = float(df_["revenue"].sum())
    ventas = int(len(df_))
    ticket = float(df_["revenue"].mean()) if ventas else 0.0
    unidades = int(df_["cantidad"].sum())
    return {"ingresos": ingresos, "ventas": ventas, "ticket": ticket, "unidades": unidades}


@st.cache_data(show_spinner=False)
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    return df


def ensure_schema(df: pd.DataFrame) -> pd.DataFrame:
    required_cols = {"fecha", "revenue", "cantidad", "producto"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas en el dataset procesado: {sorted(missing)}")

    df = df.copy()
    df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
    df = df.dropna(subset=["fecha"]).sort_values("fecha")
    return df


def compute_ingresos_por_dia(df_f: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Devuelve tabla de ingresos por día ordenada y diccionario útil (mejor/peor/gap)."""
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

    ventas_por_dia = (
        df_sem.groupby("dia_nombre", as_index=False)
        .size()
        .rename(columns={"size": "ventas"})
    )
    ventas_peor = int(ventas_por_dia.loc[ventas_por_dia["dia_nombre"] == peor["dia_nombre"], "ventas"].iloc[0])

    info = {
        "df_sem": df_sem,
        "ingresos_dia": ingresos_dia,
        "peor": peor,
        "mejor": mejor,
        "gap": gap,
        "ventas_peor": ventas_peor,
    }
    return ingresos_dia, info


def build_serie_tiempo(df_f: pd.DataFrame, agrupacion: str) -> pd.DataFrame:
    df_temp = df_f.copy()

    if agrupacion == "Día":
        df_temp["periodo"] = df_temp["fecha"].dt.floor("D")
    elif agrupacion == "Semana":
        df_temp["periodo"] = df_temp["fecha"].dt.to_period("W-MON").apply(lambda r: r.start_time)
    else:  # Mes
        df_temp["periodo"] = df_temp["fecha"].dt.to_period("M").dt.to_timestamp()

    serie = (
        df_temp.groupby("periodo", as_index=False)["revenue"]
        .sum()
        .sort_values("periodo")
        .rename(columns={"revenue": "ingresos"})
    )
    return serie


def fig_ingresos_tiempo(serie: pd.DataFrame, agrupacion: str):
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


# ----------------------------
# Configuración de la página
# ----------------------------
st.set_page_config(page_title="Resumen del Negocio", layout="wide")
st.title("Resumen del Negocio (ventas)")

RUTA_DATOS = Path("data/processed/ventas_limpias.parquet")

# ----------------------------
# Sidebar (modo cliente por defecto)
# ----------------------------
with st.sidebar:
    st.header("Controles")
    modo_admin = st.toggle("Modo admin", value=False, key="modo_admin_v3")
    st.caption("Filtra fechas y mira el resumen en segundos.")
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
# Controles (filtros)
# ----------------------------
min_f, max_f = df["fecha"].min().date(), df["fecha"].max().date()

c_f1, c_f2, c_f3 = st.columns([2, 2, 1])
with c_f1:
    rango = st.date_input(
        "Rango de fechas",
        value=(min_f, max_f),
        min_value=min_f,
        max_value=max_f,
        key="rango_v3",
    )
with c_f2:
    agrupacion = st.selectbox(
        "Agrupar ingresos por",
        ["Mes", "Semana", "Día"],
        index=0,
        key="agrupacion_v3",
    )
with c_f3:
    top_n = st.selectbox(
        "Top servicios",
        [5, 10, 15, 20],
        index=0,
        key="top_n_v3",
    )

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
# Periodo anterior (misma duración) para comparar
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

# ----------------------------
# KPIs principales (con comparación)
# ----------------------------
k1, k2, k3, k4 = st.columns(4)
k1.metric("Ingresos", eur(kpis_act["ingresos"]), delta=eur(delta_ing) if kpis_prev["ingresos"] else None)
k2.metric("Ventas (tickets)", f"{kpis_act['ventas']}", delta=f"{delta_ven}" if kpis_prev["ventas"] else None)
k3.metric("Ticket medio", eur(kpis_act["ticket"]), delta=eur(delta_tic) if kpis_prev["ticket"] else None)
k4.metric("Unidades", f"{kpis_act['unidades']}", delta=f"{delta_uni}" if kpis_prev["unidades"] else None)

st.divider()

# ----------------------------
# Cálculos reutilizables (día semana)
# ----------------------------
ingresos_dia, info_dias = compute_ingresos_por_dia(df_f)
peor_dia = info_dias["peor"]
mejor_dia = info_dias["mejor"]
gap = info_dias["gap"]
ventas_peor = info_dias["ventas_peor"]

# ----------------------------
# Resumen ejecutivo (modo venta)
# ----------------------------
st.subheader("Resumen ejecutivo")

cE1, cE2, cE3 = st.columns(3)
with cE1:
    st.metric("Día flojo", peor_dia["dia_nombre"])
    st.caption(f"Ingresos: {eur(peor_dia['ingresos'])}")
with cE2:
    st.metric("Mejor día", mejor_dia["dia_nombre"])
    st.caption(f"Ingresos: {eur(mejor_dia['ingresos'])}")
with cE3:
    st.metric("Oportunidad (Gap)", eur(gap))
    st.caption(f"{mejor_dia['dia_nombre']} vs {peor_dia['dia_nombre']}")

st.write(
    f"**Recomendación rápida:** concentra una acción comercial el **{peor_dia['dia_nombre']}** "
    f"(upsell/pack/recordatorio). Con **{ventas_peor}** ventas ese día, pequeñas mejoras escalan rápido."
)
st.divider()

# ----------------------------
# Simulador rápido (escenario) - arriba para convertir
# ----------------------------
st.subheader("Simulador rápido (escenario)")
st.caption("Estimación basada en tu ticket medio y el volumen de ventas del día más flojo.")

cS1, cS2 = st.columns([2, 1])
with cS1:
    subida_eur = st.slider("Subida de ticket medio en el día flojo (€)", 0, 10, 3, key="subida_ticket_v3")
with cS2:
    horizonte = st.selectbox("Horizonte", ["Mes", "Trimestre"], index=1, key="horizonte_v3")

mult = 1 if horizonte == "Mes" else 3
impacto = float(ventas_peor * subida_eur * mult)

st.info(
    f"Si el **{peor_dia['dia_nombre']}** sube el ticket medio **+{subida_eur}€**, "
    f"el impacto estimado sería **{eur(impacto)}** en **{horizonte.lower()}**."
)
st.caption("Ejemplo de acción: pack (servicio + extra), upsell en caja, recordatorio por WhatsApp o ajuste de precios.")

st.divider()

# ----------------------------
# Top + Plan de mejora (acciones)
# ----------------------------
fig_top, top_df = fig_top_servicios(df_f, int(top_n))
top_item = top_df.iloc[0]["producto"] if not top_df.empty else "tu servicio principal"

st.subheader("Plan de mejora (3 acciones)")
a1, a2, a3 = st.columns(3)

with a1:
    st.success(f"1) Empujar **{top_item}** el **{peor_dia['dia_nombre']}**.")
    st.write(f"Por qué: es el día con menos ingresos (**{eur(peor_dia['ingresos'])}**).")
    st.write(f"Impacto (escenario): +{subida_eur}€ por ticket ⇒ **{eur(impacto)}** en {horizonte.lower()}.")

with a2:
    st.warning("2) Reducir dependencia del Top 3.")
    top3 = (
        df_f.groupby("producto", as_index=False)["revenue"]
        .sum()
        .sort_values("revenue", ascending=False)
        .head(3)["revenue"]
        .sum()
    )
    pct_top3 = (top3 / kpis_act["ingresos"] * 100) if kpis_act["ingresos"] else 0.0
    st.write(f"Dato: el Top 3 concentra ~**{pct_top3:.1f}%** de los ingresos.")
    st.write("Acción: packs/combos para elevar servicios medios.")

with a3:
    st.info("3) Añadir 1 dimensión para decisiones reales.")
    st.write("Si añades **empleado** o **método de pago**, se puede desglosar dónde se gana/pierde.")
    st.write("Eso permite optimizar turnos, precios y promociones.")

st.divider()

# ----------------------------
# Gráficos (para soporte visual)
# ----------------------------
st.subheader("Visualización (soporte)")

serie = build_serie_tiempo(df_f, agrupacion)
st.plotly_chart(fig_ingresos_tiempo(serie, agrupacion), use_container_width=True)

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

st.divider()

# ----------------------------
# Datos en crudo (oculto por defecto)
# ----------------------------
with st.expander("Ver detalle (opcional)"):
    st.dataframe(df_f.sort_values("fecha", ascending=False), use_container_width=True)