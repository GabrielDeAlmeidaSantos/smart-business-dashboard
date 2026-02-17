from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

# ------------------------------------
# Utilidades
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
    """KPIs básicos."""
    if df_ is None or df_.empty:
        return {"ingresos": 0.0, "ventas": 0, "ticket": 0.0, "unidades": 0}

    ingresos = float(df_["revenue"].sum())
    ventas = int(len(df_))  # cada fila = ticket/linea (según tu pipeline)
    ticket = float(df_["revenue"].mean()) if ventas else 0.0
    unidades = float(df_["cantidad"].sum())
    return {"ingresos": ingresos, "ventas": ventas, "ticket": ticket, "unidades": unidades}


@st.cache_data(show_spinner=False)
def load_data(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)


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
    """Ingresos por día de semana + info (mejor/peor/gap/ventas_peor)."""
    mapa_dias = {
        0: "Lunes",
        1: "Martes",
        2: "Miércoles",
        3: "Jueves",
        4: "Viernes",
        5: "Sábado",
        6: "Domingo",
    }

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
    ventas_peor = int(
        ventas_por_dia.loc[
            ventas_por_dia["dia_nombre"] == peor["dia_nombre"], "ventas"
        ].iloc[0]
    )

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
    modo_admin = st.toggle("Modo admin", value=False, key="modo_admin_v4")
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
        key="rango_v4",
    )
with c_f2:
    agrupacion = st.selectbox(
        "Agrupar ingresos por",
        ["Mes", "Semana", "Día"],
        index=0,
        key="agrupacion_v4",
    )
with c_f3:
    top_n = st.selectbox(
        "Top servicios",
        [5, 10, 15, 20],
        index=0,
        key="top_n_v4",
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
# Comparativa (misma duración, periodo anterior)
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

# Texto de referencia de las flechas (MUY importante para no confundir)
st.caption(
    f"📌 Flechas verdes y rojas = comparación con el **periodo anterior** de la misma duración "
)

# ----------------------------
# KPIs principales
# ----------------------------
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
# Cálculos reutilizables (día semana)
# ----------------------------
ingresos_dia, info_dias = compute_ingresos_por_dia(df_f)
peor_dia = info_dias["peor"]
mejor_dia = info_dias["mejor"]
gap = info_dias["gap"]
ventas_peor = info_dias["ventas_peor"]

# ----------------------------
# Resumen ejecutivo (venta)
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
# Simulador (escenario) + Impactos
# ----------------------------
st.subheader("Simulador rápido (escenario)")
st.caption("Estimación basada en tu ticket medio y el volumen de ventas del día más flojo.")

cS1, cS2 = st.columns([2, 1])
with cS1:
    subida_eur = st.slider(
        "Subida de ticket medio en el día flojo (€)",
        min_value=0,
        max_value=10,
        value=3,
        step=1,
        key="subida_ticket_v4",
    )
with cS2:
    horizonte = st.selectbox("Horizonte (mensaje)", ["Mes", "Trimestre"], index=1, key="horizonte_v4")

# Impactos SIEMPRE visibles (mes/trimestre/año)
impacto_mes = float(ventas_peor * subida_eur * 1)
impacto_trimestre = float(ventas_peor * subida_eur * 3)
impacto_anual = float(ventas_peor * subida_eur * 12)

mult = 1 if horizonte == "Mes" else 3
impacto_msg = float(ventas_peor * subida_eur * mult)

st.info(
    f"Si el **{peor_dia['dia_nombre']}** sube el ticket medio **+{subida_eur}€**, "
    f"el impacto estimado sería **{eur(impacto_msg)}** en **{horizonte.lower()}**."
)

cImp1, cImp2, cImp3 = st.columns(3)
with cImp1:
    st.metric("Impacto mensual", eur(impacto_mes))
with cImp2:
    st.metric("Impacto trimestral", eur(impacto_trimestre))
with cImp3:
    st.metric("Impacto anual", eur(impacto_anual))

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
    st.write(
        f"Impacto (escenario): +{subida_eur}€ por ticket ⇒ "
        f"**{eur(impacto_trimestre)}** por trimestre (aprox.)."
    )

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
# Recomendaciones accionables (sin subir precios)
# ----------------------------
st.subheader("Cómo mejorar sin subir precios (acciones sugeridas)")

# Heurísticas simples (no inventan datos, solo proponen tácticas)
top3_concentracion_alta = pct_top3 >= 55.0  # ajustable
servicio_top = top_item
dia_flojo = peor_dia["dia_nombre"]

colR1, colR2 = st.columns(2)

with colR1:
    st.write(f"### 1) Subir ticket medio el **{dia_flojo}** (sin tocar precios)")
    st.write("Acciones prácticas:")
    st.markdown(
        "- **Packs**: servicio + extra (ej: *servicio principal + tratamiento rápido*).\n"
        "- **Add-ons** en caja: ofrece 2 opciones (A/B) para que el cliente elija.\n"
        "- **Umbral**: “si hoy añades un extra, te incluyo X” (bonus en vez de descuento)."
    )
    st.caption("Objetivo: aumentar € por cliente sin subir el precio base. Mantén 2–3 opciones máximo.")

with colR2:
    st.write(f"### 2) Aprovechar tu servicio top (**{servicio_top}**)")
    st.write("Acciones prácticas:")
    st.markdown(
        "- **Cross-sell natural**: añade un complemento que encaje con el servicio.\n"
        "- **Bundle**: “servicio_top + extra recomendado” (mismo cliente, más valor).\n"
        "- **Guion de recomendación**: “Para que dure más, te recomiendo X (2 opciones)”."
    )
    st.caption("Objetivo: que el top no solo genere ingresos, sino que arrastre extras.")

st.write("### 3) Reducir riesgo de dependencia (Top 3)")
if top3_concentracion_alta:
    st.warning(
        f"El Top 3 concentra ~{pct_top3:.1f}% de los ingresos. Eso es alto: dependes de pocos servicios."
    )
    st.markdown(
        "- Crea **1 pack** para un servicio medio y empújalo 2 días/semana.\n"
        "- Haz un **menú de extras** fijo (2–3) y mide adopción.\n"
        "- Si tienes huecos: **lista de espera + confirmación** para reducir cancelaciones."
    )
else:
    st.success(
        f"El Top 3 concentra ~{pct_top3:.1f}% (razonable). Puedes enfocarte más en ticket medio y fidelización."
    )
    st.markdown(
        "- Mantén packs simples y mide qué extras se venden.\n"
        "- Optimiza el día flojo con una acción semanal (recordatorio + pack)."
    )

st.divider()

# ----------------------------
# Cierre comercial (para puerta fría)
# ----------------------------
st.subheader("Impacto económico")

st.success(
    f"Con una mejora simple (**+{subida_eur}€** por ticket) en el **{peor_dia['dia_nombre']}**, "
    f"este negocio podría generar aproximadamente **{eur(impacto_anual)} extra al año** "
    f"(estimación basada en el histórico del rango seleccionado)."
)

st.write(
    "Este dashboard sirve para detectar **qué día está flojo**, **qué servicios mandan** "
    "y **qué acción concreta** tiene más impacto. Se actualiza automáticamente cuando subes el Excel."
)

st.divider()

# ----------------------------
# Gráficos (soporte visual)
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
# Datos (oculto)
# ----------------------------
with st.expander("Ver detalle (opcional)"):
    st.dataframe(df_f.sort_values("fecha", ascending=False), use_container_width=True)