from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


# ----------------------------
# Configuración de la página
# ----------------------------
st.set_page_config(page_title="Dashboard de Negocio", layout="wide")
st.title("Dashboard de Negocio (Genérico)")

RUTA_KPIS = Path("data/processed/kpis.parquet")
RUTA_DATOS = Path("data/processed/ventas_limpias.parquet")

with st.sidebar:
    st.header("Flujo")
    st.caption("1) Sube/actualiza el Excel en `data/input/`")
    st.caption("2) Ejecuta el pipeline para regenerar el dashboard:")
    st.code("python src/pipeline.py", language="bash")


# ----------------------------
# Carga de datos procesados
# ----------------------------
if not (RUTA_KPIS.exists() and RUTA_DATOS.exists()):
    st.warning("No hay datos procesados aún. Ejecuta primero: `python src/pipeline.py`")
    st.stop()

kpis = pd.read_parquet(RUTA_KPIS).iloc[0]
df = pd.read_parquet(RUTA_DATOS)

# Normalizar tipos
df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
df = df.dropna(subset=["fecha"])
df = df.sort_values("fecha")

# ----------------------------
# Controles (filtros)
# ----------------------------
min_f, max_f = df["fecha"].min().date(), df["fecha"].max().date()

c_f1, c_f2, c_f3 = st.columns([2, 2, 1])
with c_f1:
    rango = st.date_input("Rango de fechas", value=(min_f, max_f), min_value=min_f, max_value=max_f)
with c_f2:
    agrupacion = st.selectbox("Agrupar ingresos por", ["Día", "Semana", "Mes"], index=2)
with c_f3:
    top_n = st.slider("Top productos/servicios", 3, 20, 8)

# Filtrar
inicio = pd.to_datetime(rango[0])
fin = pd.to_datetime(rango[1]) + pd.Timedelta(days=1)
df_f = df[(df["fecha"] >= inicio) & (df["fecha"] < fin)].copy()

if df_f.empty:
    st.warning("No hay registros en ese rango de fechas.")
    st.stop()

# ----------------------------
# KPIs (del rango filtrado)
# ----------------------------
ingresos = float(df_f["revenue"].sum())
transacciones = int(len(df_f))
ticket_medio = float(df_f["revenue"].mean()) if transacciones else 0.0
unidades = float(df_f["cantidad"].sum())

k1, k2, k3, k4 = st.columns(4)
k1.metric("Ingresos", f"{ingresos:.2f} €")
k2.metric("Transacciones", transacciones)
k3.metric("Ticket medio", f"{ticket_medio:.2f} €")
k4.metric("Unidades", f"{unidades:.0f}")

st.divider()

# ----------------------------
# Ingresos en el tiempo (genérico y útil)
# ----------------------------
if agrupacion == "Día":
    df_f["periodo"] = df_f["fecha"].dt.date
elif agrupacion == "Semana":
    df_f["periodo"] = df_f["fecha"].dt.to_period("W").astype(str)
else:
    df_f["periodo"] = df_f["fecha"].dt.to_period("M").astype(str)

serie = df_f.groupby("periodo", as_index=False)["revenue"].sum()

fig_linea = px.line(serie, x="periodo", y="revenue", title="Ingresos en el tiempo")
st.plotly_chart(fig_linea, use_container_width=True)

# ----------------------------
# Top productos/servicios (universal)
# ----------------------------
top = (
    df_f.groupby("producto", as_index=False)["revenue"].sum()
    .sort_values("revenue", ascending=False)
    .head(top_n)
)

fig_top = px.bar(
    top,
    x="revenue",
    y="producto",
    orientation="h",
    title=f"Top {top_n} productos/servicios por ingresos",
)
st.plotly_chart(fig_top, use_container_width=True)

st.divider()

# ----------------------------
# Insights (vende mucho)
# ----------------------------
st.subheader("Insights rápidos")

top_item = top.iloc[0]
mejor_periodo = serie.sort_values("revenue", ascending=False).iloc[0]

# Concentración (qué % del total hace el top 3)
top3 = (
    df_f.groupby("producto", as_index=False)["revenue"].sum()
    .sort_values("revenue", ascending=False)
    .head(3)["revenue"]
    .sum()
)
pct_top3 = (top3 / ingresos * 100) if ingresos else 0.0

col_i1, col_i2 = st.columns(2)
with col_i1:
    st.write(f"• Tu principal generador de ingresos es **{top_item['producto']}** con **{top_item['revenue']:.2f} €**.")
    st.write(f"• El mejor periodo fue **{mejor_periodo['periodo']}** con **{mejor_periodo['revenue']:.2f} €**.")
with col_i2:
    st.write(f"• El **Top 3** concentra aproximadamente **{pct_top3:.1f}%** de los ingresos del rango seleccionado.")
    st.write("• Si quieres más utilidad: añade columna de **método de pago** o **empleado** y lo desglosamos.")

# ----------------------------
# Datos en crudo (oculto por defecto)
# ----------------------------
with st.expander("Ver datos (normalizados y limpios)"):
    st.dataframe(df_f.sort_values("fecha", ascending=False), use_container_width=True)