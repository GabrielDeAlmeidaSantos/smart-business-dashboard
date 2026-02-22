# app.py
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from analytics.schema import ensure_schema, safe_date_str
from analytics.kpis import (
    eur,
    eur_day_from_annual,
    pct_str,
    compute_kpis,
    ingresos_por_dia_semana,
    build_serie_tiempo,
)
from analytics.profile import detect_business_profile
from analytics.insights import generate_insights
from analytics.history_store import HistoryStore
from analytics.ranking import rank_insights, select_plan


# ----------------------------
# Data loading
# ----------------------------
@st.cache_data(show_spinner=False)
def load_data(path: str) -> pd.DataFrame:
    """Carga datos ya procesados (parquet)."""
    return pd.read_parquet(path)


# ----------------------------
# Auth (simple + suficiente)
# ----------------------------
def is_authorized(client_id: str, code: str) -> bool:
    """Authorize write-access using secrets per client_id.

    - Local dev: .streamlit/secrets.toml
    - Deploy: Secrets UI
    """
    try:
        expected = st.secrets["clients"][client_id]
        return str(code).strip() == str(expected).strip()
    except Exception:
        return False


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
    fig.update_traces(
        hovertemplate="<b>%{x}</b><br>Ingresos: %{customdata[0]}<extra></extra>"
    )
    fig.update_yaxes(title_text="Ingresos (€)")
    fig.update_xaxes(title_text="Día")
    return fig


# ----------------------------
# Configuración app
# ----------------------------
st.set_page_config(page_title="Resumen del Negocio", layout="wide")
st.title("Resumen del Negocio (ventas)")

RUTA_DATOS = Path("data/processed/ventas_limpias.parquet")


# ----------------------------
# Sidebar (cliente + auth + admin)
# ----------------------------
with st.sidebar:
    st.header("Controles")

    modo_admin = st.toggle("Modo admin", value=False)
    st.caption("Filtra fechas y mira el resumen en segundos.")
    st.divider()

    client_id = st.text_input(
        "Cliente (ID)",
        value="default",
        help="Identificador del cliente (carpeta en data/clients/).",
    )
    st.caption(f"Historial: data/clients/{client_id}/history.json")

    access_code = st.text_input(
        "Código de acceso",
        type="password",
        help="Habilita guardado de plan y seguimiento.",
    )
    authorized = is_authorized(client_id, access_code)

    autoguardar = st.toggle(
        "Autoguardar plan del periodo",
        value=True,
        help="Guarda el plan automáticamente si aún no existe para este periodo. No machaca seguimiento.",
    )

    if not authorized:
        st.warning("Acceso restringido: sin código no se guardan cambios (plan/seguimiento).")

    st.divider()
    if modo_admin:
        st.subheader("Flujo (admin)")
        st.caption("1) Sube/actualiza el Excel en `data/input/`")
        st.caption("2) Ejecuta el pipeline para regenerar el dashboard:")
        st.code("python src/pipeline.py", language="bash")
        st.caption("Admin puede forzar regeneración del plan del periodo (ojo: reinicia seguimiento).")


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
    )
with c_f2:
    agrupacion = st.selectbox("Agrupar ingresos por", ["Mes", "Semana", "Día"], index=0)
with c_f3:
    top_n = st.selectbox("Top servicios", [5, 10, 15, 20], index=0)

if not isinstance(rango, (list, tuple)) or len(rango) != 2:
    st.warning("Selecciona un rango de fechas válido (inicio y fin).")
    st.stop()

inicio = pd.to_datetime(rango[0])
fin = pd.to_datetime(rango[1]) + pd.Timedelta(days=1)

df_f = df[(df["fecha"] >= inicio) & (df["fecha"] < fin)].copy()
if df_f.empty:
    st.warning("No hay registros en ese rango de fechas.")
    st.stop()

# Period key: evita colisiones (mismo mes con rangos distintos)
end_inclusive = pd.to_datetime(rango[1])
period_key = f"{inicio.date().isoformat()}_{end_inclusive.date().isoformat()}"

# ----------------------------
# Comparativa (misma duración, periodo anterior)
# ----------------------------
duracion = fin - inicio
inicio_prev = inicio - duracion
fin_prev = inicio
df_prev = df[(df["fecha"] >= inicio_prev) & (df["fecha"] < fin_prev)].copy()

kpis_act = compute_kpis(df_f)
kpis_prev = compute_kpis(df_prev)

delta_ing = kpis_act.ingresos - kpis_prev.ingresos
delta_ven = kpis_act.ventas - kpis_prev.ventas
delta_tic = kpis_act.ticket - kpis_prev.ticket
delta_uni = kpis_act.unidades - kpis_prev.unidades

st.caption("📌 Flechas verdes y rojas = comparación con el **periodo anterior** de la misma duración.")

# ----------------------------
# Perfil detectado
# ----------------------------
profile = detect_business_profile(df_f)

st.markdown("## Perfil detectado")
p1, p2, p3 = st.columns([2, 1, 1])
with p1:
    st.write(f"**Tipo:** {profile.business_type} | **Subtipo:** {profile.subtype}")
    st.caption("Evidencia: " + " | ".join(profile.evidence))
with p2:
    st.metric("Confianza", f"{int(profile.confidence * 100)}%")
with p3:
    st.caption("Si no es correcto, ajusta datos o reglas del perfilado.")

st.divider()

# ----------------------------
# Reutilizables (día semana)
# ----------------------------
ingresos_dia, info_dias = ingresos_por_dia_semana(df_f)
peor_dia = info_dias["peor"]
mejor_dia = info_dias["mejor"]
gap = info_dias["gap"]
ventas_peor = info_dias["ventas_peor"]

# ----------------------------
# HERO + simulador (impacto coherente con rango)
# ----------------------------
st.markdown("## Oportunidad detectada")
st.caption("Estimación conservadora: no asume subir precios; asume vender un extra/pack para elevar ticket.")

cS1, cS2 = st.columns([2, 1])
with cS1:
    subida_eur = st.slider(
        "Mejora de ticket medio en el día flojo (€)",
        min_value=0,
        max_value=10,
        value=3,
        step=1,
        help="Ejemplo: pack +3€, extra recomendado, cross-sell simple.",
    )

with cS2:
    horizonte = st.selectbox(
        "Horizonte (basado en el rango)",
        ["1× rango", "3× rango", "12× rango"],
        index=1,
        help="Evita inflar: el impacto base se calcula en el rango seleccionado y se escala por 1/3/12.",
    )

h_mult = {"1× rango": 1, "3× rango": 3, "12× rango": 12}[horizonte]

# Impacto base: lo que pasaría DENTRO del rango si subes el ticket del día flojo
impacto_rango = float(ventas_peor * subida_eur)
impacto_horizonte = float(impacto_rango * h_mult)

# “Equivalente mensual/anual” solo tiene sentido si lo basas en el rango real.
# Si quieres, mostramos equivalente €/día basado en duración del rango.
days_in_range = max(int((fin - inicio).days), 1)
impacto_por_dia = impacto_rango / days_in_range
impacto_anualizado = impacto_por_dia * 365

hero1, hero2, hero3 = st.columns([2, 1, 1])

with hero1:
    st.metric(
        "Impacto estimado",
        eur(impacto_horizonte),
        delta="estimación conservadora (extra/pack/upsell)",
    )
    st.caption(
        f"Base: rango **{inicio.date().isoformat()} → {(fin - pd.Timedelta(days=1)).date().isoformat()}** "
        f"({days_in_range} días) | Día flojo: **{peor_dia['dia_nombre']}** | "
        f"Tickets del día flojo en el rango: **{ventas_peor}**"
    )

with hero2:
    st.metric("Impacto en el rango", eur(impacto_rango))
    st.caption(f"Horizonte seleccionado: **{horizonte}**")

with hero3:
    st.metric("Equivalente €/día (rango)", eur(impacto_por_dia))
    st.metric("Anualizado (solo referencia)", eur(impacto_anualizado))

st.info(
    f"Si el **{peor_dia['dia_nombre']}** sube el ticket medio **+{subida_eur}€**, "
    f"el impacto estimado sería **{eur(impacto_horizonte)}** (horizonte **{horizonte}**)."
)

st.divider()

# ----------------------------
# Top servicios para textos
# ----------------------------
fig_top, top_df = fig_top_servicios(df_f, int(top_n))
top_item = top_df.iloc[0]["producto"] if not top_df.empty else "tu servicio principal"

# ----------------------------
# Generar insights + ranking + plan
# ----------------------------
candidates = generate_insights(
    df_range=df_f,
    kpis=kpis_act,
    profile_type=profile.business_type,
    profile_subtype=profile.subtype,
    peor_dia_nombre=peor_dia["dia_nombre"],
    mejor_dia_nombre=mejor_dia["dia_nombre"],
    gap_dias_eur=float(gap),
    ventas_peor_dia=int(ventas_peor),
    top_item=top_item,
    slider_subida_ticket_eur=float(subida_eur),
    horizon_multiplier=int(h_mult),
    horizon_label=str(horizonte),
)

history_path = Path(f"data/clients/{client_id}/history.json")
history_path.parent.mkdir(parents=True, exist_ok=True)
history = HistoryStore(client_id=client_id, path=history_path)

ranked = rank_insights(candidates, history, period_key=period_key)
plan = select_plan(ranked, k=3)

# Autoguardar plan si no existe (solo si autorizado)
if autoguardar and authorized and (history.get_period(period_key) is None):
    history.upsert_period_plan(
        period_key=period_key,
        insight_ids=[x.insight.insight_id for x in plan],
        meta={
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "business_type": profile.business_type,
            "subtype": profile.subtype,
            "horizon_label": str(horizonte),
            "horizon_multiplier": int(h_mult),
        },
    )

# Admin: forzar regeneración (resetea seguimiento)
if modo_admin and authorized:
    if st.button(f"Regenerar plan (admin) para {period_key}", use_container_width=True):
        history.force_replace_period_plan(
            period_key=period_key,
            insight_ids=[x.insight.insight_id for x in plan],
            meta={"forced": True, "generated_at": datetime.now().isoformat(timespec="seconds")},
        )
        st.success("Plan regenerado (seguimiento reiniciado).")

# ----------------------------
# Radar (Top3 + expander)
# ----------------------------
st.markdown("## Radar de oportunidades (priorizado)")

def render_radar(items):
    for s in items:
        ins = s.insight
        low, high = ins.estimated_impact_eur

        st.markdown(f"**💰 {ins.title}**")
        c1, c2, c3, c4 = st.columns([2, 1, 1, 1])

        with c1:
            # antes era trigger; ahora usamos evidencia (más honesto y numérico)
            st.caption(ins.evidence)

        with c2:
            st.write(f"**Impacto ({horizonte}):** {eur(low)} – {eur(high)}")

        with c3:
            st.write(f"**Esfuerzo:** {ins.effort}")

        with c4:
            st.write(f"**Tiempo:** {ins.time_to_apply_min} min")

        st.write(f"**Acción:** {ins.action_hint}")
        if modo_admin:
            st.caption(f"Motivo ranking: {s.reason} | score={s.score:.2f}")
        st.divider()

render_radar(ranked[:3])

with st.expander("Ver radar completo"):
    render_radar(ranked[3:10])

# ----------------------------
# Resumen ejecutivo
# ----------------------------
st.markdown("## Resumen ejecutivo (en 10 segundos)")
cE1, cE2, cE3 = st.columns(3)
with cE1:
    st.metric("Día flojo", peor_dia["dia_nombre"])
    st.caption(f"Ingresos: {eur(peor_dia['ingresos'])}")
with cE2:
    st.metric("Mejor día", mejor_dia["dia_nombre"])
    st.caption(f"Ingresos: {eur(mejor_dia['ingresos'])}")
with cE3:
    st.metric("Gap real entre días", eur(gap))
    st.caption(f"{mejor_dia['dia_nombre']} vs {peor_dia['dia_nombre']}")

st.divider()

# ----------------------------
# Plan de mejora (dinámico)
# ----------------------------
st.markdown("## Plan de mejora (3 acciones) — dinámico")

cols = st.columns(3)
for col, s in zip(cols, plan):
    ins = s.insight
    low, high = ins.estimated_impact_eur
    with col:
        st.success(ins.title)
        st.write(f"**Impacto ({horizonte}):** {eur(low)} – {eur(high)}")
        st.write(f"**Esfuerzo:** {ins.effort} | **Tiempo:** {ins.time_to_apply_min} min")
        st.write(f"**Acción:** {ins.action_hint}")
        st.caption(f"Evidencia: {ins.evidence}")
        if modo_admin:
            st.caption(f"Motivo ranking: {s.reason}")

st.divider()

# ----------------------------
# Seguimiento (feedback propietario)
# ----------------------------
st.markdown("## Seguimiento (para que no se repita y el sistema aprenda)")
st.caption("Marca si se aplicó y si funcionó. Esto afecta el ranking futuro.")

period_data = history.get_period(period_key)

if period_data is None:
    st.warning(f"No hay plan guardado para {period_key}. Si estás autorizado, activa autoguardado o usa admin.")
else:
    items = period_data.get("items") or []
    if len(items) == 0:
        st.warning("Plan vacío en historial. Regenera plan en admin o revisa autoguardado.")
    else:
        for it in items:
            iid = it.get("insight_id")
            status = it.get("status", "planned")
            outcome = it.get("outcome", "unknown")
            note = it.get("note", "")

            st.write(f"**{iid}**")
            c1, c2, c3 = st.columns([1, 1, 2])
            with c1:
                new_status = st.selectbox(
                    "Aplicación",
                    ["planned", "done", "skipped"],
                    index=["planned", "done", "skipped"].index(status) if status in ["planned", "done", "skipped"] else 0,
                    key=f"status_{period_key}_{iid}",
                    help="done=lo hice; skipped=no lo hice; planned=pendiente",
                )
            with c2:
                if new_status == "done":
                    new_outcome = st.selectbox(
                        "Resultado",
                        ["unknown", "improved", "not_improved"],
                        index=["unknown", "improved", "not_improved"].index(outcome)
                        if outcome in ["unknown", "improved", "not_improved"]
                        else 0,
                        key=f"outcome_{period_key}_{iid}",
                        help="improved=mejoró KPI; not_improved=no mejoró",
                    )
                else:
                    st.selectbox(
                        "Resultado",
                        ["unknown"],
                        index=0,
                        key=f"outcome_{period_key}_{iid}_locked",
                        help="Solo puedes marcar resultado cuando el estado es done.",
                    )
                    new_outcome = "unknown"
            with c3:
                new_note = st.text_input(
                    "Nota (opcional)",
                    value=str(note),
                    key=f"note_{period_key}_{iid}",
                    placeholder="Ej: ofrecí extra A, lo aceptó 3 de 10 clientes...",
                )

            if st.button("Guardar seguimiento", key=f"save_{period_key}_{iid}"):
                if not authorized:
                    st.error("No autorizado: introduce el código para guardar cambios.")
                else:
                    history.update_item(
                        period_key=period_key,
                        insight_id=iid,
                        status=new_status,
                        outcome=new_outcome,
                        note=new_note,
                    )
                    st.success("Guardado.")

            st.divider()

# ----------------------------
# KPIs principales
# ----------------------------
st.markdown("## Contexto del rango (KPIs)")
k1, k2, k3, k4 = st.columns(4)
k1.metric(
    "Ingresos",
    eur(kpis_act.ingresos),
    delta=f"{eur(delta_ing)} ({pct_str(delta_ing, kpis_prev.ingresos)})" if kpis_prev.ingresos else None,
)
k2.metric(
    "Ventas (tickets)",
    f"{kpis_act.ventas}",
    delta=f"{delta_ven} ({pct_str(delta_ven, kpis_prev.ventas)})" if kpis_prev.ventas else None,
)
k3.metric(
    "Ticket medio",
    eur(kpis_act.ticket),
    delta=f"{eur(delta_tic)} ({pct_str(delta_tic, kpis_prev.ticket)})" if kpis_prev.ticket else None,
)
k4.metric(
    "Unidades",
    f"{int(kpis_act.unidades)}",
    delta=f"{int(delta_uni)} ({pct_str(delta_uni, kpis_prev.unidades)})" if kpis_prev.unidades else None,
)

st.divider()

# ----------------------------
# Impacto final
# ----------------------------
st.markdown("## Impacto económico (mensaje final)")
st.success(
    f"Con una mejora simple (**+{subida_eur}€** por ticket) en el **{peor_dia['dia_nombre']}**, "
    f"el impacto estimado es **{eur(impacto_horizonte)}** (horizonte **{horizonte}**, basado en el rango seleccionado)."
)
st.caption(
    f"Referencia anualizada (solo para tener escala): **{eur(impacto_anualizado)}**/año "
    f"≈ **{eur(impacto_por_dia)}**/día."
)
st.write(
    "Esto se recalcula automáticamente al actualizar el Excel y te deja claro qué palanca atacar y cuánto dinero mueve."
)

st.divider()

# ----------------------------
# Visualización (soporte)
# ----------------------------
st.markdown("## Visualización (soporte)")
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

with st.expander("Ver detalle (opcional)"):
    st.dataframe(df_f.sort_values("fecha", ascending=False), use_container_width=True)
