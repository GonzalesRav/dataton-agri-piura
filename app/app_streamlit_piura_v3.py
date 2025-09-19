
# app_streamlit_piura_v3.py
# ------------------------------------------------------------
# Dashboard Piura (v3): indicadores accionables para elección de cultivos
# Requisitos:
#   pip install streamlit pandas numpy plotly geopandas shapely
# Ejecución:
#   streamlit run app_streamlit_piura_v3.py
# Entradas (subir desde la barra lateral):
#   - dataset_piura_anual.csv   (salida del preprocesamiento)
#   - dataset_piura_clean.csv   (mensual, para calendario)
#   - distritos_piura.geojson   (opcional, para mapa por distrito - se colorea por flexibilidad)
# ------------------------------------------------------------

import io
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# ---------------------------
# Helpers
# ---------------------------

def _read_csv(uploaded):
    if uploaded is None:
        return None
    try:
        return pd.read_csv(uploaded)
    except UnicodeDecodeError:
        uploaded.seek(0)
        return pd.read_csv(uploaded, encoding="latin-1")

def _norm_cols(df):
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    return df

def shannon_index(df, area_col="area_sembrada_ha"):
    total = df[area_col].sum()
    if total <= 0 or df.empty:
        return np.nan
    p = df[area_col] / total
    p = p[p > 0]
    return float(-np.sum(p * np.log(p)))

def herfindahl_index(df, area_col="area_sembrada_ha"):
    total = df[area_col].sum()
    if total <= 0 or df.empty:
        return np.nan
    p = df[area_col] / total
    return float(np.sum(p**2))

def compute_flex(df_anual):
    if df_anual is None or df_anual.empty:
        return pd.DataFrame()
    g = df_anual.groupby(["anio", "tipo_cultivo"], as_index=False)["area_sembrada_ha"].sum()
    piv = g.pivot_table(index="anio", columns="tipo_cultivo", values="area_sembrada_ha", aggfunc="sum").fillna(0.0)
    piv["flexibilidad"] = piv.get("Transitorio", 0.0) / (piv.get("Transitorio", 0.0) + piv.get("Permanente", 0.0)).replace({0: np.nan})
    piv = piv.reset_index()
    return piv

def risk_semaforo(df_anual, area_thr=15.0, price_thr=-10.0):
    if df_anual is None or df_anual.empty:
        return pd.DataFrame()
    df = df_anual.copy().sort_values(["cultivo_std", "anio"])
    df["area_prom_3_prev"] = df.groupby("cultivo_std")["area_sembrada_ha"].apply(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
    df["delta_area_pct"] = (df["area_sembrada_ha"] - df["area_prom_3_prev"]) / df["area_prom_3_prev"] * 100.0
    df.loc[~np.isfinite(df["delta_area_pct"]), "delta_area_pct"] = np.nan
    if "precio_prom_s_kg" in df.columns:
        df["precio_prom_3_prev"] = df.groupby("cultivo_std")["precio_prom_s_kg"].apply(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
        df["delta_precio_pct"] = (df["precio_prom_s_kg"] - df["precio_prom_3_prev"]) / df["precio_prom_3_prev"] * 100.0
        df.loc[~np.isfinite(df["delta_precio_pct"]), "delta_precio_pct"] = np.nan
    else:
        df["delta_precio_pct"] = np.nan
    df["score_riesgo"] = 0
    df.loc[df["delta_area_pct"] > area_thr, "score_riesgo"] += 1
    df.loc[df["delta_precio_pct"] < price_thr, "score_riesgo"] += 1
    def lab(s):
        if s >= 2: return "Alto"
        if s == 1: return "Medio"
        return "Bajo"
    df["riesgo"] = df["score_riesgo"].apply(lab)
    return df

def calendar_transitorios(df_clean):
    if df_clean is None or df_clean.empty:
        return None, None
    d = df_clean.copy()
    d = d[d["tipo_cultivo"].eq("Transitorio")]
    if d.empty:
        return None, None
    sow = d.groupby(["cultivo_std", "mes"], as_index=False)["siembra"].sum()
    har = d.groupby(["cultivo_std", "mes"], as_index=False)["cosecha"].sum()
    return sow, har

# ---------------------------
# App
# ---------------------------

st.set_page_config(page_title="Piura • Elección de Cultivos (v3)", layout="wide")
st.title("🌾 Piura • Elección de Cultivos (v3)")
st.caption("Indicadores: evolución, producción–superficie–precio, diversificación (transitorios), flexibilidad agrícola, riesgo de sobreoferta, calendario (transitorios) y recomendaciones.")

with st.sidebar:
    st.header("Datos de entrada")
    anual_file = st.file_uploader("dataset_piura_anual.csv", type=["csv"])
    clean_file = st.file_uploader("dataset_piura_clean.csv (opcional para calendario)", type=["csv"])
    gj_file = st.file_uploader("distritos_piura.geojson (opcional)", type=["geojson","json"])

    if anual_file:
        df_anual = _norm_cols(_read_csv(anual_file))
    else:
        df_anual = None

    if clean_file:
        df_clean = _norm_cols(_read_csv(clean_file))
    else:
        df_clean = None

    st.markdown("---")
    st.header("Filtros")
    if df_anual is not None and not df_anual.empty:
        years = sorted(df_anual["anio"].dropna().unique().tolist()) if "anio" in df_anual.columns else []
        y_min, y_max = (years[0], years[-1]) if years else (None, None)
        sel_years = st.select_slider("Rango de años", options=years if years else [0], value=(y_min, y_max) if years else (0, 0))
        cultivos = sorted(df_anual["cultivo_std"].dropna().unique().tolist()) if "cultivo_std" in df_anual.columns else []
        sel_cult_compare = st.multiselect("Comparar 2–3 cultivos", options=cultivos, default=cultivos[:3] if len(cultivos) >= 3 else cultivos)
        if all(v is not None for v in sel_years):
            df_anual_f = df_anual[df_anual["anio"].between(sel_years[0], sel_years[1])].copy()
        else:
            df_anual_f = df_anual.copy()
    else:
        df_anual_f = None
        sel_cult_compare = []

if df_anual_f is None or df_anual_f.empty:
    st.info("Sube el archivo **dataset_piura_anual.csv** para comenzar. 👈")
    st.stop()

# --- 1) Evolución de áreas sembradas ---
st.subheader("1) Evolución de áreas sembradas")
d1 = df_anual_f[df_anual_f["cultivo_std"].isin(sel_cult_compare)] if sel_cult_compare else df_anual_f.head(0)
if d1.empty:
    st.warning("Selecciona 2–3 cultivos para comparar.")
else:
    g1 = d1.groupby(["anio", "cultivo_std"], as_index=False)["area_sembrada_ha"].sum()
    fig1 = px.line(g1, x="anio", y="area_sembrada_ha", color="cultivo_std", markers=True)
    st.plotly_chart(fig1, use_container_width=True)
    with st.expander("Nota metodológica"):
        st.write("Se muestra la evolución anual del área sembrada (ha) por cultivo seleccionado.")

# --- 2) Producción vs superficie vs precio ---
st.subheader("2) Producción vs superficie vs precio")
d2 = df_anual_f.copy()
if "precio_prom_s_kg" not in d2.columns:
    d2["precio_prom_s_kg"] = np.nan
d2["valor_produccion_aprox_s"] = d2["produccion_t"] * 1000.0 * d2["precio_prom_s_kg"]
fig2 = px.scatter(d2, x="area_sembrada_ha", y="produccion_t", size="valor_produccion_aprox_s", color="precio_prom_s_kg", hover_data=["anio","cultivo_std"])
st.plotly_chart(fig2, use_container_width=True)

# --- 3) Índice de diversificación (transitorios) ---
st.subheader("3) Índice de diversificación (transitorios)")
d_trans = df_anual_f[df_anual_f["tipo_cultivo"].eq("Transitorio")].copy()
div_series = []
for y, dfy in d_trans.groupby("anio"):
    div = shannon_index(dfy, "area_sembrada_ha")
    hhi = herfindahl_index(dfy, "area_sembrada_ha")
    div_series.append({"anio": y, "shannon": div, "herfindahl": hhi})
div_df = pd.DataFrame(div_series).sort_values("anio")
fig3a = px.line(div_df, x="anio", y="shannon", markers=True)
fig3b = px.line(div_df, x="anio", y="herfindahl", markers=True)
cols = st.columns(2)
cols[0].plotly_chart(fig3a, use_container_width=True)
cols[1].plotly_chart(fig3b, use_container_width=True)

# --- 4) Índice de flexibilidad agrícola ---
st.subheader("4) Índice de flexibilidad agrícola")
flex_df = compute_flex(df_anual_f)
fig4 = px.line(flex_df, x="anio", y="flexibilidad", markers=True)
st.plotly_chart(fig4, use_container_width=True)

# --- 5) Semáforo de riesgo de sobreoferta ---
st.subheader("5) Semáforo de riesgo de sobreoferta")
risk_df = risk_semaforo(df_anual_f, area_thr=15.0, price_thr=-10.0)
if not risk_df.empty:
    latest = risk_df["anio"].max()
    last_view = risk_df[risk_df["anio"] == latest].sort_values(["score_riesgo","cultivo_std"], ascending=[False, True])
    st.dataframe(last_view[["anio","cultivo_std","delta_area_pct","delta_precio_pct","score_riesgo","riesgo"]])
else:
    st.info("No se pudo calcular el semáforo.")

# --- 6) Calendario dinámico ---
st.subheader("6) Calendario dinámico (siembra/cosecha)")
if df_clean is not None and not df_clean.empty:
    sow, har = calendar_transitorios(df_clean)
    if sow is not None:
        sow["sow_int"] = sow.groupby("cultivo_std")["siembra"].transform(lambda s: s / s.max() if s.max() not in [0, np.nan] else 0)
        har["har_int"] = har.groupby("cultivo_std")["cosecha"].transform(lambda s: s / s.max() if s.max() not in [0, np.nan] else 0)
        sow_piv = sow.pivot_table(index="cultivo_std", columns="mes", values="sow_int", aggfunc="mean").fillna(0)
        har_piv = har.pivot_table(index="cultivo_std", columns="mes", values="har_int", aggfunc="mean").fillna(0)
        fig6a = px.imshow(sow_piv, aspect="auto")
        fig6b = px.imshow(har_piv, aspect="auto")
        cols = st.columns(2)
        cols[0].plotly_chart(fig6a, use_container_width=True)
        cols[1].plotly_chart(fig6b, use_container_width=True)
    else:
        st.info("No se encontraron transitorios.")
else:
    st.info("Sube dataset_piura_clean.csv para ver calendario.")

# --- 7) Panel de recomendaciones ---
st.subheader("7) Recomendaciones prácticas")
notes = []
if not risk_df.empty:
    latest = risk_df["anio"].max()
    last = risk_df[risk_df["anio"] == latest]
    alto = last[last["riesgo"] == "Alto"]["cultivo_std"].tolist()
    medio = last[last["riesgo"] == "Medio"]["cultivo_std"].tolist()
    if alto:
        notes.append("Riesgo ALTO de sobreoferta en: " + ", ".join(alto))
    if medio:
        notes.append("Riesgo MEDIO en: " + ", ".join(medio))
if flex_df is not None and not flex_df.empty:
    fx_last = flex_df[flex_df["anio"] == flex_df["anio"].max()]["flexibilidad"].iloc[0]
    if pd.notnull(fx_last):
        notes.append(f"Flexibilidad agrícola último año: {fx_last:.2f}")
if div_df is not None and not div_df.empty:
    sh_last = div_df[div_df["anio"] == div_df["anio"].max()]["shannon"].iloc[0]
    if pd.notnull(sh_last):
        notes.append(f"Diversificación (Shannon) transitorios último año: {sh_last:.2f}")
if notes:
    for n in notes:
        st.write("- " + n)
else:
    st.info("Carga datos para recomendaciones.")
