# app/app.py
# -*- coding: utf-8 -*-
import json
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# ---------- Rutas ----------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data_proc"
GEO_DIR  = BASE_DIR / "geo"

CSV_BASE = DATA_DIR / "campania_agricola_piura_proc.csv"
CSV_TRI  = DATA_DIR / "agregados_trimestrales.csv"
CSV_ANU  = DATA_DIR / "agregados_anuales.csv"
CSV_COV  = DATA_DIR / "cobertura_precio_distrito_anio.csv"
GJ_DIST  = GEO_DIR  / "distritos_piura.geojson"   # debe tener campo UBIGEO

# ---------- Config Streamlit ----------
st.set_page_config(
    page_title="AgriPiura • Oferta, Rendimiento y Precios",
    layout="wide",
    page_icon="🌾",
)

st.markdown("# 🌾 AgriPiura — Oferta, Rendimiento y Precios")
st.caption("Datatón 2025 • ODS 2 Hambre Cero • Reproducible con datos abiertos")

# ---------- Utilidades ----------
@st.cache_data(show_spinner=False)
def load_data():
    df = pd.read_csv(CSV_BASE)
    tri = pd.read_csv(CSV_TRI) if CSV_TRI.exists() else pd.DataFrame()
    anu = pd.read_csv(CSV_ANU) if CSV_ANU.exists() else pd.DataFrame()
    cov = pd.read_csv(CSV_COV) if CSV_COV.exists() else pd.DataFrame()
    # Fechas
    if "FECHA_YYYYMM" in df.columns:
        df["FECHA_YYYYMM"] = pd.to_datetime(df["FECHA_YYYYMM"])
        df["ANIO"] = df["FECHA_YYYYMM"].dt.year
        df["MES_NUM"] = df["FECHA_YYYYMM"].dt.month
    # Asegurar tipos
    for c in ["PRODUCCION","COSECHA","SIEMBRA","VERDE_ACTUAL","PRECIO_CHACRA","PRODUCCION_KG"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    # Cargar GeoJSON
    geojson = None
    if GJ_DIST.exists():
        with open(GJ_DIST, "r", encoding="utf-8") as f:
            geojson = json.load(f)
    return df, tri, anu, cov, geojson

def weighted_mean(values, weights):
    v = pd.to_numeric(values, errors="coerce")
    w = pd.to_numeric(weights, errors="coerce")
    m = (~v.isna()) & (~w.isna()) & (w > 0)
    if m.sum() == 0:
        return np.nan
    return np.average(v[m], weights=w[m])

def kpi_number(val, label, help_text=None, format_str=None):
    c = st.container()
    if format_str:
        c.metric(label, format_str.format(val) if val==val else "—")
    else:
        c.metric(label, f"{val:,.2f}" if pd.notna(val) else "—", help=help_text)

# ---------- Carga ----------
df, tri, anu, cov, gj = load_data()
if df.empty:
    st.error("No se encontró el CSV procesado. Corre primero `scripts/preprocesar_piura.py`.")
    st.stop()

# ---------- Sidebar (filtros globales) ----------
st.sidebar.header("Filtros")
cultivos = ["(Todos)"] + sorted([c for c in df["CULTIVO"].dropna().unique().tolist()])
cultivo_sel = st.sidebar.selectbox("Cultivo", options=cultivos)

anios = sorted(df["ANIO"].dropna().unique().tolist())
if len(anios)==0:
    st.warning("No hay años detectados en la base.")
    st.stop()
anio_min, anio_max = min(anios), max(anios)
rango_anios = st.sidebar.slider("Años", int(anio_min), int(anio_max), (int(anio_min), int(anio_max)))

nivel = st.sidebar.radio("Nivel espacial", options=["Distrito"], index=0)
lag_meses = st.sidebar.slider("Lag VERDE_ACTUAL (meses)", 1, 3, 1)

# Filtro DataFrame base
mask = (df["ANIO"]>=rango_anios[0]) & (df["ANIO"]<=rango_anios[1])
if cultivo_sel != "(Todos)":
    mask &= (df["CULTIVO"] == cultivo_sel)
df_f = df.loc[mask].copy()

# VERDE lag (proxy de oferta futura)
df_f = df_f.sort_values("FECHA_YYYYMM")
df_f["VERDE_LAG"] = df_f.groupby(["UBIGEO","CULTIVO"])["VERDE_ACTUAL"].shift(lag_meses)

# ---------- KPIs ----------
col1, col2, col3, col4 = st.columns(4)
prod_tot = df_f["PRODUCCION"].sum()
rend_prom = (df_f["PRODUCCION"].sum() / df_f["COSECHA"].sum()) if df_f["COSECHA"].sum() > 0 else np.nan
precio_pond = weighted_mean(df_f["PRECIO_CHACRA"], df_f["PRODUCCION_KG"])
# Cobertura: si hay tabla cov, usarla; si no, calculamos rápido
if not cov.empty:
    mask_cov = (cov["ANIO"]>=rango_anios[0]) & (cov["ANIO"]<=rango_anios[1])
    cov_sel = cov.loc[mask_cov].copy()
    cov_pct = cov_sel["MESES_CON_PRECIO_OK"].sum() / cov_sel["MESES_CON_PROD"].sum() if cov_sel["MESES_CON_PROD"].sum() > 0 else 1.0
    all_ok = bool((cov_sel["ALL_PRICE_OK"]).all()) if len(cov_sel) else True
else:
    tmp = df_f.copy()
    tmp["_has_prod"] = tmp["PRODUCCION"].fillna(0) > 0
    tmp["_price_ok"] = tmp["PRECIO_CHACRA"].notna() & (tmp["PRECIO_CHACRA"] > 0)
    num = (tmp["_has_prod"] & tmp["_price_ok"]).sum()
    den = (tmp["_has_prod"]).sum()
    cov_pct = num/den if den>0 else 1.0
    all_ok = (cov_pct == 1.0)

with col1: kpi_number(prod_tot, "Producción total (t)")
with col2: kpi_number(rend_prom, "Rendimiento prom. (t/ha)")
with col3: kpi_number(precio_pond, "Precio chacra pond. (S/ kg)")
with col4:
    c = st.container()
    c.metric("Cobertura de precio", f"{cov_pct*100:,.1f}%")
    c.caption("ALL_PRICE_OK: ✅" if all_ok else "ALL_PRICE_OK: ⚠️")

# ---------- Tabs ----------
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Diagnóstico", "Mercado", "Predicción", "Calidad de datos", "Fuentes"])

# ========== Tab 1: Diagnóstico ==========
with tab1:
    left, right = st.columns([1.15, 1])
    with left:
        st.subheader("Serie: Producción (t) y VERDE_LAG (ha)")
        # agregación mensual
        g1 = (df_f.groupby("FECHA_YYYYMM", as_index=False)
                  .agg(PROD=("PRODUCCION","sum"),
                       VERDE_LAG=("VERDE_LAG","sum")))
        fig1 = px.bar(g1, x="FECHA_YYYYMM", y="PROD", labels={"PROD":"Producción (t)", "FECHA_YYYYMM":"Mes"})
        fig1.add_scatter(x=g1["FECHA_YYYYMM"], y=g1["VERDE_LAG"], mode="lines", name=f"VERDE_LAG ({lag_meses}m)")
        fig1.update_layout(margin=dict(l=10,r=10,t=10,b=10))
        st.plotly_chart(fig1, use_container_width=True)

    with right:
        st.subheader("Siembra/Cosecha y Rendimiento (t/ha)")
        g2 = (df_f.groupby("FECHA_YYYYMM", as_index=False)
                  .agg(SIEM=("SIEMBRA","sum"),
                       COSE=("COSECHA","sum"),
                       PROD=("PRODUCCION","sum")))
        g2["REND_THA"] = np.where(g2["COSE"]>0, g2["PROD"]/g2["COSE"], np.nan)
        fig2 = px.bar(g2, x="FECHA_YYYYMM", y=["SIEM","COSE"], barmode="group",
                      labels={"value":"ha","variable":"Superficie"})
        fig2.add_scatter(x=g2["FECHA_YYYYMM"], y=g2["REND_THA"], mode="lines", name="Rendimiento (t/ha)", yaxis="y2")
        fig2.update_layout(
            yaxis2=dict(title="t/ha", overlaying="y", side="right"),
            margin=dict(l=10,r=10,t=10,b=10)
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Mapa: métrica anual por distrito")
    if gj is None:
        st.warning("No se encontró el GeoJSON de distritos. Coloca `geo/distritos_piura.geojson` con campo UBIGEO.")
    else:
        metric_opt = st.selectbox("Métrica a mapear",
                                  options=["REND_THA_ANUAL","PROD_SUM","PRECIO_POND"],
                                  index=0,
                                  help="REND_THA_ANUAL (t/ha), PROD_SUM (t), PRECIO_POND (S/ kg)")
        # Base anual
        if anu.empty:
            st.info("No hay agregados anuales; usaré una agregación rápida en memoria.")
            base_map = (df_f.groupby(["UBIGEO","ANIO"], as_index=False)
                          .agg(PROD_SUM=("PRODUCCION","sum"),
                               COSE_SUM=("COSECHA","sum"),
                               PRECIO_POND=("PRECIO_CHACRA", lambda s: weighted_mean(s, df_f.loc[s.index,"PRODUCCION_KG"]))))
            base_map["REND_THA_ANUAL"] = np.where(base_map["COSE_SUM"]>0, base_map["PROD_SUM"]/base_map["COSE_SUM"], np.nan)
        else:
            base_map = anu.copy()
            base_map = base_map[(base_map["ANIO"]>=rango_anios[0]) & (base_map["ANIO"]<=rango_anios[1])]
            base_map = base_map.groupby("UBIGEO", as_index=False).agg(
                REND_THA_ANUAL=("REND_THA_ANUAL","mean"),
                PROD_SUM=("PROD_SUM","sum"),
                PRECIO_POND=("PRECIO_POND","mean")
            )

        # Plotly choropleth
        fig_map = px.choropleth(
            base_map,
            geojson=gj,
            color=metric_opt,
            featureidkey="properties.UBIGEO",  # ajusta si tu GeoJSON usa otra clave
            locations="UBIGEO",
            projection="mercator",
            color_continuous_scale="YlGn"
        )
        fig_map.update_geos(fitbounds="locations", visible=False)
        fig_map.update_layout(margin=dict(l=10,r=10,t=10,b=10))
        st.plotly_chart(fig_map, use_container_width=True)

# ========== Tab 2: Mercado ==========
with tab2:
    st.subheader("Producción vs Precio chacra")
    g3 = (df_f.groupby("FECHA_YYYYMM", as_index=False)
              .agg(PROD=("PRODUCCION","sum"),
                   PRECIO_POND=("PRECIO_CHACRA", lambda s: weighted_mean(s, df_f.loc[s.index,"PRODUCCION_KG"]))))
    fig3 = px.bar(g3, x="FECHA_YYYYMM", y="PROD", labels={"PROD":"Producción (t)"})
    fig3.add_scatter(x=g3["FECHA_YYYYMM"], y=g3["PRECIO_POND"], mode="lines", name="Precio chacra (S/ kg)")
    fig3.update_layout(margin=dict(l=10,r=10,t=10,b=10))
    st.plotly_chart(fig3, use_container_width=True)

    # tabla de sobreoferta simple: picos de producción (p85) con caída de precios (diferencia negativa)
    st.subheader("Eventos de posible sobreoferta")
    g3["PRECIO_POND_LAG1"] = g3["PRECIO_POND"].shift(1)
    g3["DELTA_PRECIO"] = g3["PRECIO_POND"] - g3["PRECIO_POND_LAG1"]
    p85 = g3["PROD"].quantile(0.85) if g3["PROD"].notna().any() else np.nan
    eventos = g3[(g3["PROD"]>=p85) & (g3["DELTA_PRECIO"]<0)]
    st.dataframe(eventos, use_container_width=True)

# ========== Tab 3: Predicción ==========
with tab3:
    st.subheader(f"Proyección 1–{lag_meses} meses usando VERDE_LAG")
    # factor histórico mediana(PROD/VERDE_LAG) por cultivo (si seleccionaste uno), sino global
    base_pred = df_f.copy()
    base_pred = base_pred[base_pred["VERDE_LAG"].notna() & (base_pred["VERDE_LAG"]>0)]
    if cultivo_sel != "(Todos)":
        factor = np.median((base_pred["PRODUCCION"] / base_pred["VERDE_LAG"]).dropna())
    else:
        # toma mediana por cultivo y promedia (robusto simple)
        tmp = (base_pred.assign(RATIO=base_pred["PRODUCCION"]/base_pred["VERDE_LAG"])
                        .groupby("CULTIVO")["RATIO"].median().dropna())
        factor = float(tmp.median()) if len(tmp) else np.nan

    g4 = (df_f.groupby("FECHA_YYYYMM", as_index=False)
              .agg(PROD=("PRODUCCION","sum"),
                   VERDE_LAG=("VERDE_LAG","sum")))
    g4["PROY_PROD"] = g4["VERDE_LAG"] * factor if pd.notna(factor) else np.nan

    fig4 = px.line(g4, x="FECHA_YYYYMM", y="PROD", labels={"PROD":"Producción (t)"}, title=None)
    fig4.add_scatter(x=g4["FECHA_YYYYMM"], y=g4["PROY_PROD"], mode="lines", name="Proyección (t)")
    fig4.update_layout(margin=dict(l=10,r=10,t=10,b=10))
    st.plotly_chart(fig4, use_container_width=True)

    st.caption("Modelo simple y abierto: Proyección = VERDE_LAG × mediana(PRODUCCION/VERDE_LAG). Reemplazable por ARIMA/Regresión más adelante.")

# ========== Tab 4: Calidad de datos ==========
with tab4:
    st.subheader("Cobertura de precio por distrito-año")
    if cov.empty:
        st.info("No se encontró `cobertura_precio_distrito_anio.csv`. Muestra cálculo aproximado en memoria.")
        tmp = df_f.copy()
        tmp["_has_prod"] = tmp["PRODUCCION"].fillna(0) > 0
        tmp["_price_ok"] = tmp["PRECIO_CHACRA"].notna() & (tmp["PRECIO_CHACRA"] > 0)
        cov_mem = (tmp.groupby(["UBIGEO","ANIO"], as_index=False)
                      .agg(MESES_CON_PROD=("_has_prod","sum"),
                           MESES_CON_PRECIO_OK=("_price_ok","sum")))
        cov_mem["PRICE_COVERAGE"] = np.where(
            cov_mem["MESES_CON_PROD"]>0,
            cov_mem["MESES_CON_PRECIO_OK"]/cov_mem["MESES_CON_PROD"], 1.0
        )
        cov_mem["ALL_PRICE_OK"] = cov_mem["PRICE_COVERAGE"].eq(1.0)
        st.dataframe(cov_mem, use_container_width=True)
    else:
        mask_cov = (cov["ANIO"]>=rango_anios[0]) & (cov["ANIO"]<=rango_anios[1])
        st.dataframe(cov.loc[mask_cov].sort_values(["ANIO","UBIGEO"]), use_container_width=True)

    st.markdown("---")
    st.subheader("Metodología de calidad")
    st.markdown("""
- **Cobertura de precio**: proporción de meses con producción > 0 que registran un precio chacra válido (>0).
- **ALL_PRICE_OK**: `True` si todos los meses con producción registran precio válido.
- **Rendimiento (t/ha)**: calculado sobre agregados `sum(PRODUCCION)/sum(COSECHA)` para evitar sesgos por meses con `COSECHA=0`.
""")

# ========== Tab 5: Fuentes ==========
with tab5:
    st.subheader("Fuentes y reproducibilidad")
    st.markdown("""
**Datasets**
- Campaña agrícola de los principales cultivos — Gobierno Regional Piura (GRP).
- GeoJSON distritos (UBIGEO 6 dígitos) — INEI/MINAM (procesado a Piura).

**Unidades**
- PRODUCCIÓN en toneladas (t); convertida a kilogramos (kg) para comparar con **PRECIO_CHACRA** (S/ por kg).

**Proxy y predicción**
- `VERDE_ACTUAL` con desfase (**lag** 1–3 meses) como proxy de producción futura:
  - Proyección simple: `VERDE_LAG × mediana(PRODUCCION/VERDE_LAG)`.

**Licencia**
- Código: MIT. Datos abiertos según licenciamiento de las fuentes originales.

**Notas**
- Este tablero **complementa** los oficiales (MIDAGRI): integra rendimiento, cobertura de precios y proyección abierta y replicable.
""")

# ---------- Footer ----------
st.markdown("---")
st.caption("Construido con Streamlit • Código abierto • Reproducible")
