# app/app.py
# -*- coding: utf-8 -*-
import json
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import plotly.graph_objects as go
from streamlit_plotly_events import plotly_events

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

def help_box(title: str, body_md: str):
    """
    Muestra una ayuda contextual.
    Usa st.popover si está disponible, si no usa st.expander como fallback.
    """
    try:
        pop = st.popover(title)  # Disponible en streamlit >=1.29
        with pop:
            st.markdown(body_md)
    except Exception:
        with st.expander(title):
            st.markdown(body_md)

def render_map_with_click(fig):
    """
    Dibuja el fig y retorna UBIGEO del punto clickeado (o None).
    """
    selected = plotly_events(
        fig,
        click_event=True,
        select_event=False,
        hover_event=False,
        override_width="100%",
        override_height=420
    )
    if selected and len(selected) > 0:
        pt = selected[0]
        # En choropleth, Plotly coloca el código de área en 'location'
        return pt.get("location")
    return None

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
st.sidebar.caption("""
**Lag VERDE_ACTUAL:**  
- Desplaza `VERDE_ACTUAL` **hacia adelante** para compararlo con la producción futura.  
- Ejemplo: con lag=2, el valor de *noviembre* de `VERDE_ACTUAL` se compara con la producción de *enero*.  
- Sirve como **alerta temprana** de posibles picos de oferta.
""")
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
    # left, right = st.columns([1.15, 1])
    # with left:
        # st.subheader("Serie: Producción total (t) y Superficie verde desplazada (ha)")
        # # agregación mensual
        # g1 = (df_f.groupby("FECHA_YYYYMM", as_index=False)
        #           .agg(PROD=("PRODUCCION","sum"),
        #                VERDE_LAG=("VERDE_LAG","sum")))
        # fig1 = px.bar(g1, x="FECHA_YYYYMM", y="PROD", labels={"PROD":"Producción (t)", "FECHA_YYYYMM":"Mes"})
        # fig1.add_scatter(x=g1["FECHA_YYYYMM"], y=g1["VERDE_LAG"], mode="lines", name=f"VERDE_LAG ({lag_meses}m)")
        # fig1.update_layout(margin=dict(l=10,r=10,t=10,b=10))
        # st.plotly_chart(fig1, use_container_width=True)
        # help_box(
        # "ℹ️ ¿Cómo leer este gráfico?",
        # f"""
        # **Barras**: producción mensual (t).  
        # **Línea**: Superficie verde actual **desplazada** *{lag_meses}* mes(es).  
        # - El *desplazamiento* mueve los valores de la superficie verde hacia adelante *{lag_meses}* mes(es) para compararlos con la producción **futura**. Ej. Si vemos la barra de producción de mayo, la línea muestra la superficie verde de marzo.  
        # - Si la línea sube hoy, es común ver barras altas **después** → alerta de posible **pico de oferta**.
        # - Superficie verde actual funciona como **proxy de oferta futura**:  si hoy el “verde” sube, en los próximos meses suele **aumentar la producción**.  
        # - Si la línea sube, espera barras más altas **después** (picos de oferta).
        # """
        # )

    st.subheader("Serie: Producción (t) y Superficie verde — lag (ha)")
    # Agregación mensual ordenada
    tmp = df_f.sort_values("FECHA_YYYYMM").copy()
    g1 = (tmp.groupby("FECHA_YYYYMM", as_index=False)
            .agg(
                produccion_total_t=("PRODUCCION","sum"),
                superficie_verde_lag_ha=("VERDE_LAG","sum")
            ))

    # -----  padding para que superficie verde lag no quede pegada arriba -----
    y2_max = g1["superficie_verde_lag_ha"].max() if g1["superficie_verde_lag_ha"].notna().any() else 0
    y2_top = y2_max * 2 if y2_max and y2_max > 0 else 1

    # ----- Figura con doble eje -----
    fig1 = go.Figure()

    # Barras: Producción (t) - eje izquierdo
    fig1.add_trace(go.Bar(
        x=g1["FECHA_YYYYMM"],
        y=g1["produccion_total_t"],
        name="Producción (t)",
        hovertemplate="Mes: %{x|%Y-%m}<br>Producción: %{y:,.0f} t<extra></extra>"
    ))

    # Línea: Superficie verde — lag (ha) - eje derecho
    fig1.add_trace(go.Scatter(
        x=g1["FECHA_YYYYMM"],
        y=g1["superficie_verde_lag_ha"],
        name=f"Superficie verde — lag {lag_meses} m (ha)",
        mode="lines+markers",
        yaxis="y2",
        hovertemplate="Mes: %{x|%Y-%m}<br>Superficie verde (lag): %{y:,.0f} ha<extra></extra>"
    ))

    fig1.update_layout(
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(title="Mes",  tickformat="%Y-%m"),
        yaxis=dict(title="Producción (t)"),
        yaxis2=dict(
            title="Superficie verde (ha)",
            overlaying="y",
            side="right",
            range=[0, y2_top]
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
    )

    st.plotly_chart(fig1, use_container_width=True)

    help_box(
        "ℹ️ ¿Cómo leer este gráfico?",
        f"""
    - **Barras**: producción mensual (t).  
    - **Línea**: superficie verde **desplazada** (lag) {lag_meses} mes(es) (proxy de oferta futura).  
    - **Lectura**: si la línea sube hoy, espera barras más altas **después** (picos de oferta).  
    - **Ejemplo**: la producción de mayo se compara con la superficie verde de **marzo** si el lag es **2**.
    """
    )

    # with right:
    #     st.markdown("""
    #     <div style="height:100%; display:flex; align-items:flex-end;">
    #         <div>
    #         <b>ℹ️ ¿Cómo leer este gráfico?</b><br>
    #         - <b>Barras</b>: producción mensual (t).<br>
    #         - <b>Línea</b>: superficie verde desplazada (lag) {lag_meses} mes(es) (proxy de oferta futura).<br>
    #         - <b>Lectura</b>: si la línea sube hoy, espera barras más altas después (picos de oferta).<br>
    #         - <b>Ejemplo</b>: la producción de mayo se compara con la superficie verde de marzo si el lag es 2.
    #         </div>
    #     </div>
    #     """, unsafe_allow_html=True)

    st.subheader("Mapa: métrica anual por distrito")
    help_box(
        "ℹ️ ¿Cómo leer el mapa?",
        """
    - Muestra la **métrica anual por distrito** con los filtros aplicados (cultivo y años).
    - Distritos sin datos se ven en **gris claro** y el tooltip indica **“Sin valor registrado”**.
    - Útil para priorizar **territorios** con menor rendimiento, menor producción o menor precio medio.
    """
    )

    if gj is None:
        st.warning("No se encontró el GeoJSON de distritos. Coloca `geo/distritos_piura.geojson` con campo UBIGEO y NOMBDIST.")
    else:
        # ---- 1) Extraer lista completa de distritos desde el GeoJSON (UBIGEO + NOMBDIST)
        gj_rows = []
        for ft in gj.get("features", []):
            props = ft.get("properties", {})
            gj_rows.append({
                "UBIGEO": str(props.get("UBIGEO", "")).zfill(6),
                "NOMBDIST": props.get("NOMBDIST", "")
            })
        df_dists = pd.DataFrame(gj_rows).drop_duplicates(subset=["UBIGEO"])

        # ---- 2) Agregados por distrito-año desde df_f (filtrado por cultivo / años)
        g_map = (df_f.groupby(["UBIGEO","ANIO"], as_index=False)
                    .agg(
                        produccion_total_t=("PRODUCCION","sum"),
                        cosecha_total_ha=("COSECHA","sum")
                    ))

        g_map["rendimiento_t_ha"] = np.where(
            g_map["cosecha_total_ha"] > 0,
            g_map["produccion_total_t"] / g_map["cosecha_total_ha"],
            np.nan
        )

        wprice = (df_f.groupby(["UBIGEO","ANIO"])
                    .apply(lambda g: weighted_mean(g["PRECIO_CHACRA"], g["PRODUCCION_KG"]))
                    .rename("precio_chacra_pond_skg")
                    .reset_index())
        g_map = g_map.merge(wprice, on=["UBIGEO","ANIO"], how="left")

        base_map = (g_map.groupby("UBIGEO", as_index=False)
                        .agg(
                            produccion_total_t=("produccion_total_t","sum"),
                            rendimiento_t_ha=("rendimiento_t_ha","mean"),
                            precio_chacra_pond_skg=("precio_chacra_pond_skg","mean")
                        ))

        # ---- 3) Convertir UBIGEO a str en ambos DF antes del merge
        df_dists["UBIGEO"] = df_dists["UBIGEO"].astype(str).str.zfill(6)
        base_map["UBIGEO"] = base_map["UBIGEO"].astype(str).str.zfill(6)

        # Merge con TODOS los distritos
        base_map = df_dists.merge(base_map, on="UBIGEO", how="left")

        # ---- 4) Selector de métrica legible
        metric_options = {
            "Rendimiento (t/ha)": "rendimiento_t_ha",
            "Producción total (t)": "produccion_total_t",
            "Precio chacra (S/ kg)": "precio_chacra_pond_skg",
        }
        metric_label = st.selectbox("Métrica a mapear", options=list(metric_options.keys()), index=0)
        metric_col = metric_options[metric_label]

        # ---- 5) Separar válidos vs. NaN para pintar NaN en gris claro
        df_valid = base_map[base_map[metric_col].notna()].copy()
        df_nan   = base_map[base_map[metric_col].isna()].copy()

        # Choropleth principal (válidos)
        fig_map = px.choropleth(
            df_valid,
            geojson=gj,
            locations="UBIGEO",
            featureidkey="properties.UBIGEO",
            color=metric_col,
            projection="mercator",
            color_continuous_scale="YlGn",
            labels={metric_col: metric_label}
        )
        fig_map.update_geos(fitbounds="locations", visible=False)

        # Tooltip personalizados para válidos
        hover_valid = (
        "<b>Distrito:</b> %{customdata[0]}<br>"
        "<b>UBIGEO:</b> %{location}<br>"
        "<b>" + metric_label + ":</b> %{z:,.2f}"
        "<extra></extra>"
        )

        fig_map.update_traces(
            hovertemplate=hover_valid,
            customdata=np.stack([df_valid["NOMBDIST"].values], axis=-1)
        )

        # Capa para NaN (gris claro)
        if not df_nan.empty:
            fig_map.add_trace(go.Choropleth(
                geojson=gj,
                locations=df_nan["UBIGEO"],
                z=[0]*len(df_nan),
                featureidkey="properties.UBIGEO",
                showscale=False,
                marker_line_width=0.2,
                marker_line_color="white",
                colorscale=[[0, "lightgray"], [1, "lightgray"]],
                hovertemplate=(
                    "<b>Distrito:</b> %{customdata[0]}<br>"
                    "<b>UBIGEO:</b> %{location}<br>"
                    f"<b>{metric_label}:</b> Sin valor registrado"
                    "<extra></extra>"
                ),
                customdata=np.stack([df_nan["NOMBDIST"].values], axis=-1)
            ))

        fig_map.update_layout(
            margin=dict(l=10, r=10, t=10, b=10),
            coloraxis_colorbar=dict(title=metric_label),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
        )

        st.plotly_chart(fig_map, use_container_width=True)

# ========== Tab 2: Mercado =========
with tab2:
    st.subheader("Producción vs Precio chacra")
    # Agregación mensual filtrada (ya aplica cultivo y rango de años del sidebar)
    g3 = (df_f.groupby("FECHA_YYYYMM", as_index=False)
              .agg(PROD=("PRODUCCION","sum")))

    # Precio chacra ponderado por volumen (kg)
    # Usamos los índices de df_f para alinear pesos con precios
    price_pond = (df_f
                  .groupby("FECHA_YYYYMM")
                  .apply(lambda g: weighted_mean(g["PRECIO_CHACRA"], g["PRODUCCION_KG"]))
                  .rename("PRECIO_POND")
                  .reset_index())

    g3 = g3.merge(price_pond, on="FECHA_YYYYMM", how="left")

    # Figura con doble eje y
    fig = go.Figure()

    # Barras: Producción (eje izquierdo)
    fig.add_trace(go.Bar(
        x=g3["FECHA_YYYYMM"],
        y=g3["PROD"],
        name="Producción (t)",
        hovertemplate="Mes: %{x|%Y-%m}<br>Producción: %{y:,.0f} t<extra></extra>"
    ))

    # Línea: Precio chacra (eje derecho)
    fig.add_trace(go.Scatter(
        x=g3["FECHA_YYYYMM"],
        y=g3["PRECIO_POND"],
        name="Precio chacra (S/ kg)",
        mode="lines+markers",
        yaxis="y2",
        hovertemplate="Mes: %{x|%Y-%m}<br>Precio chacra: S/ %{y:,.2f}/kg<extra></extra>"
    ))

    fig.update_layout(
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(title="Mes"),
        yaxis=dict(title="Producción (t)"),
        yaxis2=dict(title="Precio chacra (S/ kg)", overlaying="y", side="right"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
    )

    st.plotly_chart(fig, use_container_width=True)
    st.markdown("""
        **¿Qué estás viendo?**  
        - **Barras**: producción mensual total (toneladas).  
        - **Línea**: precio promedio ponderado en chacra (S/ por kg).  
        **¿Para qué sirve?**  
        - Detectar posibles **sobreofertas**: meses con picos de producción que coinciden con **caídas de precio** (impacto en ingresos).  
        **Cómo leerlo:**  
        - Barras altas + línea que baja = señal de presión de oferta en el mercado.  
        - Barras bajas + línea que sube = escasez relativa, precios más altos.
        """)
    # Tabla de “sobreoferta”
    st.subheader("Eventos de posible sobreoferta")
    g3 = g3.sort_values("FECHA_YYYYMM")
    g3["PRECIO_POND_LAG1"] = g3["PRECIO_POND"].shift(1)
    g3["DELTA_PRECIO"] = g3["PRECIO_POND"] - g3["PRECIO_POND_LAG1"]
    p85 = g3["PROD"].quantile(0.85) if g3["PROD"].notna().any() else np.nan
    eventos = g3[(g3["PROD"] >= p85) & (g3["DELTA_PRECIO"] < 0)]
    st.dataframe(eventos, use_container_width=True)
    st.markdown("""
        **¿Qué muestra esta tabla?**  
        - Meses donde la producción estuvo en el **percentil 85 o superior** (picos) y el **precio cayó** respecto al mes anterior.  
        - Sirve como evidencia concreta de **sobreoferta** que presiona el precio a la baja.
        """)    
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
    st.markdown(f"""
    **¿Qué estás viendo?**  
    - **Línea 1 (observada)**: producción registrada.  
    - **Línea 2 (proyección)**: `VERDE_LAG × mediana(PRODUCCION/VERDE_LAG)` con lag de **{lag_meses}** mes(es).  
    **¿Para qué sirve?**  
    - Anticipar producción **1–{lag_meses}** meses antes con un método **abierto y replicable**.  
    **Cómo leerlo:**  
    - Si las curvas coinciden, el proxy funciona bien.  
    - Si la proyección supera la observada, podría venir **sobreoferta**; si queda por debajo, **escasez**.
    """)

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
