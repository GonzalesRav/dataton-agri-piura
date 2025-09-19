
# app_streamlit_piura_v2.py
# ------------------------------------------------------------
# Dashboard: Campañas Agrícolas - Piura (auto-mapeo GRP + precios anualizados)
# Autor: Joki + ChatGPT
# Requisitos:
#   pip install streamlit pandas numpy plotly openpyxl
# Ejecución:
#   streamlit run app_streamlit_piura_v2.py
# ------------------------------------------------------------

import io
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# ---------------------------
# Utilidades
# ---------------------------

def _rename_lower(df: pd.DataFrame):
    return df.rename(columns={c: c.strip().lower() for c in df.columns})

def _read_any(uploaded):
    if uploaded is None:
        return None
    name = uploaded.name.lower()
    if name.endswith(".csv"):
        try:
            return pd.read_csv(uploaded)
        except UnicodeDecodeError:
            uploaded.seek(0)
            return pd.read_csv(uploaded, encoding="latin-1")
    if name.endswith(".xlsx") or name.endswith(".xls"):
        return pd.read_excel(uploaded)
    data = uploaded.read()
    try:
        return pd.read_csv(io.BytesIO(data))
    except Exception:
        try:
            return pd.read_excel(io.BytesIO(data))
        except Exception:
            return None

def _standardize_grp(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Acepta el esquema en mayúsculas del GRP (ANO, MES, PROVINCIA, DISTRITO, CULTIVO, SIEMBRA, COSECHA, PRODUCCION, PRECIO_CHACRA)
    y lo mapea al esquema estándar del dashboard:
    anio, campana(opc), provincia, distrito, cultivo, area_sembrada_ha, area_cosechada_ha, produccion_t, rendimiento_t_ha, precio_chacra (mensual)
    """
    if df_raw is None or df_raw.empty:
        return None

    # Trabajar con copia y columnas lower para simplificar
    df = _rename_lower(df_raw.copy())

    # Aliases posibles
    aliases = {
        "año": "ano",
        "year": "ano",
        "campaña": "campana",
        "producción": "produccion",
        "rendimiento_t/ha": "rendimiento_t_ha",
        "rendimiento_t_ha": "rendimiento_t_ha",
        "precio_chacra_s_kg": "precio_chacra"
    }
    for k, v in aliases.items():
        if k in df.columns and v not in df.columns:
            df[v] = df[k]

    # Si viene el esquema GRP "Formato_dataset_productos.csv"
    # Columnas esperadas: ano, mes, provincia, distrito, cultivo, siembra, cosecha, produccion, precio_chacra
    # + depart., ubigeo, fecha_muestra, etc. (opcionales)
    # Mapear a estándar
    std = pd.DataFrame()
    if "ano" in df.columns:
        std["anio"] = pd.to_numeric(df["ano"], errors="coerce").astype("Int64")
    elif "anio" in df.columns:
        std["anio"] = pd.to_numeric(df["anio"], errors="coerce").astype("Int64")

    for src, dst in [
        ("provincia", "provincia"),
        ("distrito", "distrito"),
        ("cultivo", "cultivo"),
        ("siembra", "area_sembrada_ha"),
        ("cosecha", "area_cosechada_ha"),
        ("produccion", "produccion_t"),
        ("precio_chacra", "precio_chacra")
    ]:
        if src in df.columns:
            std[dst] = df[src]

    # Título caso texto
    for c in ["provincia", "distrito", "cultivo"]:
        if c in std.columns:
            std[c] = std[c].astype(str).str.strip().str.title()

    # Tipos numéricos
    for c in ["area_sembrada_ha", "area_cosechada_ha", "produccion_t", "precio_chacra"]:
        if c in std.columns:
            std[c] = pd.to_numeric(std[c], errors="coerce")

    # Derivar rendimiento t/ha si no existe
    if "rendimiento_t_ha" not in std.columns:
        if "produccion_t" in std.columns and "area_cosechada_ha" in std.columns:
            std["rendimiento_t_ha"] = np.where(
                (std["area_cosechada_ha"] > 0) & np.isfinite(std["area_cosechada_ha"]),
                std["produccion_t"] / std["area_cosechada_ha"],
                np.nan
            )

    # Campaña opcional a partir de mes (si existe)
    if "mes" in df.columns:
        std["mes"] = pd.to_numeric(df["mes"], errors="coerce").astype("Int64")
        # Regla simple: Jul-Dic = "Grande", Ene-Jun = "Chica" (ajústalo si tienes calendario local)
        std["campana"] = np.where(std["mes"].isin([7,8,9,10,11,12]), "Grande",
                           np.where(std["mes"].isin([1,2,3,4,5,6]), "Chica", None))

    # Pasar departamento/distrito/ubigeo si existen
    for src in ["departamento", "ubigeo"]:
        if src in df.columns:
            std[src] = df[src]

    # Orden sugerido de columnas
    order = ["anio","campana","mes","departamento","provincia","distrito","ubigeo",
             "cultivo","area_sembrada_ha","area_cosechada_ha","produccion_t","rendimiento_t_ha","precio_chacra"]
    cols = [c for c in order if c in std.columns] + [c for c in std.columns if c not in order]
    std = std[cols]

    return std

def _annualize_prices_from_grp(std_df: pd.DataFrame) -> pd.DataFrame:
    """
    Si el GRP trae 'precio_chacra' mensual, calcular promedio anual por cultivo.
    Retorna DataFrame: anio, cultivo, precio_prom_s_kg
    """
    if std_df is None or std_df.empty:
        return None
    needed = {"anio","cultivo","precio_chacra"}
    if not needed.issubset(set(std_df.columns)):
        return None
    tmp = std_df.dropna(subset=["anio","cultivo","precio_chacra"]).copy()
    if tmp.empty:
        return None
    ann = tmp.groupby(["anio","cultivo"], as_index=False)["precio_chacra"].mean()
    ann = ann.rename(columns={"precio_chacra":"precio_prom_s_kg"})
    return ann

@st.cache_data(show_spinner=False)
def load_grp_standardized(file):
    raw = _read_any(file)
    if raw is None or raw.empty:
        return None
    return _standardize_grp(raw)

@st.cache_data(show_spinner=False)
def load_prices(file):
    df = _read_any(file)
    if df is None or df.empty:
        return None
    df = _rename_lower(df)
    # normalizar mínimos
    if "fecha" in df.columns:
        df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
        df["anio"] = df["fecha"].dt.year
    if "cultivo" in df.columns:
        df["cultivo"] = df["cultivo"].astype(str).str.strip().str.title()
    target = None
    if {"anio","cultivo","precio_soles_kg"}.issubset(df.columns):
        target = df.groupby(["anio","cultivo"], as_index=False)["precio_soles_kg"].mean()
        target = target.rename(columns={"precio_soles_kg":"precio_prom_s_kg"})
    elif {"anio","cultivo","precio_prom_s_kg"}.issubset(df.columns):
        target = df[["anio","cultivo","precio_prom_s_kg"]].copy()
    return target

def compute_risk(df_grp, df_prices, scope):
    if df_grp is None or df_grp.empty:
        return pd.DataFrame()

    g = df_grp.copy()
    # filtros
    if scope.get("anio_min") is not None and scope.get("anio_max") is not None and "anio" in g.columns:
        g = g[g["anio"].between(scope["anio_min"], scope["anio_max"])]
    for key, col in [("provincia","provincia"),("distrito","distrito"),("campana","campana"),("cultivo","cultivo")]:
        if scope.get(key) and col in g.columns:
            g = g[g[col].isin(scope[key])]

    # agregación anual por cultivo
    if not {"anio","cultivo","area_sembrada_ha"}.issubset(g.columns):
        return pd.DataFrame()

    agg = g.groupby(["anio","cultivo"], as_index=False).agg(area_sembrada_ha=("area_sembrada_ha","sum"))
    agg = agg.sort_values(["cultivo","anio"])

    # rolling promedio 3 años previos (excluye el año actual)
    agg["area_prom_3_prev"] = agg.groupby("cultivo")["area_sembrada_ha"].apply(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
    agg["delta_area_pct"] = (agg["area_sembrada_ha"] - agg["area_prom_3_prev"]) / agg["area_prom_3_prev"] * 100.0
    agg.loc[~np.isfinite(agg["delta_area_pct"]), "delta_area_pct"] = np.nan

    # merge precios si existen
    if df_prices is not None and not df_prices.empty and {"anio","cultivo","precio_prom_s_kg"}.issubset(df_prices.columns):
        m = pd.merge(agg, df_prices, on=["anio","cultivo"], how="left")
        m["precio_prom_3_prev"] = m.groupby("cultivo")["precio_prom_s_kg"].apply(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
        m["delta_precio_pct"] = (m["precio_prom_s_kg"] - m["precio_prom_3_prev"]) / m["precio_prom_3_prev"] * 100.0
        m.loc[~np.isfinite(m["delta_precio_pct"]), "delta_precio_pct"] = np.nan
    else:
        m = agg.copy()
        m["delta_precio_pct"] = np.nan

    # scoring
    m["score_riesgo"] = 0
    m.loc[m["delta_area_pct"] > 15, "score_riesgo"] += 1
    m.loc[m["delta_precio_pct"] < -10, "score_riesgo"] += 1

    def etiqueta(sc):
        if sc >= 2: return "Alto"
        if sc == 1: return "Medio"
        return "Bajo"

    m["riesgo"] = m["score_riesgo"].apply(etiqueta)
    return m

# ---------------------------
# UI
# ---------------------------

st.set_page_config(page_title="Piura • Campañas Agrícolas (v2)", layout="wide")

st.title("🌾 Campañas Agrícolas • Piura (v2)")
st.caption("Auto-mapea el esquema GRP (mayúsculas) y anualiza PRECIO_CHACRA si no subes archivo de precios.")

with st.sidebar:
    st.header("1) Cargar datos")
    grp_file = st.file_uploader("GRP - Campaña agrícola (CSV/XLSX)", type=["csv","xlsx","xls"])
    prices_file = st.file_uploader("Opcional: Precios (CSV/XLSX)", type=["csv","xlsx","xls"])

    df_grp = load_grp_standardized(grp_file) if grp_file else None
    df_prices_ext = load_prices(prices_file) if prices_file else None

    st.markdown("---")
    st.header("2) Filtros")

    if df_grp is not None:
        # Si no se subió archivo de precios, intentar anualizar desde PRECIO_CHACRA mensual del GRP
        derived_prices = _annualize_prices_from_grp(df_grp) if df_prices_ext is None else None
        df_prices = df_prices_ext if df_prices_ext is not None else derived_prices

        # Info de columnas
        years = sorted(df_grp["anio"].dropna().unique().tolist()) if "anio" in df_grp.columns else []
        crops = sorted(df_grp["cultivo"].dropna().unique().tolist()) if "cultivo" in df_grp.columns else []
        provincias = sorted(df_grp["provincia"].dropna().unique().tolist()) if "provincia" in df_grp.columns else []
        distritos = sorted(df_grp["distrito"].dropna().unique().tolist()) if "distrito" in df_grp.columns else []
        campanas = sorted([c for c in df_grp.get("campana", pd.Series(dtype=str)).dropna().unique().tolist()]) if "campana" in df_grp.columns else []

        year_min, year_max = st.select_slider("Rango de años", options=years if years else [0], value=(years[0], years[-1]) if years else (0, 0))
        sel_cultivo = st.multiselect("Cultivo(s)", options=crops, default=crops[:3] if len(crops) > 0 else [])
        sel_prov = st.multiselect("Provincia(s)", options=provincias, default=[])
        sel_dist = st.multiselect("Distrito(s)", options=distritos, default=[])
        sel_camp = st.multiselect("Campaña(s)", options=campanas, default=[])

        scope = {
            "anio_min": year_min, "anio_max": year_max,
            "cultivo": sel_cultivo, "provincia": sel_prov,
            "distrito": sel_dist, "campana": sel_camp
        }
    else:
        df_prices = None
        scope = {}

if df_grp is None:
    st.info("Carga el archivo de la campaña agrícola del GRP para empezar. 👈")
    st.stop()

# Aplicar filtros a df_grp
f = df_grp.copy()
if "anio" in f.columns and scope.get("anio_min") is not None and scope.get("anio_max") is not None:
    f = f[(f["anio"] >= scope["anio_min"]) & (f["anio"] <= scope["anio_max"])]
for key, col in [("cultivo","cultivo"),("provincia","provincia"),("distrito","distrito"),("campana","campana")]:
    if scope.get(key) and col in f.columns:
        f = f[f[col].isin(scope[key])]

# KPIs
col1, col2, col3, col4 = st.columns(4)
with col1:
    total_area = f["area_sembrada_ha"].sum() if "area_sembrada_ha" in f.columns else np.nan
    st.metric("Área sembrada (ha)", f"{total_area:,.0f}" if pd.notnull(total_area) else "—")
with col2:
    total_cosech = f["area_cosechada_ha"].sum() if "area_cosechada_ha" in f.columns else np.nan
    st.metric("Área cosechada (ha)", f"{total_cosech:,.0f}" if pd.notnull(total_cosech) else "—")
with col3:
    total_prod = f["produccion_t"].sum() if "produccion_t" in f.columns else np.nan
    st.metric("Producción (t)", f"{total_prod:,.0f}" if pd.notnull(total_prod) else "—")
with col4:
    avg_rend = f["rendimiento_t_ha"].replace({0: np.nan}).mean() if "rendimiento_t_ha" in f.columns else np.nan
    st.metric("Rendimiento (t/ha)", f"{avg_rend:,.2f}" if pd.notnull(avg_rend) else "—")

st.markdown("---")

# Gráfico 1: Evolución de área por cultivo
if {"anio","cultivo","area_sembrada_ha"}.issubset(f.columns):
    g1 = f.groupby(["anio","cultivo"], as_index=False)["area_sembrada_ha"].sum()
    fig1 = px.line(g1, x="anio", y="area_sembrada_ha", color="cultivo",
                   markers=True, title="Evolución del área sembrada por cultivo (ha)")
    st.plotly_chart(fig1, use_container_width=True)

# Gráfico 2: Top cultivos por área (último año del filtro)
if {"anio","cultivo","area_sembrada_ha"}.issubset(f.columns) and f["anio"].notna().any():
    last_year = int(f["anio"].max())
    top = f[f["anio"] == last_year].groupby("cultivo", as_index=False)["area_sembrada_ha"].sum().sort_values("area_sembrada_ha", ascending=False).head(10)
    fig2 = px.bar(top, x="cultivo", y="area_sembrada_ha", title=f"Top 10 cultivos por área sembrada (ha) • {last_year}")
    st.plotly_chart(fig2, use_container_width=True)

# Gráfico 3: Rendimiento por cultivo (promedio)
if {"cultivo","rendimiento_t_ha"}.issubset(f.columns):
    rmean = f.groupby("cultivo", as_index=False)["rendimiento_t_ha"].mean().sort_values("rendimiento_t_ha", ascending=False).head(15)
    fig3 = px.bar(rmean, x="cultivo", y="rendimiento_t_ha", title="Rendimiento promedio por cultivo (t/ha)")
    st.plotly_chart(fig3, use_container_width=True)

st.markdown("---")

# Semáforo de riesgo
st.subheader("🟢🟡🔴 Semáforo de riesgo de sobreoferta (por cultivo y año)")
risk_df = compute_risk(df_grp, df_prices, scope)
if risk_df.empty:
    st.info("Cargue datos válidos para calcular el semáforo. Si añade precios (o si PRECIO_CHACRA está en el GRP), el cálculo será más robusto.")
else:
    if "anio" in risk_df.columns:
        latest = risk_df["anio"].max()
        view = risk_df[risk_df["anio"] == latest].sort_values(["score_riesgo", "cultivo"], ascending=[False, True])
    else:
        view = risk_df.copy()
    cols = [c for c in ["anio","cultivo","delta_area_pct","delta_precio_pct","score_riesgo","riesgo"] if c in view.columns]
    st.dataframe(view[cols], use_container_width=True)

    # Heatmap
    if {"anio","cultivo","score_riesgo"}.issubset(risk_df.columns):
        pivot = risk_df.pivot_table(index="cultivo", columns="anio", values="score_riesgo", aggfunc="max").fillna(0)
        heat = px.imshow(pivot, aspect="auto", title="Mapa de calor: riesgo por cultivo x año (0=bajo, 2=alto)")
        st.plotly_chart(heat, use_container_width=True)

# Descarga del dataset filtrado
st.markdown("---")
st.subheader("Descarga")
csv = f.to_csv(index=False).encode("utf-8")
st.download_button("⬇️ Descargar datos filtrados (CSV)", data=csv, file_name="piura_filtrado.csv", mime="text/csv")

# Recuadro de recomendaciones
st.markdown("---")
st.subheader("Recomendaciones automáticas (beta)")
with st.expander("Cómo se generan"):
    st.write(
        "- Señal de **oferta**: expansión del área sembrada vs promedio de 3 años previos (> +15%).\n"
        "- Señal de **precio**: caída vs promedio 3 años previos (< -10%). Se toma de archivo de precios o de `PRECIO_CHACRA` anualizado del GRP.\n"
        "- Score: 0 (bajo), 1 (medio), 2 (alto)."
    )

if "anio" in df_grp.columns:
    latest = df_grp["anio"].max()
else:
    latest = None

if latest is not None and not risk_df.empty:
    latest_view = risk_df[risk_df["anio"] == latest]
    alto = latest_view[latest_view["riesgo"] == "Alto"]["cultivo"].tolist()
    medio = latest_view[latest_view["riesgo"] == "Medio"]["cultivo"].tolist()

    if alto:
        st.warning("🔴 Cultivos con **riesgo alto** de sobreoferta en el año más reciente: " + ", ".join(sorted(set(alto))))
    else:
        st.success("Sin cultivos con riesgo alto en el año más reciente según las señales disponibles.")

    if medio:
        st.info("🟡 Cultivos con **riesgo medio**: " + ", ".join(sorted(set(medio))))

st.caption("v2: Auto-mapeo GRP + anualización de PRECIO_CHACRA. Construido con cariño y cafeína. ☕🌱")
