
# app_streamlit_piura_v6.py
# ------------------------------------------------------------
# Fix: mapa distrital no mostraba datos (merge int vs str y join por nombre).
# - Unifica UBIGEO como string con zfill(6) en ambos lados.
# - KPIs por distrito calculados por UBIGEO (sin depender de nombre).
# - El nombre de distrito para tooltip proviene del GeoJSON.
# - Sin uploaders: lee desde rutas internas (data_proc/ y geo/).
# ------------------------------------------------------------

from pathlib import Path
import json
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ---------------------------
# Rutas internas
# ---------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data_proc"
GEO_DIR  = BASE_DIR / "geo"

CSV_ANUAL = DATA_DIR / "dataset_piura_anual.csv"
CSV_CLEAN = DATA_DIR / "dataset_piura_clean.csv"
GJ_DIST   = GEO_DIR  / "distritos_piura.geojson"   # requiere properties: UBIGEO, NOMBDIST

# ---------------------------
# Utilidades generales
# ---------------------------
def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {path}")
    try:
        return pd.read_csv(path)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin-1")

def _norm_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    return df

def to_year_datetime(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series.astype("Int64").astype(str) + "-01-01", errors="coerce")

def shannon_index_df(df, area_col="area_sembrada_ha"):
    total = df[area_col].sum()
    if total <= 0 or df.empty:
        return np.nan
    p = df[area_col] / total
    p = p[p > 0]
    return float(-np.sum(p * np.log(p)))

def herfindahl_index_df(df, area_col="area_sembrada_ha"):
    total = df[area_col].sum()
    if total <= 0 or df.empty:
        return np.nan
    p = df[area_col] / total
    return float(np.sum(p**2))

def compute_flex_regional(df_anual):
    g = df_anual.groupby(["anio","tipo_cultivo"], as_index=False)["area_sembrada_ha"].sum()
    piv = g.pivot_table(index="anio", columns="tipo_cultivo", values="area_sembrada_ha", aggfunc="sum").fillna(0.0)
    denom = (piv.get("Transitorio", 0.0) + piv.get("Permanente", 0.0))
    flex = np.where(denom > 0, piv.get("Transitorio", 0.0) / denom, np.nan)
    out = pd.DataFrame({"anio": piv.index, "flexibilidad": flex})
    return out.sort_values("anio")

def risk_semaforo(df_anual, area_thr=15.0, price_thr=-10.0):
    df = df_anual.copy().sort_values(["cultivo_std","anio"])
    df["area_prom_3_prev"] = (
        df.groupby("cultivo_std", group_keys=False)["area_sembrada_ha"]
          .apply(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
    )
    denom_area = df["area_prom_3_prev"].replace({0: np.nan})
    df["delta_area_pct"] = (df["area_sembrada_ha"] - denom_area) / denom_area * 100.0

    if "precio_prom_s_kg" in df.columns:
        df["precio_prom_3_prev"] = (
            df.groupby("cultivo_std", group_keys=False)["precio_prom_s_kg"]
              .apply(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
        )
        denom_precio = df["precio_prom_3_prev"].replace({0: np.nan})
        df["delta_precio_pct"] = (df["precio_prom_s_kg"] - denom_precio) / denom_precio * 100.0
    else:
        df["delta_precio_pct"] = np.nan

    df["score_riesgo"] = 0
    df.loc[df["delta_area_pct"] > area_thr, "score_riesgo"] += 1
    df.loc[df["delta_precio_pct"] < price_thr, "score_riesgo"] += 1
    df["riesgo"] = df["score_riesgo"].map({2:"Alto",1:"Medio",0:"Bajo"}).fillna("Bajo")
    return df

# ---------------------------
# KPIs distritales por UBIGEO (solo usa UBIGEO como llave)
# ---------------------------
def kpis_by_ubigeo(df_clean_period: pd.DataFrame) -> pd.DataFrame:
    """
    KPIs por distrito para el periodo filtrado:
      - flexibilidad = transitorios / (transitorios + permanentes), usando SIEMBRA
      - diversificación (Shannon) solo transitorios, usando SIEMBRA
    Output: ubigeo (str, 6 dígitos), flexibilidad (0..1), shannon_trans
    """
    d = df_clean_period.copy()
    # columnas mínimas
    needed = {"ubigeo","tipo_cultivo","cultivo_std","siembra"}
    if not needed.issubset(d.columns):
        return pd.DataFrame()

    # normalizar UBIGEO como string 6 dígitos
    d["ubigeo"] = d["ubigeo"].astype(str).str.strip().str.zfill(6)

    # Flexibilidad por UBIGEO
    g_tp = d.groupby(["ubigeo","tipo_cultivo"], as_index=False)["siembra"].sum()
    piv = g_tp.pivot_table(index="ubigeo", columns="tipo_cultivo", values="siembra", aggfunc="sum").fillna(0)
    denom = (piv.get("Transitorio",0) + piv.get("Permanente",0))
    piv["flexibilidad"] = np.where(denom > 0, piv.get("Transitorio",0)/denom, np.nan)
    piv = piv.reset_index()[["ubigeo","flexibilidad"]]

    # Diversificación (solo transitorios) por UBIGEO
    div_list = []
    d_trans = d[d["tipo_cultivo"].eq("Transitorio")]
    for ubi, grp in d_trans.groupby("ubigeo"):
        sums = grp.groupby("cultivo_std", as_index=False)["siembra"].sum()
        sh = shannon_index_df(sums, area_col="siembra")
        div_list.append({"ubigeo": ubi, "shannon_trans": sh})
    div_df = pd.DataFrame(div_list)

    out = piv.merge(div_df, on="ubigeo", how="left")
    return out

# ---------------------------
# Cargar datos
# ---------------------------
df_anual = _norm_cols(_read_csv(CSV_ANUAL))
df_clean = _norm_cols(_read_csv(CSV_CLEAN))

# alias comunes
aliases = {"año":"anio","cultivo":"cultivo_std","tipo":"tipo_cultivo",
           "produccion":"produccion_t","precio_promedio_s_kg":"precio_prom_s_kg"}
for old,new in aliases.items():
    if old in df_anual.columns and new not in df_anual.columns:
        df_anual[new] = df_anual[old]

# ---------------------------
# App
# ---------------------------
st.set_page_config(page_title="Piura • Elección de Cultivos (v6)", layout="wide")
st.title("🌾 Piura • Elección de Cultivos (v6) — mapa corregido")
st.caption("El mapa distrital usa UBIGEO como llave única, con distritos sin datos en gris y tooltip 'sin datos'.")

# Filtros
years = sorted(df_anual["anio"].dropna().unique().tolist())
y_min, y_max = years[0], years[-1]
sel_years = st.select_slider("Rango de años", options=years, value=(y_min, y_max))

df_anual_f = df_anual[df_anual["anio"].between(sel_years[0], sel_years[1])].copy()
df_clean_f = df_clean[df_clean["anio"].between(sel_years[0], sel_years[1])].copy()

# ---------------------------
# Mapa distrital: flexibilidad (color) + diversificación en tooltip
# ---------------------------
st.subheader("Mapa distrital • flexibilidad (color) y diversificación (tooltip)")
try:
    with open(GJ_DIST, "r", encoding="utf-8") as f:
        gj = json.load(f)

    # Normalizar UBIGEO en GeoJSON (string 6 dígitos)
    for feat in gj["features"]:
        props = feat.get("properties", {})
        props["UBIGEO"]   = str(props.get("UBIGEO","")).strip().zfill(6)
        props["NOMBDIST"] = str(props.get("NOMBDIST","")).strip()
        feat["properties"] = props

    # KPIs por UBIGEO (periodo)
    kpis = kpis_by_ubigeo(df_clean_f)
    if not kpis.empty:
        kpis["ubigeo"] = kpis["ubigeo"].astype(str).str.zfill(6)

    # Tabla base de nombres desde GeoJSON
    df_geo = pd.DataFrame([
        {"ubigeo": feat["properties"]["UBIGEO"], "distrito": feat["properties"]["NOMBDIST"]}
        for feat in gj["features"]
    ])

    # Left join: todos los distritos presentes en geojson
    df_map = df_geo.merge(kpis, on="ubigeo", how="left")

    df_yes = df_map[df_map["flexibilidad"].notna()].copy()
    df_no  = df_map[df_map["flexibilidad"].isna()].copy()

    figm = go.Figure()

    # Capa 1: SIN datos (gris claro)
    if not df_no.empty:
        figm.add_trace(go.Choropleth(
            geojson=gj,
            locations=df_no["ubigeo"],
            z=[0]*len(df_no),
            featureidkey="properties.UBIGEO",
            colorscale=[(0, "#e6e6e6"), (1, "#e6e6e6")],
            showscale=False,
            marker_line_width=0.2,
            hovertemplate="<b>%{customdata[0]}</b><br>"
                          "Flexibilidad: sin datos<br>"
                          "Diversificación (Shannon): sin datos<extra></extra>",
            customdata=np.stack([df_no["distrito"]], axis=-1),
            name="Sin datos"
        ))

    # Capa 2: CON datos (escala Viridis por flexibilidad)
    if not df_yes.empty:
        cd = np.stack([
            df_yes["distrito"],
            df_yes["flexibilidad"],
            df_yes["shannon_trans"]
        ], axis=-1)

        figm.add_trace(go.Choropleth(
            geojson=gj,
            locations=df_yes["ubigeo"],
            z=df_yes["flexibilidad"],
            featureidkey="properties.UBIGEO",
            colorscale="Viridis",
            zmin=0.0, zmax=1.0,
            colorbar_title="Flexibilidad",
            marker_line_width=0.2,
            hovertemplate="<b>%{customdata[0]}</b><br>"
                          "Flexibilidad: %{customdata[1]:.2f}<br>"
                          "Diversificación (Shannon): %{customdata[2]:.2f}<extra></extra>",
            customdata=cd,
            name="Con datos"
        ))

    figm.update_geos(fitbounds="locations", visible=False)
    figm.update_layout(
        title="Flexibilidad (color) y Diversificación (tooltip) por distrito",
        margin=dict(l=0, r=0, t=50, b=0)
    )

    st.plotly_chart(figm, use_container_width=True)
    st.caption(f"Con datos: {len(df_yes)} distritos • Sin datos: {len(df_no)} distritos")

except FileNotFoundError:
    st.info("No se encontró el GeoJSON de distritos. Asegura la ruta: geo/distritos_piura.geojson")
except Exception as e:
    st.warning(f"Ocurrió un problema cargando el mapa: {e}")
