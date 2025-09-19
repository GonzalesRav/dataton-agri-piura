
# app_streamlit_piura_v5.py
# ------------------------------------------------------------
# Dashboard Piura (v5): versión amigable para ciudadanía
# - Sin uploaders: lee datasets desde rutas internas
# - Calendario simplificado: bandas de meses y rangos (siembra/cosecha)
# - Mapa por distrito: tooltip con nombre y KPIs (diversificación transitorios y flexibilidad)
# - Indicadores regionales: evolución, burbuja, diversificación, flexibilidad, riesgo
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
GJ_DIST   = GEO_DIR  / "distritos_piura.geojson"   # debe tener 'UBIGEO' y 'NOMBDIST'

# ---------------------------
# Utilidades
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
    # promedio 3 años previos (por cultivo)
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

def month_band(data: pd.Series, threshold=0.4):
    """
    Recibe serie 12 meses normalizada (0..1) y devuelve texto compacto de rangos, ej: 'Ago–Oct; Dic–Ene'.
    threshold: por encima de qué intensidad se considera 'activo'.
    """
    # asegurar índice 1..12 con 0 si falta
    full = pd.Series({m: 0.0 for m in range(1,13)})
    full.update(data)
    active = [m for m, v in full.items() if v >= threshold]
    if not active:
        return "—"
    # agrupar meses contiguos
    ranges = []
    start = prev = active[0]
    for m in active[1:]:
        if m == prev + 1:
            prev = m
        else:
            ranges.append((start, prev))
            start = prev = m
    ranges.append((start, prev))
    # convertir a texto
    mnames = {1:"Ene",2:"Feb",3:"Mar",4:"Abr",5:"May",6:"Jun",7:"Jul",8:"Ago",9:"Sep",10:"Oct",11:"Nov",12:"Dic"}
    parts = []
    for a,b in ranges:
        parts.append(mnames[a] if a==b else f"{mnames[a]}–{mnames[b]}")
    return "; ".join(parts)

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


def simple_calendar(df_clean, top_n=10):
    """
    Devuelve un dataframe amigable: por cultivo transitorio,
    - banda de meses 'activos' de siembra y cosecha (texto)
    - barras horizontales con 12 segmentos (intensidad relativa)
    Solo muestra top_n cultivos por superficie promedio anual.
    """
    d = df_clean[df_clean["tipo_cultivo"].eq("Transitorio")].copy()
    if d.empty:
        return None, None, None

    # Elegir top cultivos por siembra total
    top = (d.groupby("cultivo_std", as_index=False)["siembra"].sum()
             .sort_values("siembra", ascending=False)
             .head(top_n)["cultivo_std"].tolist())
    d = d[d["cultivo_std"].isin(top)]

    sow = d.groupby(["cultivo_std","mes"], as_index=False)["siembra"].sum()
    har = d.groupby(["cultivo_std","mes"], as_index=False)["cosecha"].sum()
    # normalizar 0..1 por cultivo
    sow["sow_int"] = sow.groupby("cultivo_std")["siembra"].transform(lambda s: s / s.max() if s.max() not in [0, np.nan] else 0)
    har["har_int"] = har.groupby("cultivo_std")["cosecha"].transform(lambda s: s / s.max() if s.max() not in [0, np.nan] else 0)

    # bandas de meses (texto)
    bands = []
    for c in sorted(set(top)):
        s = sow[sow["cultivo_std"]==c].set_index("mes")["sow_int"]
        h = har[har["cultivo_std"]==c].set_index("mes")["har_int"]
        bands.append({
            "cultivo_std": c,
            "siembra_meses": month_band(s, threshold=0.4),
            "cosecha_meses": month_band(h, threshold=0.4)
        })
    bands_df = pd.DataFrame(bands)

    # tablas 12 columnas (para barras apiladas horizontales)
    sow_w = sow.pivot_table(index="cultivo_std", columns="mes", values="sow_int", aggfunc="mean").fillna(0).reindex(sorted(top))
    har_w = har.pivot_table(index="cultivo_std", columns="mes", values="har_int", aggfunc="mean").fillna(0).reindex(sorted(top))

    return bands_df, sow_w, har_w

def kpIs_by_district(df_clean_period):
    """
    Calcula KPIs por distrito para el periodo filtrado (según slider de años):
      - flexibilidad = transitorios / (transitorios + permanentes), usando SIEMBRA
      - diversificación (Shannon) solo transitorios, usando SIEMBRA
    Retorna DF con ubigeo, distrito, flexibilidad, shannon
    """
    d = df_clean_period.copy()
    # asegurar columnas
    needed = {"ubigeo","distrito","tipo_cultivo","cultivo_std","siembra"}
    if not needed.issubset(d.columns):
        return pd.DataFrame()

    # Flexibilidad por distrito
    g_tp = d.groupby(["ubigeo","distrito","tipo_cultivo"], as_index=False)["siembra"].sum()
    piv = g_tp.pivot_table(index=["ubigeo","distrito"], columns="tipo_cultivo", values="siembra", aggfunc="sum").fillna(0)
    piv["flexibilidad"] = np.where((piv.get("Transitorio",0)+piv.get("Permanente",0))>0,
                                   piv.get("Transitorio",0)/(piv.get("Transitorio",0)+piv.get("Permanente",0)),
                                   np.nan)
    piv = piv.reset_index()

    # Diversificación transitorios por distrito
    div_list = []
    for (ubi, dist), grp in d[d["tipo_cultivo"].eq("Transitorio")].groupby(["ubigeo","distrito"]):
        sh = shannon_index_df(grp.groupby("cultivo_std", as_index=False)["siembra"].sum(), area_col="siembra")
        div_list.append({"ubigeo": ubi, "distrito": dist, "shannon_trans": sh})
    div_df = pd.DataFrame(div_list)

    out = piv.merge(div_df, on=["ubigeo","distrito"], how="left")
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
st.set_page_config(page_title="Piura • Elección de Cultivos (v5)", layout="wide")
st.title("🌾 Piura • Elección de Cultivos (v5) — versión amigable")
st.caption("Incluye mapa distrital con KPIs en el tooltip, calendario simplificado y notas metodológicas para lectura crítica.")

# Filtros
years = sorted(df_anual["anio"].dropna().unique().tolist())
y_min, y_max = years[0], years[-1]
sel_years = st.select_slider("Rango de años", options=years, value=(y_min, y_max))

df_anual_f = df_anual[df_anual["anio"].between(sel_years[0], sel_years[1])].copy()
df_clean_f = df_clean[df_clean["anio"].between(sel_years[0], sel_years[1])].copy()

cultivos = sorted(df_anual["cultivo_std"].dropna().unique().tolist())
sel_cult_compare = st.multiselect("Comparar 2–3 cultivos (evolución de superficie)", options=cultivos, default=cultivos[:3] if len(cultivos)>=3 else cultivos)

st.markdown("---")


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
# # 0) Mapa por distrito (flexibilidad + diversificación transitorios)
# st.subheader("0) Mapa distrital • flexibilidad y diversificación (periodo seleccionado)")
# try:
#     import json, plotly.graph_objects as go

#     with open(GJ_DIST, "r", encoding="utf-8") as f:
#         gj = json.load(f)

#     # --- KPIs por distrito ---
#     kpis = kpIs_by_district(df_clean_f)

#     # Asegurar que ubigeo sea string en ambos
#     if not kpis.empty:
#         kpis["ubigeo"] = kpis["ubigeo"].astype(str)

#     geo_rows = []
#     for feat in gj["features"]:
#         props = feat.get("properties", {})
#         ubi  = str(props.get("UBIGEO", ""))
#         dist = props.get("NOMBDIST", "")
#         geo_rows.append({"ubigeo": ubi, "distrito": dist})
#     df_geo = pd.DataFrame(geo_rows)

#     # merge outer, así entran todos los distritos (con y sin datos)
#     df_map = df_geo.merge(kpis, on=["ubigeo","distrito"], how="left")

#     df_yes = df_map[df_map["flexibilidad"].notna()].copy()
#     df_no  = df_map[df_map["flexibilidad"].isna()].copy()

#     figm = go.Figure()

#     # Capa 1: sin datos (gris claro)
#     if not df_no.empty:
#         figm.add_trace(go.Choropleth(
#             geojson=gj,
#             locations=df_no["ubigeo"],
#             z=[0]*len(df_no),
#             featureidkey="properties.UBIGEO",
#             colorscale=[(0, "#e6e6e6"), (1, "#e6e6e6")],
#             showscale=False,
#             marker_line_width=0.2,
#             hovertemplate="<b>%{customdata[0]}</b><br>"
#                           "Flexibilidad: sin datos<br>"
#                           "Diversificación: sin datos<extra></extra>",
#             customdata=np.stack([df_no["distrito"]], axis=-1),
#             name="Sin datos"
#         ))

#     # Capa 2: con datos (Viridis por flexibilidad)
#     if not df_yes.empty:
#         cd = np.stack([
#             df_yes["distrito"],
#             df_yes["flexibilidad"],
#             df_yes["shannon_trans"]
#         ], axis=-1)

#         figm.add_trace(go.Choropleth(
#             geojson=gj,
#             locations=df_yes["ubigeo"],
#             z=df_yes["flexibilidad"],
#             featureidkey="properties.UBIGEO",
#             colorscale="Viridis",
#             zmin=0.0, zmax=1.0,
#             colorbar_title="Flexibilidad",
#             marker_line_width=0.2,
#             hovertemplate="<b>%{customdata[0]}</b><br>"
#                           "Flexibilidad: %{customdata[1]:.2f}<br>"
#                           "Diversificación (Shannon): %{customdata[2]:.2f}<extra></extra>",
#             customdata=cd,
#             name="Con datos"
#         ))

#     figm.update_geos(fitbounds="locations", visible=False)
#     figm.update_layout(
#         title="Flexibilidad (color) y Diversificación (tooltip) por distrito",
#         margin=dict(l=0, r=0, t=50, b=0)
#     )
#     st.plotly_chart(figm, use_container_width=True)

# except FileNotFoundError:
#     st.info("No se encontró el GeoJSON de distritos. Asegura la ruta: geo/distritos_piura.geojson")
# except Exception as e:
#     st.warning(f"Ocurrió un problema cargando el mapa: {e}")



st.markdown("---")

# 1) Evolución de áreas sembradas (comparativa, regional)
st.subheader("1) Evolución de áreas sembradas (comparativa) — Regional")
d1 = df_anual_f[df_anual_f["cultivo_std"].isin(sel_cult_compare)] if sel_cult_compare else df_anual_f.head(0)
if d1.empty:
    st.warning("Selecciona 2–3 cultivos para comparar.")
else:
    g1 = d1.groupby(["anio","cultivo_std"], as_index=False)["area_sembrada_ha"].sum()
    g1["anio_dt"] = to_year_datetime(g1["anio"])
    fig1 = px.line(g1, x="anio_dt", y="area_sembrada_ha", color="cultivo_std", markers=True,
                   title="Evolución del área sembrada (ha) — Regional")
    fig1.update_layout(xaxis_title="Año", yaxis_title="Área sembrada (ha)")
    fig1.update_xaxes(tickformat="%Y")
    st.plotly_chart(fig1, use_container_width=True)
with st.expander("Descripción y consideraciones"):
    st.write(
        "- Total regional (Piura). Compara 2–3 cultivos. Tendencias al alza sostenida pueden anticipar sobreoferta.\n"
        "- Limitación: no muestra heterogeneidad provincial/distrital."
    )

# 2) Producción vs superficie vs precio (burbuja, regional)
st.subheader("2) Producción vs superficie vs precio — Regional (cultivo–año)")
d2 = df_anual_f.copy()
if "precio_prom_s_kg" not in d2.columns:
    d2["precio_prom_s_kg"] = np.nan
d2["valor_produccion_aprox_s"] = d2["produccion_t"] * 1000.0 * d2["precio_prom_s_kg"]
fig2 = px.scatter(
    d2, x="area_sembrada_ha", y="produccion_t",
    size="valor_produccion_aprox_s", color="precio_prom_s_kg",
    hover_data=["anio","cultivo_std"],
    title="Área–Producción–Precio (tamaño ≈ valor; color = precio chacra S/ kg)"
)
fig2.update_layout(xaxis_title="Área sembrada (ha)", yaxis_title="Producción (t)")
st.plotly_chart(fig2, use_container_width=True)
with st.expander("Descripción y consideraciones"):
    st.write(
        "- Cada punto: cultivo–año. Tamaño ≈ valor económico (t × 1000 × S/ kg). Color = precio chacra promedio.\n"
        "- Si no hay precios, el color es neutro. Precios promedio regionales; variabilidad local no representada."
    )

# 3) Diversificación (transitorios) — Regional
st.subheader("3) Diversificación (solo transitorios) — Regional")
d_trans = df_anual_f[df_anual_f["tipo_cultivo"].eq("Transitorio")].copy()
div_rows = []
for y, dfy in d_trans.groupby("anio"):
    div_rows.append({"anio": y,
                     "shannon": shannon_index_df(dfy, "area_sembrada_ha"),
                     "herfindahl": herfindahl_index_df(dfy, "area_sembrada_ha")})
div_df = pd.DataFrame(div_rows).sort_values("anio")
div_df["anio_dt"] = to_year_datetime(div_df["anio"])
col_a, col_b = st.columns(2)
with col_a:
    fig3a = px.line(div_df, x="anio_dt", y="shannon", markers=True, title="Diversificación (Shannon) • transitorios — Regional")
    fig3a.update_layout(xaxis_title="Año", yaxis_title="Índice de Shannon")
    fig3a.update_xaxes(tickformat="%Y")
    st.plotly_chart(fig3a, use_container_width=True)
with col_b:
    fig3b = px.line(div_df, x="anio_dt", y="herfindahl", markers=True, title="Concentración (Herfindahl) • transitorios — Regional")
    fig3b.update_layout(xaxis_title="Año", yaxis_title="Índice de Herfindahl")
    fig3b.update_xaxes(tickformat="%Y")
    st.plotly_chart(fig3b, use_container_width=True)
with st.expander("Descripción y consideraciones"):
    st.write(
        "- Calculado solo con transitorios para evitar sesgo por permanentes.\n"
        "- Shannon alto = portafolio más balanceado; Herfindahl alto = concentración.\n"
        "- Periodo: según slider. Agregación regional."
    )

# 4) Flexibilidad agrícola — Regional
st.subheader("4) Flexibilidad agrícola — Regional")
flex_df = compute_flex_regional(df_anual_f)
flex_df["anio_dt"] = to_year_datetime(flex_df["anio"])
fig4 = px.line(flex_df, x="anio_dt", y="flexibilidad", markers=True, title="Flexibilidad = Transitorios / (Transitorios + Permanentes) — Regional")
fig4.update_layout(xaxis_title="Año", yaxis_title="Índice de flexibilidad (0–1)")
fig4.update_xaxes(tickformat="%Y")
st.plotly_chart(fig4, use_container_width=True)
with st.expander("Descripción y consideraciones"):
    st.write(
        "- Mide el margen de maniobra (proporción transitoria sobre total cultivado). Periodo según slider.\n"
        "- Si se usa solo superficie **cultivada reportada**, puede sobreestimar la flexibilidad frente a la superficie agrícola total oficial."
    )

# 5) Semáforo de riesgo — Regional
st.subheader("5) Semáforo de riesgo de sobreoferta (Δ área y Δ precio) — Regional")
risk_df = risk_semaforo(df_anual_f, area_thr=15.0, price_thr=-10.0)
if risk_df.empty:
    st.info("No fue posible calcular el semáforo: revisa columnas anuales requeridas.")
else:
    last_year = int(risk_df["anio"].max())
    cols_show = ["anio","cultivo_std","delta_area_pct","delta_precio_pct","score_riesgo","riesgo"]
    st.write(f"**Tabla (año más reciente: {last_year})**")
    st.dataframe(risk_df[risk_df["anio"] == last_year][cols_show].sort_values(["score_riesgo","cultivo_std"], ascending=[False, True]), use_container_width=True)
    piv = (risk_df.pivot_table(index="cultivo_std", columns="anio", values="score_riesgo", aggfunc="max")
                   .fillna(0).sort_index())
    fig5 = px.imshow(piv, aspect="auto", title="Riesgo por cultivo × año (0=bajo, 2=alto)")
    st.plotly_chart(fig5, use_container_width=True)
with st.expander("Descripción y consideraciones"):
    st.write(
        "- Reglas: +1 si el área > **+15%** vs promedio 3 años previos; +1 si el precio < **−10%** vs promedio 3 años previos.\n"
        "- Si faltan precios, solo actúa la señal de área. Series cortas pueden dar señales inestables."
    )

# 6) Calendario amigable — Bandas de meses (top cultivos transitorios)
st.subheader("6) Calendario de siembra y cosecha (amigable) • transitorios")
need_clean = {"anio","mes","cultivo_std","tipo_cultivo","siembra","cosecha"}
if not need_clean.issubset(df_clean_f.columns):
    st.info("No hay columnas suficientes en dataset mensual para calendario.")
else:
    bands_df, sow_w, har_w = simple_calendar(df_clean_f, top_n=12)
    if bands_df is None:
        st.info("No hay transitorios en el periodo.")
    else:
        st.write("**Resumen por cultivo (bandas de meses 'activos')**")
        st.dataframe(bands_df.rename(columns={"cultivo_std":"Cultivo","siembra_meses":"Siembra","cosecha_meses":"Cosecha"}),
                     use_container_width=True)
        # barras horizontales compactas (12 segmentos)
        def _bars_from_wide(wide, title):
            fig = go.Figure()
            months = ["E","F","M","A","M","J","J","A","S","O","N","D"]
            for i, (cultivo,row) in enumerate(wide.iterrows()):
                cum = 0.0
                # cada mes como barra apilada en horizontal para cada cultivo
                for m in range(1,13):
                    fig.add_trace(go.Bar(
                        x=[row[m]], y=[cultivo], orientation='h', name=months[m-1],
                        showlegend=(i==0), hovertemplate=f"{cultivo} · {months[m-1]}: %{ 'x' }<extra></extra>"
                    ))
            fig.update_layout(barmode="stack", title=title, xaxis_title="Intensidad relativa (0–1)", yaxis_title="Cultivo transitorio")
            return fig

        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(_bars_from_wide(sow_w, "Siembra — Intensidad relativa por mes"), use_container_width=True)
        with col2:
            st.plotly_chart(_bars_from_wide(har_w, "Cosecha — Intensidad relativa por mes"), use_container_width=True)

with st.expander("Descripción y consideraciones (calendario)"):
    st.write(
        "- En vez de mapas de calor, se muestran **bandas** y **barras** por mes para facilitar lectura.\n"
        "- Intensidad normalizada 0–1 por cultivo (comparabilidad). Los rangos de meses activos se derivan con un umbral (0.4), ajustable."
    )

# 7) Recomendaciones prácticas
st.subheader("7) Recomendaciones prácticas")
notes = []
# riesgo
if not risk_df.empty:
    last = risk_df[risk_df["anio"] == risk_df["anio"].max()]
    alto = last[last["riesgo"] == "Alto"]["cultivo_std"].unique().tolist()
    medio = last[last["riesgo"] == "Medio"]["cultivo_std"].unique().tolist()
    if alto:
        notes.append("🔴 Riesgo **alto** de sobreoferta en: " + ", ".join(sorted(alto)))
    if medio:
        notes.append("🟡 Riesgo **medio** en: " + ", ".join(sorted(medio)))
# flex regional último año
fx = compute_flex_regional(df_anual_f)
if not fx.empty and pd.notnull(fx.iloc[-1]["flexibilidad"]):
    notes.append(f"🧭 Flexibilidad regional (último año): **{fx.iloc[-1]['flexibilidad']:.2f}** (0–1).")
# diversificación regional último año
if not div_df.empty if 'div_df' in locals() else False:
    notes.append(f"📈 Diversificación regional (Shannon) transitorios (último año): **{div_df.iloc[-1]['shannon']:.2f}**.")
if notes:
    for n in notes:
        st.write("• " + n)
else:
    st.info("Cargue datos válidos para generar recomendaciones.")

st.caption("v5 • Diseño amigable: mapa con KPIs en tooltip, calendarios simplificados, y notas metodológicas claras.")
