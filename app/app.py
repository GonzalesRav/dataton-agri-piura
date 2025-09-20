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
GJ_DIST   = GEO_DIR  / "distritos_piura.geojson"   # props: UBIGEO, NOMBDIST

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

def compute_flex_regional(df_anual):
    g = df_anual.groupby(["anio","tipo_cultivo"], as_index=False)["area_sembrada_ha"].sum()
    piv = g.pivot_table(index="anio", columns="tipo_cultivo", values="area_sembrada_ha", aggfunc="sum").fillna(0.0)
    denom = (piv.get("Transitorio", 0.0) + piv.get("Permanente", 0.0))
    flex = np.where(denom > 0, piv.get("Transitorio", 0.0) / denom, np.nan)
    out = pd.DataFrame({"anio": piv.index, "flexibilidad": flex})
    return out.sort_values("anio")

def kpis_by_ubigeo(df_clean_period: pd.DataFrame) -> pd.DataFrame:
    """ KPIs por distrito (UBIGEO) en el periodo: flexibilidad y Shannon(transitorios), usando SIEMBRA. """
    d = df_clean_period.copy()
    needed = {"ubigeo","tipo_cultivo","cultivo_std","siembra","distrito"}
    if not needed.issubset(d.columns):
        return pd.DataFrame()
    d["ubigeo"] = d["ubigeo"].astype(str).str.strip().str.zfill(6)

    # Flexibilidad
    g = d.groupby(["ubigeo","tipo_cultivo"], as_index=False)["siembra"].sum()
    piv = g.pivot_table(index="ubigeo", columns="tipo_cultivo", values="siembra", aggfunc="sum").fillna(0)
    denom = (piv.get("Transitorio",0)+piv.get("Permanente",0))
    flex = np.where(denom>0, piv.get("Transitorio",0)/denom, np.nan)
    df_flex = pd.DataFrame({"ubigeo":piv.index, "flexibilidad":flex})

    # Diversificación (solo transitorios)
    div_rows = []
    d_trans = d[d["tipo_cultivo"].eq("Transitorio")]
    for ubi, grp in d_trans.groupby("ubigeo"):
        sums = grp.groupby("cultivo_std", as_index=False)["siembra"].sum()
        sh = shannon_index_df(sums, "siembra")
        div_rows.append({"ubigeo": ubi, "shannon_trans": sh})
    df_div = pd.DataFrame(div_rows)

    # Nombre distrito (más frecuente en el periodo)
    name_df = d.groupby("ubigeo")["distrito"].agg(lambda s: s.mode().iat[0] if not s.mode().empty else s.iloc[0]).reset_index()

    out = df_flex.merge(df_div, on="ubigeo", how="outer").merge(name_df, on="ubigeo", how="left")
    return out

# ---------------------------
# Cargar datos
# ---------------------------
df_anual = _norm_cols(_read_csv(CSV_ANUAL))
df_clean = _norm_cols(_read_csv(CSV_CLEAN))

# alias comunes por si acaso
aliases = {"año":"anio","cultivo":"cultivo_std","tipo":"tipo_cultivo",
           "produccion":"produccion_t","precio_promedio_s_kg":"precio_prom_s_kg"}
for old,new in aliases.items():
    if old in df_anual.columns and new not in df_anual.columns:
        df_anual[new] = df_anual[old]

# ---------------------------
# App
# ---------------------------
st.set_page_config(page_title="AgriPiura • Datatón 2025", layout="wide")
st.markdown("# 🌾 AgriPiura — Datos abiertos para planificar la siembra")
st.caption("Datatón 2025 • ODS 2 Hambre Cero + ODS 8 Trabajo decente y crecimiento económico • Reproducible con datos abiertos")

st.caption("**Gráficos:** (1) Mapas • (2) Top-10 transitorios • (3) Tendencia área/precio • (4) Dona de flexibilidad")

# Sidebar: filtros
with st.sidebar:
    st.header("🎛️ Filtros")

    st.subheader("Periodo")
    years = sorted(df_anual["anio"].dropna().unique().tolist())
    y_min, y_max = years[0], years[-1]
    sel_years = st.select_slider("Rango de años", options=years, value=(y_min, y_max))
    st.caption("Aplica a: **Todos los gráficos**")

    # datasets filtrados por tiempo
    df_anual_f = df_anual[df_anual["anio"].between(sel_years[0], sel_years[1])].copy()
    df_clean_f = df_clean[df_clean["anio"].between(sel_years[0], sel_years[1])].copy()


    st.subheader("Tendencia")
    cultivos = sorted(df_anual["cultivo_std"].dropna().unique().tolist())
    cultivo_options = ["Todos"] + cultivos
    cultivo_sel = st.selectbox(
        "Elige un cultivo (o 'Todos')",
        options=cultivo_options,
        index=0,
        help="Para el gráfico de evolución anual; permite comparar área y precio"
    )
    st.caption("Aplica a: **Gráfico (3)**")

    # st.divider()

    st.subheader("Ámbito geográfico")
    distritos = sorted(df_clean["distrito"].dropna().unique().tolist())  # usar df_clean (trae distrito)
    distrito_options = ["Todos"] + distritos
    distrito_sel = st.selectbox(
        "Distrito",
        options=distrito_options,
        index=0,
        help="Cambia el ámbito de la dona de flexibilidad"
    )
    st.caption("Aplica a: **Gráfico (4)**")

st.markdown("---")

# 1) Mapas distritales (dos indicadores)
st.subheader("1) Mapas distritales: Flexibilidad y Diversificación (periodo seleccionado)")

def build_map_figure(gj, df_map, metric_col, title_bar):
    import plotly.graph_objects as go
    df_yes = df_map[df_map[metric_col].notna()].copy()
    df_no  = df_map[df_map[metric_col].isna()].copy()

    fig = go.Figure()

    # Capa 1: SIN datos (gris claro)
    if not df_no.empty:
        fig.add_trace(go.Choropleth(
            geojson=gj,
            locations=df_no["ubigeo"],
            z=[0]*len(df_no),
            featureidkey="properties.UBIGEO",
            colorscale=[(0, "#e6e6e6"), (1, "#e6e6e6")],
            showscale=False,
            marker_line_width=0.2,
            hovertemplate="<b>%{customdata[0]}</b><br>Valor: sin datos<extra></extra>",
            customdata=np.stack([df_no["distrito"]], axis=-1),
            name="Sin datos"
        ))

    # Capa 2: CON datos (color continuo)
    if not df_yes.empty:
        cd = np.stack([df_yes["distrito"], df_yes[metric_col]], axis=-1)
        fig.add_trace(go.Choropleth(
            geojson=gj,
            locations=df_yes["ubigeo"],
            z=df_yes[metric_col],
            featureidkey="properties.UBIGEO",
            colorscale="Viridis",
            marker_line_width=0.2,
            colorbar_title=title_bar,
            hovertemplate="<b>%{customdata[0]}</b><br>" + title_bar + ": %{customdata[1]:.2f}<extra></extra>",
            customdata=cd,
            name="Con datos"
        ))

    fig.update_geos(fitbounds="locations", visible=False)
    fig.update_layout(margin=dict(l=0, r=0, t=30, b=0))
    return fig

try:
    with open(GJ_DIST, "r", encoding="utf-8") as f:
        gj = json.load(f)

    # Normalizar UBIGEO/NOMBDIST en el GeoJSON
    for feat in gj["features"]:
        props = feat.get("properties", {})
        props["UBIGEO"]   = str(props.get("UBIGEO","")).strip().zfill(6)
        props["NOMBDIST"] = str(props.get("NOMBDIST","")).strip()
        feat["properties"] = props

    # KPIs por UBIGEO en el periodo seleccionado
    kpis = kpis_by_ubigeo(df_clean_f)
    if not kpis.empty:
        kpis["ubigeo"] = kpis["ubigeo"].astype(str).str.zfill(6)

    # Base completa desde GeoJSON (para incluir SIN datos)
    df_geo = pd.DataFrame([
        {"ubigeo": feat["properties"]["UBIGEO"], "distrito_geo": feat["properties"]["NOMBDIST"]}
        for feat in gj["features"]
    ])
    df_map = df_geo.merge(kpis, on="ubigeo", how="left")
    df_map["distrito"] = df_map["distrito_geo"].fillna(df_map.get("distrito", df_map["distrito_geo"]))

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**Flexibilidad** · {sel_years[0]}–{sel_years[1]}")
        fig_flex = build_map_figure(gj, df_map, "flexibilidad", "Flexibilidad")
        st.plotly_chart(fig_flex, use_container_width=True)

    with col2:
        st.markdown(f"**Diversificación (Shannon, transitorios)** · {sel_years[0]}–{sel_years[1]}")
        fig_div = build_map_figure(gj, df_map, "shannon_trans", "Diversificación (Shannon)")
        st.plotly_chart(fig_div, use_container_width=True)

except FileNotFoundError:
    st.info("No se encontró el GeoJSON de distritos. Asegura la ruta: geo/distritos_piura.geojson")
except Exception as e:
    st.warning(f"Ocurrió un problema cargando los mapas: {e}")

with st.expander("Cómo leerlos / definiciones / relación y limitaciones"):
    st.write(
        "**Flexibilidad**: proporción de la superficie **transitoria** respecto al total (transitorios + permanentes). "
        "Se expresa entre 0 y 1. Valores altos ⇒ mayor margen de decisión cada campaña (más tierra que puede rotar de cultivo).\n\n"

        "**Diversificación (Shannon)**: calculada solo con cultivos transitorios. "
        "Mide cómo se reparte el área entre distintos cultivos. Valores altos ⇒ portafolio más balanceado y menor dependencia de un solo cultivo.\n\n"

        "**Relación entre ambos**: un distrito puede tener alta flexibilidad pero baja diversificación si casi toda su tierra transitoria se dedica a un solo cultivo (ejemplo: 80% arroz, 20% otros). "
        "Lo más resiliente es combinar **alta flexibilidad y alta diversificación**.\n\n"

        "**Interpretación práctica**: si un distrito aparece en color intenso en el mapa de flexibilidad pero con un valor bajo de diversificación, "
        "significa que sus agricultores tienen margen de maniobra para cambiar cada año, pero la mayoría apuesta al mismo cultivo ⇒ riesgo de sobreoferta o vulnerabilidad a plagas.\n\n"

        "**Consideraciones técnicas**:\n"
        "- Los valores se calculan sobre **área sembrada reportada** (ha). No se incluye superficie agrícola no cultivada.\n"
        "- En periodos cortos o con registros incompletos, los indicadores pueden variar mucho (valores inestables).\n"
        "- Distritos sin datos aparecen en **gris claro** y se interpretan como 'sin registro disponible', no como valor cero.\n"
    )



st.markdown("---")

# 2) Top-10 cultivos transitorios por área (periodo)
st.subheader("2) Top-10 cultivos transitorios por área (periodo) — Regional")
d2 = df_anual_f[df_anual_f["tipo_cultivo"].eq("Transitorio")].copy()
if d2.empty:
    st.info("No hay transitorios en el periodo seleccionado.")
else:
    top = (d2.groupby("cultivo_std", as_index=False)["area_sembrada_ha"].sum()
             .sort_values("area_sembrada_ha", ascending=False)
             .head(10))
    fig2 = px.bar(top, x="area_sembrada_ha", y="cultivo_std", orientation="h",
                  title=f"Top-10 por área sembrada • {sel_years[0]}–{sel_years[1]}")
    fig2.update_layout(xaxis_title="Área sembrada (ha)", yaxis_title="Cultivo transitorio")
    st.plotly_chart(fig2, use_container_width=True)
with st.expander("Cómo leerlo / limitaciones (Top-10)"):
    st.write(
    "- **Qué muestra**: ranking de los cultivos transitorios según su **superficie sembrada total (ha)** en el periodo seleccionado.\n"
    "- **Interpretación**: los primeros lugares concentran la mayor parte del área agrícola transitoria ⇒ son los que más pesan en la economía local y donde existe mayor riesgo de sobreoferta si todos apuestan al mismo cultivo.\n"
    "- **Ejemplo**: si el top muestra **Arroz, Maíz y Cebolla**, significa que la mayoría de hectáreas transitorias se concentran en ellos. Esto puede dar señales de alerta: precios inestables si el mapa indica baja diversificación.\n\n"
    "**Consideraciones técnicas y limitaciones**:\n"
    "- Se incluyen **solo cultivos transitorios** (los permanentes no entran porque no cambian campaña a campaña).\n"
    "- La base es el área **sembrada reportada**; no refleja superficie agrícola total ni tierras en descanso.\n"
    "- Distritos con registros incompletos pueden hacer que el ranking esté sesgado hacia zonas con mejor reporte.\n"
    "- No refleja rentabilidad ni precios: un cultivo con poca área podría ser más rentable que uno grande."
    )


st.markdown("---")

# 3) Tendencia de área y precio — Regional (un cultivo o 'Todos')
st.subheader("3) Tendencia de área sembrada y precio — Regional")

d_base = df_anual.copy()

if cultivo_sel != "Todos":
    d3 = d_base[d_base["cultivo_std"] == cultivo_sel].copy()
    titulo = f"Evolución anual — {cultivo_sel}"
else:
    d3 = d_base.copy()
    titulo = "Evolución anual — Todos los cultivos (agregado regional)"

if d3.empty:
    st.info("No hay datos para el filtro seleccionado.")
else:
    # Agregación anual
    agg = (
        d3.groupby("anio", as_index=False)
          .agg(
              area_sembrada_ha=("area_sembrada_ha", "sum"),
              produccion_t=("produccion_t", "sum"),
              precio_total_val=("precio_prom_s_kg", lambda s: (s * d3.loc[s.index, "produccion_t"]).sum(skipna=True)),
          )
    )
    agg["precio_prom_pond_s_kg"] = np.where(
        agg["produccion_t"] > 0,
        agg["precio_total_val"] / (agg["produccion_t"] * 1.0),
        np.nan
    )
    agg["anio_dt"] = to_year_datetime(agg["anio"])

    # Gráfico con dos ejes y hover unificado
    fig3 = go.Figure()

    # Área (eje izquierdo)
    fig3.add_trace(go.Scatter(
        x=agg["anio_dt"], y=agg["area_sembrada_ha"],
        name="Área sembrada (ha)", mode="lines+markers", yaxis="y1",
        hovertemplate="Área sembrada: %{y:,.0f} ha<extra></extra>"
    ))

    # Precio (eje derecho)
    fig3.add_trace(go.Scatter(
        x=agg["anio_dt"], y=agg["precio_prom_pond_s_kg"],
        name="Precio chacra (S/kg)", mode="lines+markers", yaxis="y2",
        line=dict(dash="dot"),
        hovertemplate="Precio chacra: S/ %{y:.2f} /kg<extra></extra>"
    ))

    fig3.update_layout(
        title=titulo,
        xaxis=dict(title="Año", tickformat="%Y"),
        yaxis=dict(title="Área sembrada (ha)", side="left"),
        yaxis2=dict(title="Precio chacra (S/kg)", overlaying="y", side="right"),
        legend=dict(orientation="h", y=-0.2),
        hovermode="x unified"  # ← muestra ambos valores en un solo recuadro por año
    )

    st.plotly_chart(fig3, use_container_width=True)

with st.expander("Cómo leerlo / limitaciones (área + precio)"):
    st.write(
        "- **Qué muestra**: la evolución anual del área sembrada (ha) y el precio chacra (S/ kg) del cultivo seleccionado "
        "(o promedio ponderado si se elige 'Todos').\n"
        "- **Interpretación clave**: si el área sembrada aumenta mientras el precio chacra cae, es una señal de "
        "posible **sobreoferta** en el mercado.\n"
        "- **Ejemplo**: si la curva de cebolla sube en hectáreas entre 2018 y 2019 pero el precio baja de 1.8 a 1.2 S/kg, "
        "indica que más superficie en ese cultivo redujo el precio recibido por agricultor.\n\n"
        "**Consideraciones técnicas y limitaciones**:\n"
        "- Los valores son **anuales y regionales**; pueden ocultar variaciones dentro del año o entre distritos.\n"
        "- El precio es un **promedio ponderado por producción** (cuando hay más de un registro); cultivos con muchos subtipos o distritos "
        "pueden mostrar precios distintos a lo percibido en una zona puntual.\n"
        "- El área mostrada es **reportada como sembrada**, no necesariamente cosechada.\n"
        "- En series cortas, un solo outlier (ej. caída de precio por fenómeno climático) puede distorsionar la tendencia."
    )


st.markdown("---")

# 4) Donut de flexibilidad (transitorios vs permanentes) — Regional o Distrital
st.subheader("4) Flexibilidad (transitorios vs permanentes)")

# Tomamos el dataset mensual filtrado por años (df_clean_f) para poder filtrar por distrito
df_scope = df_clean_f.copy()

ambito_titulo = "Regional (Piura)"
if distrito_sel != "Todos":
    df_scope = df_scope[df_scope["distrito"] == distrito_sel].copy()
    ambito_titulo = f"Distrito: {distrito_sel}"

g = df_scope.groupby("tipo_cultivo", as_index=False)["siembra"].sum()

if g.empty or g["siembra"].sum() == 0:
    st.info(f"No hay superficie registrada en el periodo seleccionado para {ambito_titulo}.")
else:
    vals = g.set_index("tipo_cultivo")["siembra"].to_dict()
    trans = float(vals.get("Transitorio", 0.0))
    perm  = float(vals.get("Permanente", 0.0))
    total = trans + perm
    flex_ratio = (trans/total) if total > 0 else np.nan

    labels = ["Transitorios", "Permanentes"]
    values = [trans, perm]

    fig4 = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.5)])
    fig4.update_layout(
        title=f"{ambito_titulo} • Flexibilidad = {flex_ratio:.2f} (Transitorios/Total) • {sel_years[0]}–{sel_years[1]}"
    )
    st.plotly_chart(fig4, use_container_width=True)

with st.expander("Cómo leerlo / limitaciones (flexibilidad)"):
    st.write(
        "- **Qué mide**: el margen de maniobra en la elección de cultivos de campaña. "
        "Se calcula como **Transitorios / (Transitorios + Permanentes)** usando el área sembrada (ha).\n"
        f"- **Ámbito**: este valor corresponde a **{ambito_titulo}**, de acuerdo con el filtro elegido en la barra lateral.\n"
        "- **Cómo leerlo**: valores cercanos a **1** significan que predomina la tierra transitoria (decisiones anuales más flexibles). "
        "Valores cercanos a **0** indican predominio de permanentes (decisiones rígidas a varios años).\n"
        "- **Ejemplo de interpretación**: si la dona muestra 0.75, implica que el 75% de la superficie cultivada en ese ámbito "
        "corresponde a cultivos transitorios (ej. arroz, maíz), mientras que solo 25% está en permanentes (ej. mango, palto). "
        "Ese distrito/región tiene margen alto para ajustar la siembra campaña a campaña.\n\n"
        "**Consideraciones técnicas y limitaciones**:\n"
        "- Se basa en **superficie sembrada reportada**, no en superficie agrícola total ni en tierras en descanso. "
        "Esto puede sobreestimar la flexibilidad real.\n"
        "- A nivel distrito, con pocos registros el índice puede ser extremo (0 o 1). "
        "Por ejemplo, si en el periodo no se reportaron permanentes, la flexibilidad resulta 1.00 automáticamente.\n"
        "- Valores dependen de la **calidad del reporte**; distritos con datos incompletos pueden dar señales engañosas."
    )



st.caption(
    "Construido con Streamlit • Código abierto • Reproducible • "
    "Desarrollado por Johana Gonzales 👩‍💻 [Conecta en LinkedIn](https://www.linkedin.com/in/gonzalesrav/) | [GitHub](https://github.com/GonzalesRav)"
)

