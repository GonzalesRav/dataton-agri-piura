
# app_streamlit_piura_v4.py
# ------------------------------------------------------------
# Dashboard Piura (v4): indicadores accionables para elección de cultivos
# - Lee datos desde rutas internas (sin uploaders)
# - Etiquetas y descripciones en castellano
# - Series con eje temporal formateado
# - Semáforo de riesgo robusto (manejo de faltantes y bordes)
# ------------------------------------------------------------

from pathlib import Path
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
GJ_DIST   = GEO_DIR  / "distritos_piura.geojson"  # (no utilizado en v4; dejamos la ruta preparada)

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
    """Convierte un entero de año en datetime (1 de enero del año) para formatear ejes."""
    return pd.to_datetime(series.astype("Int64").astype(str) + "-01-01", errors="coerce")

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
    """Flexibilidad = área transitorios / (transitorios + permanentes), a nivel total regional por año."""
    g = df_anual.groupby(["anio","tipo_cultivo"], as_index=False)["area_sembrada_ha"].sum()
    piv = g.pivot_table(index="anio", columns="tipo_cultivo", values="area_sembrada_ha", aggfunc="sum").fillna(0.0)
    denom = (piv.get("Transitorio", 0.0) + piv.get("Permanente", 0.0))
    flex = np.where(denom > 0, piv.get("Transitorio", 0.0) / denom, np.nan)
    out = pd.DataFrame({"anio": piv.index, "flexibilidad": flex})
    return out.sort_values("anio")

def risk_semaforo(df_anual, area_thr=15.0, price_thr=-10.0):
    """
    Señales (por cultivo y año, total regional):
      - Δ área (%): (área - prom_3_prev) / prom_3_prev * 100 > +15  -> +1
      - Δ precio (%): (precio - prom_3_prev) / prom_3_prev * 100 < -10 -> +1
    Score: 0=bajo, 1=medio, 2=alto
    """
    df = df_anual.copy().sort_values(["cultivo_std","anio"])

    # Promedio móvil de 3 años previos por cultivo (transform asegura alineación)
    df["area_prom_3_prev"] = (
        df.groupby("cultivo_std", group_keys=False)["area_sembrada_ha"]
          .apply(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
    )
    # evitemos divisiones por cero/NaN
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

    # Score
    df["score_riesgo"] = 0
    df.loc[df["delta_area_pct"] > area_thr, "score_riesgo"] += 1
    df.loc[df["delta_precio_pct"] < price_thr, "score_riesgo"] += 1

    df["riesgo"] = df["score_riesgo"].map({2:"Alto",1:"Medio",0:"Bajo"}).fillna("Bajo")
    return df

# ---------------------------
# Cargar datos
# ---------------------------
df_anual = _norm_cols(_read_csv(CSV_ANUAL))
df_clean = _norm_cols(_read_csv(CSV_CLEAN))  # para calendario mensual

# Asegurar columnas esperadas (alias comunes)
aliases = {
    "año": "anio",
    "cultivo": "cultivo_std",
    "tipo": "tipo_cultivo",
    "produccion": "produccion_t",
    "precio_promedio_s_kg": "precio_prom_s_kg",
}
for old, new in aliases.items():
    if old in df_anual.columns and new not in df_anual.columns:
        df_anual[new] = df_anual[old]

# ---------------------------
# App
# ---------------------------
st.set_page_config(page_title="Piura • Elección de Cultivos (v4)", layout="wide")
st.title("🌾 Piura • Elección de Cultivos (v4)")
st.caption("Evolución, producción–superficie–precio, diversificación (transitorios), flexibilidad agrícola, riesgo de sobreoferta, calendario (transitorios) y recomendaciones.")

# Filtros principales
years = sorted(df_anual["anio"].dropna().unique().tolist()) if "anio" in df_anual.columns else []
if years:
    y_min, y_max = years[0], years[-1]
    sel_years = st.select_slider("Rango de años", options=years, value=(y_min, y_max))
    df_anual_f = df_anual[df_anual["anio"].between(sel_years[0], sel_years[1])].copy()
else:
    st.stop()

cultivos = sorted(df_anual["cultivo_std"].dropna().unique().tolist()) if "cultivo_std" in df_anual.columns else []
sel_cult_compare = st.multiselect("Comparar 2–3 cultivos (evolución de superficie)", options=cultivos, default=cultivos[:3] if len(cultivos)>=3 else cultivos)

st.markdown("---")

# 1) Evolución de áreas sembradas (comparativa)
st.subheader("1) Evolución de áreas sembradas (comparativa)")
d1 = df_anual_f[df_anual_f["cultivo_std"].isin(sel_cult_compare)] if sel_cult_compare else df_anual_f.head(0)
if d1.empty:
    st.warning("Selecciona 2–3 cultivos para comparar.")
else:
    g1 = d1.groupby(["anio","cultivo_std"], as_index=False)["area_sembrada_ha"].sum()
    g1["anio_dt"] = to_year_datetime(g1["anio"])
    fig1 = px.line(g1, x="anio_dt", y="area_sembrada_ha", color="cultivo_std", markers=True,
                   title="Evolución del área sembrada (ha) por cultivo seleccionado")
    fig1.update_layout(xaxis_title="Año", yaxis_title="Área sembrada (ha)")
    fig1.update_xaxes(tickformat="%Y")  # formateo anual limpio
    st.plotly_chart(fig1, use_container_width=True)
    with st.expander("Descripción y consideraciones"):
        st.write(
            "- Serie anual a nivel regional (total Piura). Se grafica la superficie sembrada (hectáreas) por cultivo.\n"
            "- Tendencias de alza sostenida sin respaldo de precios pueden anticipar sobreoferta.\n"
            "- Limitación: al ser agregación regional, oculta heterogeneidades intra-distritales."
        )

# 2) Producción vs superficie vs precio (burbuja)
st.subheader("2) Producción vs superficie vs precio (cultivo–año)")
d2 = df_anual_f.copy()
if "precio_prom_s_kg" not in d2.columns:
    d2["precio_prom_s_kg"] = np.nan
d2["valor_produccion_aprox_s"] = d2["produccion_t"] * 1000.0 * d2["precio_prom_s_kg"]
fig2 = px.scatter(
    d2, x="area_sembrada_ha", y="produccion_t",
    size="valor_produccion_aprox_s", color="precio_prom_s_kg",
    hover_data=["anio","cultivo_std"],
    title="Relación área–producción–precio (tamaño ≈ valor; color = precio chacra promedio S/ kg)"
)
fig2.update_layout(xaxis_title="Área sembrada (ha)", yaxis_title="Producción (t)")
st.plotly_chart(fig2, use_container_width=True)
with st.expander("Descripción y consideraciones"):
    st.write(
        "- Cada punto representa un par cultivo–año. Ejes: área (X) y producción (Y). Tamaño ≈ valor económico; color = precio chacra promedio (S/ kg).\n"
        "- Si no hay precios disponibles, el color no codifica información.\n"
        "- Limitación: precios pueden variar por mercado local; se presenta promedio anual regional."
    )

# 3) Índice de diversificación (solo transitorios, total regional)
st.subheader("3) Índice de diversificación (solo cultivos transitorios)")
d_trans = df_anual_f[df_anual_f["tipo_cultivo"].eq("Transitorio")].copy()
div_rows = []
for y, dfy in d_trans.groupby("anio"):
    div_rows.append({
        "anio": y,
        "shannon": shannon_index(dfy, "area_sembrada_ha"),
        "herfindahl": herfindahl_index(dfy, "area_sembrada_ha"),
    })
div_df = pd.DataFrame(div_rows).sort_values("anio")
div_df["anio_dt"] = to_year_datetime(div_df["anio"])

col_a, col_b = st.columns(2)
with col_a:
    fig3a = px.line(div_df, x="anio_dt", y="shannon", markers=True, title="Diversificación (Shannon) • transitorios")
    fig3a.update_layout(xaxis_title="Año", yaxis_title="Índice de Shannon")
    fig3a.update_xaxes(tickformat="%Y")
    st.plotly_chart(fig3a, use_container_width=True)
with col_b:
    fig3b = px.line(div_df, x="anio_dt", y="herfindahl", markers=True, title="Concentración (Herfindahl) • transitorios")
    fig3b.update_layout(xaxis_title="Año", yaxis_title="Índice de Herfindahl")
    fig3b.update_xaxes(tickformat="%Y")
    st.plotly_chart(fig3b, use_container_width=True)

with st.expander("Descripción y consideraciones"):
    st.write(
        "- Cálculo **solo con cultivos transitorios** para evitar sesgo por permanentes.\n"
        "- Shannon alto = portafolio más balanceado; Herfindahl alto = concentración.\n"
        "- Limitación: indicador regional; no captura diversidad a nivel de distrito."
    )

# 4) Índice de flexibilidad agrícola (total regional)
st.subheader("4) Índice de flexibilidad agrícola")
flex_df = compute_flex(df_anual_f)
flex_df["anio_dt"] = to_year_datetime(flex_df["anio"])
fig4 = px.line(flex_df, x="anio_dt", y="flexibilidad", markers=True, title="Flexibilidad = Transitorios / (Transitorios + Permanentes)")
fig4.update_layout(xaxis_title="Año", yaxis_title="Índice de flexibilidad (0–1)")
fig4.update_xaxes(tickformat="%Y")
st.plotly_chart(fig4, use_container_width=True)
with st.expander("Descripción y consideraciones"):
    st.write(
        "- Proporción de área transitoria respecto al total (transitorios + permanentes). Mide margen de maniobra anual.\n"
        "- Si se usa solo **superficie cultivada reportada**, puede sobreestimar la flexibilidad frente a la superficie agrícola total oficial.\n"
        "- Recomendación: documentar fuente de superficie y diferenciar cuando se disponga de CENAGRO/MIDAGRI."
    )

# 5) Semáforo de riesgo de sobreoferta (robusto)
st.subheader("5) Semáforo de riesgo de sobreoferta (Δ área y Δ precio)")
risk_df = risk_semaforo(df_anual_f, area_thr=15.0, price_thr=-10.0)
if risk_df.empty:
    st.info("No fue posible calcular el semáforo: verifica columnas requeridas (anio, cultivo_std, area_sembrada_ha).")
else:
    last_year = int(risk_df["anio"].max())
    cols_show = ["anio","cultivo_std","delta_area_pct","delta_precio_pct","score_riesgo","riesgo"]
    st.write(f"**Tabla (año más reciente: {last_year})**")
    st.dataframe(risk_df[risk_df["anio"] == last_year][cols_show].sort_values(["score_riesgo","cultivo_std"], ascending=[False, True]), use_container_width=True)

    # Heatmap cultivo × año (score)
    piv = (risk_df.pivot_table(index="cultivo_std", columns="anio", values="score_riesgo", aggfunc="max")
                   .fillna(0).sort_index())
    fig5 = px.imshow(piv, aspect="auto", title="Mapa de calor • riesgo por cultivo × año (0=bajo, 2=alto)")
    st.plotly_chart(fig5, use_container_width=True)

with st.expander("Descripción y consideraciones"):
    st.write(
        "- Reglas: +1 si el área actual supera en **>15%** el promedio de 3 años previos; +1 si el precio cae **>10%** vs el promedio de 3 años previos.\n"
        "- Score: 0 (bajo), 1 (medio), 2 (alto). Si no hay precios, solo opera la señal de área.\n"
        "- Limitaciones: series cortas por cultivo pueden generar señales inestables; se recomienda validar con conocimiento local."
    )

# 6) Calendario de siembra/cosecha (transitorios, mensual)
st.subheader("6) Calendario dinámico (siembra y cosecha) • transitorios")
# columnas esperadas: anio, mes, cultivo_std, tipo_cultivo, siembra, cosecha
needed = {"anio","mes","cultivo_std","tipo_cultivo","siembra","cosecha"}
if not needed.issubset(df_clean.columns):
    st.info("No se encontraron columnas suficientes en dataset mensual para el calendario.")
else:
    dcal = df_clean[df_clean["tipo_cultivo"].eq("Transitorio")].copy()
    if dcal.empty:
        st.info("No hay registros de cultivos transitorios en el dataset mensual.")
    else:
        sow = dcal.groupby(["cultivo_std","mes"], as_index=False)["siembra"].sum()
        har = dcal.groupby(["cultivo_std","mes"], as_index=False)["cosecha"].sum()
        # normalización 0–1 por cultivo para comparabilidad
        sow["int_sow"] = sow.groupby("cultivo_std")["siembra"].transform(lambda s: s / s.max() if s.max() not in [0, np.nan] else 0)
        har["int_har"] = har.groupby("cultivo_std")["cosecha"].transform(lambda s: s / s.max() if s.max() not in [0, np.nan] else 0)

        sow_piv = sow.pivot_table(index="cultivo_std", columns="mes", values="int_sow", aggfunc="mean").fillna(0).sort_index()
        har_piv = har.pivot_table(index="cultivo_std", columns="mes", values="int_har", aggfunc="mean").fillna(0).sort_index()

        col1, col2 = st.columns(2)
        with col1:
            fig6a = px.imshow(sow_piv, aspect="auto", title="Siembra (intensidad relativa por mes)")
            fig6a.update_xaxes(title_text="Mes", tickmode="array", tickvals=list(range(1,13)), ticktext=["E","F","M","A","M","J","J","A","S","O","N","D"])
            fig6a.update_yaxes(title_text="Cultivo transitorio")
            st.plotly_chart(fig6a, use_container_width=True)
        with col2:
            fig6b = px.imshow(har_piv, aspect="auto", title="Cosecha (intensidad relativa por mes)")
            fig6b.update_xaxes(title_text="Mes", tickmode="array", tickvals=list(range(1,13)), ticktext=["E","F","M","A","M","J","J","A","S","O","N","D"])
            fig6b.update_yaxes(title_text="Cultivo transitorio")
            st.plotly_chart(fig6b, use_container_width=True)

with st.expander("Descripción y consideraciones"):
    st.write(
        "- Solo cultivos **transitorios**. Intensidad mensual normalizada (0–1) por cultivo para comparar estacionalidad.\n"
        "- El patrón es relativo (no volumen absoluto). Útil para coordinar ventanas de siembra/cosecha y evitar picos simultáneos."
    )

# 7) Recomendaciones prácticas (beta)
st.subheader("7) Recomendaciones prácticas (beta)")
notes = []
# riesgo
risk_last_year = risk_df["anio"].max() if not risk_df.empty else None
if risk_last_year is not None:
    last = risk_df[risk_df["anio"] == risk_last_year]
    alto = last[last["riesgo"] == "Alto"]["cultivo_std"].unique().tolist()
    medio = last[last["riesgo"] == "Medio"]["cultivo_std"].unique().tolist()
    if alto:
        notes.append("🔴 Riesgo **alto** de sobreoferta en: " + ", ".join(sorted(alto)))
    if medio:
        notes.append("🟡 Riesgo **medio** en: " + ", ".join(sorted(medio)))
# flex
if not compute_flex(df_anual_f).empty:
    fx_last = compute_flex(df_anual_f).iloc[-1]["flexibilidad"]
    if pd.notnull(fx_last):
        notes.append(f"🧭 Flexibilidad agrícola (último año): **{fx_last:.2f}** (0–1). Valores bajos implican poca capacidad de ajuste.")
# diversificación
if not div_df.empty if 'div_df' in locals() else False:
    sh_last = div_df.iloc[-1]["shannon"]
    if pd.notnull(sh_last):
        notes.append(f"📈 Diversificación (Shannon) transitorios (último año): **{sh_last:.2f}**. Más alto = portafolio más balanceado.")

if notes:
    for n in notes:
        st.write("• " + n)
else:
    st.info("Cargue datos válidos para generar recomendaciones.")

st.caption("v4 • Transparencia: cada gráfico incluye notas metodológicas y limitaciones para una lectura crítica.")
