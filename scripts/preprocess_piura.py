# preprocess_piura.py
# ------------------------------------------------------------
# Preprocesamiento dataset Piura (GRP) + diccionario TIPO_CULTIVO
# Entradas:
#   - ../../Formato_dataset_productos.csv          (crudo GRP)
#   - ../../diccionario_cultivos_piura.csv         (CULTIVO, COD_CULTIVO, TIPO_CULTIVO)
# Salidas (se crean en ../data_proc/):
#   - dataset_piura_clean.csv  (mensual, normalizado)
#   - dataset_piura_anual.csv  (anio-cultivo, precios anualizados)
#   - preprocess_report.md     (reporte de validaciones)
# ------------------------------------------------------------

from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np

# --- rutas y setup ---
RAW_CSV  = Path("../../Formato_dataset_productos.csv")     # crudo GRP
DICT_CSV = Path("../../diccionario_cultivos_piura.csv")    # diccionario con TIPO_CULTIVO (sin Desconocidos)
OUT_DIR  = Path("../data_proc")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_CLEAN = OUT_DIR / "dataset_piura_clean.csv"
OUT_ANUAL = OUT_DIR / "dataset_piura_anual.csv"
OUT_REP   = OUT_DIR / "preprocess_report.md"

# --- utilidades ---
def read_any_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"No existe: {path.resolve()}")
    try:
        return pd.read_csv(path)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin-1")

def norm_upper(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
        .str.upper()
    )

def main():
    t0 = datetime.now()
    report = []

    # 1) Leer insumos
    df_raw  = read_any_csv(RAW_CSV)
    df_dict = read_any_csv(DICT_CSV)

    # 2) Normalizar encabezados
    df_raw.columns  = [c.strip().upper() for c in df_raw.columns]
    df_dict.columns = [c.strip().upper() for c in df_dict.columns]

    # 3) Validaciones mínimas
    req_cols_raw = {
        "ANO","MES","DEPARTAMENTO","PROVINCIA","DISTRITO","UBIGEO",
        "CULTIVO","SIEMBRA","COSECHA","PRODUCCION","PRECIO_CHACRA"
    }
    miss_raw = req_cols_raw - set(df_raw.columns)
    if miss_raw:
        raise ValueError(f"Faltan columnas en RAW: {sorted(miss_raw)}")

    req_cols_dict = {"CULTIVO","TIPO_CULTIVO"}
    miss_dict = req_cols_dict - set(df_dict.columns)
    if miss_dict:
        raise ValueError(f"Faltan columnas en DICCIONARIO: {sorted(miss_dict)}")

    report.append("✅ Columnas mínimas presentes en ambos archivos.")

    # 4) Normalizar nombres de cultivo y hacer merge
    df_raw["CULTIVO_NORM"]  = norm_upper(df_raw["CULTIVO"])
    df_dict["CULTIVO_NORM"] = norm_upper(df_dict["CULTIVO"])

    m = pd.merge(
        df_raw,
        df_dict[["CULTIVO_NORM","TIPO_CULTIVO"]],
        on="CULTIVO_NORM",
        how="left"
    )

    # 5) Reglas de inferencia por si algo no matchea exacto (variante ortográfica local)
    no_match = m["TIPO_CULTIVO"].isna().sum()
    if no_match > 0:
        rules = {
            # transitorios genéricos
            "ALGODON":"Transitorio","FRIJOL":"Transitorio","HABA":"Transitorio","ARVEJA":"Transitorio",
            "MAIZ":"Transitorio","MANI":"Transitorio","TRIGO":"Transitorio","TOMATE":"Transitorio",
            "ZAPALLO":"Transitorio","PEPINILLO":"Transitorio","PIMIENTO":"Transitorio","PAPRIKA":"Transitorio",
            "SANDIA":"Transitorio","MELON":"Transitorio","RABANO":"Transitorio","ZANAHORIA":"Transitorio",
            "QUINUA":"Transitorio","SOYA":"Transitorio","SORGO":"Transitorio","OCA":"Transitorio",
            "OLLUCO":"Transitorio","PAPA":"Transitorio","YUCA":"Transitorio","LECHUGA":"Transitorio",
            "COLIFLOR":"Transitorio","GIRASOL":"Transitorio","ACHITA":"Transitorio","AJI":"Transitorio",
            "AJO":"Transitorio","AVENA":"Transitorio",
            # permanentes genéricos
            "MANGO":"Permanente","PALTO":"Permanente","UVA":"Permanente","VID":"Permanente",
            "LIMON":"Permanente","NARAN":"Permanente","MANDAR":"Permanente","LUCUM":"Permanente",
            "GUAYAB":"Permanente","GRANADILL":"Permanente","CAFE":"Permanente","CACAO":"Permanente",
            "CAÑA":"Permanente","CANA":"Permanente","BANAN":"Permanente","PLATAN":"Permanente",
            "ALFALFA":"Permanente","GUANABAN":"Permanente","CHIRIMOY":"Permanente","CIRUEL":"Permanente",
            "HIGUER":"Permanente","COCOTER":"Permanente","MARACUY":"Permanente","PAPAYA":"Permanente",
            "PIÑA":"Permanente","PINA":"Permanente","PITAHAY":"Permanente","TUNA":"Permanente",
            "TAMARIND":"Permanente","PACAE":"Permanente","PASTO":"Permanente","MARIGOLD":"Permanente",
            "TUMBO":"Permanente","TORONJ":"Permanente","NISPERO":"Permanente","MAMEY":"Permanente",
        }
        def infer_tipo(x: str):
            if pd.isna(x):
                return np.nan
            for key, val in rules.items():
                if key in x:
                    return val
            return np.nan

        m["TIPO_INFERIDO"] = m["TIPO_CULTIVO"]
        mask = m["TIPO_INFERIDO"].isna()
        m.loc[mask, "TIPO_INFERIDO"] = m.loc[mask, "CULTIVO_NORM"].apply(infer_tipo)
        inferidos = m["TIPO_INFERIDO"].notna().sum() - (len(m) - no_match)
        restantes = m["TIPO_INFERIDO"].isna().sum()
        # criterio por defecto (conservador para campañas): transitorio
        m["TIPO_CULTIVO"] = m["TIPO_INFERIDO"].fillna("Transitorio")
        m.drop(columns=["TIPO_INFERIDO"], inplace=True)
        report.append(f"🔧 Filas sin match exacto: {no_match:,}. Inferidos por reglas: {inferidos:,}. Restantes asumidos Transitorio: {restantes:,}.")
    else:
        report.append("✅ Todas las filas matchearon con el diccionario.")

    # 6) Coherencia de unidades + derivadas
    # Asunción base:
    # - SIEMBRA/COSECHA en hectáreas (ha)
    # - PRODUCCION en toneladas métricas (t)
    # - PRECIO_CHACRA en S/ kg
    for col in ["SIEMBRA","COSECHA","PRODUCCION","PRECIO_CHACRA","UBIGEO","ANO","MES"]:
        if col in m.columns:
            m[col] = pd.to_numeric(m[col], errors="coerce")

    # rendimiento (t/ha)
    m["RENDIMIENTO_T_HA"] = np.where(
        (m["COSECHA"] > 0) & np.isfinite(m["COSECHA"]),
        m["PRODUCCION"] / m["COSECHA"],
        np.nan
    )
    # valor de producción (S/)
    m["VALOR_PROD_S"] = m["PRODUCCION"] * 1000.0 * m["PRECIO_CHACRA"]

    # 7) Estándares de texto y campaña
    m["ANIO"]        = m["ANO"].astype("Int64")
    m["MES"]         = m["MES"].astype("Int64")
    m["CULTIVO_STD"] = m["CULTIVO"].astype(str).str.strip().str.title()
    m["PROVINCIA"]   = m["PROVINCIA"].astype(str).str.strip().str.title()
    m["DISTRITO"]    = m["DISTRITO"].astype(str).str.strip().str.title()
    m["CAMPANA"]     = np.where(m["MES"].isin([7,8,9,10,11,12]), "Grande",
                         np.where(m["MES"].isin([1,2,3,4,5,6]), "Chica", None))

    # 8) Exportar mensual limpio
    cols_out = [
        "ANIO","MES","CAMPANA","DEPARTAMENTO","PROVINCIA","DISTRITO","UBIGEO",
        "CULTIVO_STD","CULTIVO_NORM","TIPO_CULTIVO",
        "SIEMBRA","COSECHA","PRODUCCION","RENDIMIENTO_T_HA",
        "PRECIO_CHACRA","VALOR_PROD_S"
    ]
    cols_out = [c for c in cols_out if c in m.columns]
    clean = m[cols_out].copy().sort_values(["ANIO","MES","CULTIVO_STD"])
    clean.to_csv(OUT_CLEAN, index=False)

    # 9) Agregación anual (anio–cultivo–tipo) + precio anualizado
    anual = clean.groupby(["ANIO","CULTIVO_STD","TIPO_CULTIVO"], as_index=False).agg(
        area_sembrada_ha=("SIEMBRA","sum"),
        area_cosechada_ha=("COSECHA","sum"),
        produccion_t=("PRODUCCION","sum"),
        precio_prom_s_kg=("PRECIO_CHACRA","mean"),   # promedio simple (puedes ponderar por volumen si quieres)
        valor_prod_s=("VALOR_PROD_S","sum"),
        rendimiento_t_ha=("RENDIMIENTO_T_HA","mean")
    ).sort_values(["ANIO","CULTIVO_STD"])
    anual.to_csv(OUT_ANUAL, index=False)

    # 10) Validaciones básicas (banderas)
    issues = []
    if "SIEMBRA" in clean.columns and "COSECHA" in clean.columns:
        bad_area = (clean["SIEMBRA"] < 0).sum() + (clean["COSECHA"] < 0).sum()
        if bad_area > 0:
            issues.append(f"- {bad_area} filas con área negativa.")
    if "PRODUCCION" in clean.columns:
        bad_prod = (clean["PRODUCCION"] < 0).sum()
        if bad_prod > 0:
            issues.append(f"- {bad_prod} filas con producción negativa.")
    if "RENDIMIENTO_T_HA" in clean.columns:
        hi_r = (clean["RENDIMIENTO_T_HA"] > 60).sum()  # umbral genérico para outliers
        if hi_r > 0:
            issues.append(f"- {hi_r} filas con rendimiento > 60 t/ha (posible outlier).")

    # 11) Reporte
    report.append(f"Filas RAW:   {len(df_raw):,}")
    report.append(f"Filas CLEAN: {len(clean):,}")
    report.append(f"Filas ANUAL: {len(anual):,}")
    if issues:
        report.append("⚠️ Posibles inconsistencias:\n" + "\n".join(issues))
    else:
        report.append("✅ Sin inconsistencias evidentes con las reglas básicas.")
    report.append(f"Salvados:\n- {OUT_CLEAN}\n- {OUT_ANUAL}\n- {OUT_REP}")

    with OUT_REP.open("w", encoding="utf-8") as f:
        f.write("# Preprocess Report\n\n")
        for line in report:
            f.write(f"{line}\n")

    print("\n".join(report))
    print(f"Duración: {datetime.now() - t0}")

if __name__ == "__main__":
    main()
