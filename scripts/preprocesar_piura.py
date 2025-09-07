# scripts/preprocesar_piura.py
# -*- coding: utf-8 -*-
from pathlib import Path
import numpy as np
import pandas as pd

# ---------- Config ----------
RAW_CSV  = Path("../../Formato_dataset_productos.csv")  # ajusta si tu ruta cambia
OUT_DIR  = Path("../data_proc")
OUT_BASE = OUT_DIR / "campania_agricola_piura_proc.csv"
OUT_TRI  = OUT_DIR / "agregados_trimestrales.csv"
OUT_ANU  = OUT_DIR / "agregados_anuales.csv"
OUT_COV  = OUT_DIR / "cobertura_precio_distrito_anio.csv"
OUT_QLT  = OUT_DIR / "quality_summary.csv"

IMPORTANT_NUMS = ["SIEMBRA","COSECHA","PRODUCCION","VERDE_ACTUAL","PRECIO_CHACRA"]

# ---------- Utils ----------
def _ensure_outdir():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

def _report_nulls_and_zeros(df: pd.DataFrame, cols):
    print("\n== Nulos y ceros por columna ==")
    rows = []
    n_total = len(df)
    for c in cols:
        col_num = pd.to_numeric(df[c], errors="coerce")
        n_null  = col_num.isna().sum()
        n_zero  = (col_num == 0).sum()
        rows.append({
            "columna": c,
            "n_total": n_total,
            "n_nulos": n_null,
            "pct_nulos": n_null / n_total if n_total else np.nan,
            "n_ceros": n_zero,
            "pct_ceros": n_zero / n_total if n_total else np.nan
        })
        print(f"{c:14s} | nulos: {n_null:6d} | ceros: {n_zero:6d}")
    return pd.DataFrame(rows)

# ---------- Carga y limpieza ----------
def load_raw(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"No se encontró el CSV crudo en: {path.resolve()}")
    df = pd.read_csv(path, low_memory=False, encoding="utf-8")
    df.columns = [c.strip().upper() for c in df.columns]
    return df

def clean_and_derive(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()

    # UBIGEO en 6 dígitos
    if "UBIGEO" in d.columns:
        d["UBIGEO"] = d["UBIGEO"].astype(str).str.zfill(6)

    # Fechas
    if "FECHA_MUESTRA" in d.columns:
        d["FECHA_MUESTRA"] = pd.to_datetime(d["FECHA_MUESTRA"].astype(str), format="%Y%m%d", errors="coerce")

    if "MES" in d.columns:
        d["MES_STR"] = d["MES"].astype(str)
        d["ANIO"]    = pd.to_numeric(d["MES_STR"].str[:4], errors="coerce").astype("Int64")
        d["MES_NUM"] = pd.to_numeric(d["MES_STR"].str[4:6], errors="coerce").astype("Int64")
        d["FECHA_YYYYMM"] = pd.to_datetime(
            d["ANIO"].astype(str) + d["MES_NUM"].astype(str).str.zfill(2),
            format="%Y%m", errors="coerce"
        )
        d["TRIM"] = d["FECHA_YYYYMM"].dt.quarter

    # Numéricos clave
    for c in IMPORTANT_NUMS:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")

    # *** Cambio 1: PRECIO_CHACRA = 0 -> NaN (no hay precio reportado) ***
    if "PRECIO_CHACRA" in d.columns:
        d["PRECIO_CHACRA"] = d["PRECIO_CHACRA"].replace(0, np.nan)

    # Producción en kg
    if "PRODUCCION" in d.columns:
        d["PRODUCCION_KG"] = d["PRODUCCION"] * 1000

    # Rendimiento mensual (t/ha)
    if {"PRODUCCION","COSECHA"}.issubset(d.columns):
        d["REND_MES_THA"] = np.where(d["COSECHA"] > 0, d["PRODUCCION"]/d["COSECHA"], np.nan)

    # Periodos string útiles
    if "ANIO" in d.columns and "TRIM" in d.columns:
        d["PERIODO_TRIM"]  = d["ANIO"].astype(str) + "-T" + d["TRIM"].astype(str)
    if "ANIO" in d.columns:
        d["PERIODO_ANUAL"] = d["ANIO"].astype(str)

    return d

# ---------- Métricas de calidad ----------
def quality_checks_and_table(d: pd.DataFrame) -> pd.DataFrame:
    print("\n== Checks básicos ==")
    if "FECHA_YYYYMM" in d.columns:
        pct_fecha = d["FECHA_YYYYMM"].notna().mean()
        print(f"% filas con FECHA_YYYYMM válida: {pct_fecha:.1%}")

    if "UBIGEO" in d.columns:
        assert d["UBIGEO"].str.len().eq(6).all(), "UBIGEO no tiene 6 dígitos en alguna fila"

    qtable = _report_nulls_and_zeros(d, [c for c in IMPORTANT_NUMS if c in d.columns])

    # Desfase producción>0 & cosecha=0
    if {"PRODUCCION","COSECHA"}.issubset(d.columns):
        casos_desfase = ((d["PRODUCCION"]>0) & (d["COSECHA"]==0)).sum()
        print("Filas con PRODUCCION>0 y COSECHA=0:", casos_desfase)

    # Producción>0 con precio faltante (ahora sí detecta NaN tras reemplazo de 0)
    if {"PRODUCCION","PRECIO_CHACRA"}.issubset(d.columns):
        faltan_precio = ((d["PRODUCCION"]>0) & (d["PRECIO_CHACRA"].isna())).sum()
        print("Filas con PRODUCCION>0 y PRECIO_CHACRA faltante:", faltan_precio)

    return qtable

# ---------- Agregaciones (sin apply deprecado) ----------
def agregados_trimestrales(d: pd.DataFrame) -> pd.DataFrame:
    req = {"UBIGEO","ANIO","TRIM","PRODUCCION","COSECHA","PRECIO_CHACRA","PRODUCCION_KG"}
    if not req.issubset(d.columns):
        return pd.DataFrame()

    tmp = d.copy()
    # Precio ponderado: sum(precio*kg) / sum(kg), usando solo filas con precio y kg válidos
    tmp["PRECIO_X_KG"] = tmp["PRECIO_CHACRA"] * tmp["PRODUCCION_KG"]

    tri = (tmp.groupby(["UBIGEO","ANIO","TRIM"], as_index=False)
             .agg(PROD_SUM=("PRODUCCION","sum"),
                  COSE_SUM=("COSECHA","sum"),
                  PRECIO_MEAN=("PRECIO_CHACRA","mean"),
                  PRECIO_COUNT=("PRECIO_CHACRA", lambda s: s.notna().sum()),
                  PROD_KG_SUM=("PRODUCCION_KG","sum"),
                  PRECIO_X_KG_SUM=("PRECIO_X_KG","sum")))

    tri["REND_THA_TRIM"] = np.where(tri["COSE_SUM"]>0, tri["PROD_SUM"]/tri["COSE_SUM"], np.nan)
    tri["PRECIO_POND"]   = np.where(tri["PROD_KG_SUM"]>0,
                                    tri["PRECIO_X_KG_SUM"]/tri["PROD_KG_SUM"],
                                    np.nan)
    tri.drop(columns=["PRECIO_X_KG_SUM"], inplace=True)
    return tri

def agregados_anuales(d: pd.DataFrame) -> pd.DataFrame:
    req = {"UBIGEO","ANIO","PRODUCCION","COSECHA","PRECIO_CHACRA","PRODUCCION_KG"}
    if not req.issubset(d.columns):
        return pd.DataFrame()

    tmp = d.copy()
    tmp["PRECIO_X_KG"] = tmp["PRECIO_CHACRA"] * tmp["PRODUCCION_KG"]

    anu = (tmp.groupby(["UBIGEO","ANIO"], as_index=False)
             .agg(PROD_SUM=("PRODUCCION","sum"),
                  COSE_SUM=("COSECHA","sum"),
                  PRECIO_MEAN=("PRECIO_CHACRA","mean"),
                  PRECIO_COUNT=("PRECIO_CHACRA", lambda s: s.notna().sum()),
                  PROD_KG_SUM=("PRODUCCION_KG","sum"),
                  PRECIO_X_KG_SUM=("PRECIO_X_KG","sum")))

    anu["REND_THA_ANUAL"] = np.where(anu["COSE_SUM"]>0, anu["PROD_SUM"]/anu["COSE_SUM"], np.nan)
    anu["PRECIO_POND"]    = np.where(anu["PROD_KG_SUM"]>0,
                                     anu["PRECIO_X_KG_SUM"]/anu["PROD_KG_SUM"],
                                     np.nan)
    anu.drop(columns=["PRECIO_X_KG_SUM"], inplace=True)
    return anu

# ---------- Cobertura de precios ----------
def cobertura_precios(d: pd.DataFrame) -> pd.DataFrame:
    req = {"UBIGEO","ANIO","PRODUCCION","PRECIO_CHACRA"}
    if not req.issubset(d.columns):
        return pd.DataFrame()

    x = d.copy()
    x["_has_prod"] = x["PRODUCCION"].fillna(0) > 0
    x["_price_ok"] = x["PRECIO_CHACRA"].notna() & (x["PRECIO_CHACRA"] > 0)

    cov = (x.groupby(["UBIGEO","ANIO"], as_index=False)
             .agg(MESES_CON_PROD=("_has_prod","sum"),
                  MESES_CON_PRECIO_OK=("_price_ok","sum")))

    cov["PRICE_COVERAGE"] = np.where(
        cov["MESES_CON_PROD"]>0,
        cov["MESES_CON_PRECIO_OK"]/cov["MESES_CON_PROD"],
        1.0
    )
    cov["ALL_PRICE_OK"] = cov["PRICE_COVERAGE"].eq(1.0)
    return cov

# ---------- Main ----------
def main():
    print("Cargando CSV crudo...")
    df = load_raw(RAW_CSV)

    print("Limpiando y derivando columnas...")
    d = clean_and_derive(df)

    print("Ejecutando checks...")
    qtable = quality_checks_and_table(d)

    print("Construyendo agregados...")
    tri = agregados_trimestrales(d)
    anu = agregados_anuales(d)

    print("Calculando cobertura de precios...")
    cov = cobertura_precios(d)

    print("Exportando...")
    _ensure_outdir()
    d.to_csv(OUT_BASE, index=False)
    if not tri.empty: tri.to_csv(OUT_TRI, index=False)
    if not anu.empty: anu.to_csv(OUT_ANU, index=False)
    if not cov.empty: cov.to_csv(OUT_COV, index=False)
    if not qtable.empty: qtable.to_csv(OUT_QLT, index=False)

    print("\nListo ✅")
    print(f"- Base limpia:         {OUT_BASE.resolve()}")
    print(f"- Trimestrales:        {OUT_TRI.resolve() if OUT_TRI.exists() else 'no generado'}")
    print(f"- Anuales:             {OUT_ANU.resolve() if OUT_ANU.exists() else 'no generado'}")
    print(f"- Cobertura de precio: {OUT_COV.resolve() if OUT_COV.exists() else 'no generado'}")
    print(f"- Resumen calidad:     {OUT_QLT.resolve() if OUT_QLT.exists() else 'no generado'}")

if __name__ == "__main__":
    main()
