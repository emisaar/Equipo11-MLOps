
import json, yaml, numpy as np, pandas as pd
from pathlib import Path

def add_time_features(df, dt_col):
    df = df.copy()
    if dt_col in df.columns:
        dt = df[dt_col]
        df["hour"] = dt.dt.hour
        df["dayofweek"] = dt.dt.dayofweek
        df["month"] = dt.dt.month
    return df

def add_lags(df, dt_col, target, lags):
    df = df.copy()
    if dt_col in df.columns:
        df = df.sort_values(dt_col)
    for L in lags:
        df[f"{target}_lag{L}"] = df[target].shift(L)
    return df

def main():
    root = Path(__file__).resolve().parents[1]
    params = yaml.safe_load(open(root / "params.yaml", "r", encoding="utf-8"))

    dt_col = params["features"]["datetime_column"]
    target = params["features"]["target"]["preferred"]
    lags = params["features"]["lags"]

    df_m = pd.read_parquet(root / "data" / "interim" / "modified_clean.parquet")    

     # Convertir target a numérico (coerción segura)
    df_m[target] = pd.to_numeric(df_m[target], errors="coerce")

    # Features de tiempo y lags
    df_m = add_time_features(df_m, dt_col)
    df_m = add_lags(df_m, dt_col, target, lags)
    
    # Drop de filas con NaN introducidos por los lags y reset de índice
    df_m = df_m.dropna().reset_index(drop=True)
    
    # Guardado de procesados
    (root / "data" / "processed").mkdir(parents=True, exist_ok=True)
    df_m.to_parquet(root / "data" / "processed" / "modified_processed.parquet", index=False)    

    # Meta minimal para etapas siguientes
    meta = {"target": target, "dt_col": dt_col, "lags": lags}
    (root / "data" / "processed" / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

if __name__ == "__main__":
    main()
