
import yaml, numpy as np, pandas as pd
from pathlib import Path

def clean_cols(df):
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df

def ensure_datetime(df, dt_col):
    if dt_col in df.columns and not np.issubdtype(df[dt_col].dtype, np.datetime64):
        df[dt_col] = pd.to_datetime(df[dt_col], errors="coerce", infer_datetime_format=True)
    return df

def clean_dataset(df, dt_col):
    df = df.copy()
    # strip/normalize text
    for c in df.select_dtypes(include=["object"]).columns:
        df[c] = df[c].astype(str).str.strip().replace({"": np.nan, "nan": np.nan, "NaN": np.nan})
    # drop duplicates
    df = df.drop_duplicates()
    # keep valid datetime and sort
    if dt_col in df.columns:
        df = df.dropna(subset=[dt_col]).sort_values(dt_col)
    # domain: negative power/consumption invalid -> NaN
    for c in df.columns:
        lc = c.lower()
        if any(k in lc for k in ["power","consumption","load"]) and pd.api.types.is_numeric_dtype(df[c]):
            df.loc[df[c] < 0, c] = np.nan
    return df

def main():
    root = Path(__file__).resolve().parents[1]
    params = yaml.safe_load(open(root / "params.yaml", "r", encoding="utf-8"))
    dt_col = params["features"]["datetime_column"]

    df_m = pd.read_csv(root / params["data"]["raw_modified"])    

    df_m = clean_cols(df_m)    
    df_m = ensure_datetime(df_m, dt_col) 
    df_m = clean_dataset(df_m, dt_col) 

    (root / "data" / "interim").mkdir(parents=True, exist_ok=True)
    df_m.to_parquet(root / "data" / "interim" / "modified_clean.parquet", index=False)    

    rep = {
        "shapes": {"modified": list(df_m.shape)},
        "datetime_column": dt_col,
        "nulls_modified": df_m.isna().sum().to_dict(),        
    }
    (root / "reports").mkdir(exist_ok=True)
    pd.Series(rep).to_json(root / "reports" / "eda_prepare.json")

if __name__ == "__main__":
    main()
