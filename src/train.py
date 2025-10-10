
import json, yaml, numpy as np, pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

def build_xy(df, target, dt_col):
    X = df.drop(columns=[target] + ([dt_col] if dt_col in df.columns else []), errors="ignore")
    # coerce non-numeric to numeric
    for c in X.columns:
        if not pd.api.types.is_numeric_dtype(X[c]):
            X[c] = pd.to_numeric(X[c], errors="coerce")
    y = pd.to_numeric(df[target], errors="coerce")
    # align
    idx = y.dropna().index.intersection(X.dropna(how="any").index)
    return X.loc[idx], y.loc[idx]

def split(df, X, y, dt_col, test_size=0.2):
    if dt_col in df.columns:
        df_sorted = df.loc[X.index].sort_values(dt_col)
        idx = df_sorted.index
        split_i = int((1 - test_size) * len(idx))
        tr, te = idx[:split_i], idx[split_i:]
        return X.loc[tr], X.loc[te], y.loc[tr], y.loc[te]
    else:
        return train_test_split(X, y, test_size=test_size, random_state=42)

def main():
    root = Path(__file__).resolve().parents[1]
    models_dir = root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    params = yaml.safe_load(open(root / "params.yaml", "r", encoding="utf-8"))
    meta = json.loads((root / "data" / "processed" / "meta.json").read_text())
    target, dt_col = meta["target"], meta["dt_col"]

    df_m = pd.read_parquet(root / "data" / "processed" / "modified_processed.parquet")    

    results = []
    models = {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(**params["models"]["ridge"]),
        "RandomForest": RandomForestRegressor(**params["models"]["random_forest"])
    }

    def train_one(df, tag):
        X, y = build_xy(df, target, dt_col)
        pre = ColumnTransformer([("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), X.columns.tolist())])
        Xtr, Xte, ytr, yte = split(df, X, y, dt_col, params["split"]["test_size"])
        for name, model in models.items():
            pipe = Pipeline([("prep", pre), ("model", model)])
            pipe.fit(Xtr, ytr)
            pred = pipe.predict(Xte)
            results.append({"dataset": tag, "model": name,
                            "rmse": float(np.sqrt(((yte - pred)**2).mean())),
                            "mae": float(np.abs(yte - pred).mean()),
                            "r2": float(r2_score(yte, pred))})
            joblib.dump(pipe, models_dir / f"{tag}_{name}.joblib")

    train_one(df_m, "modified")    

    pd.DataFrame(results).to_csv(root / "reports" / "model_metrics.csv", index=False)

if __name__ == "__main__":
    main()
