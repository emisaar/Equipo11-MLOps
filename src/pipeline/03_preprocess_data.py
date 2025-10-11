from pathlib import Path
import yaml
from classes import PreprocessData

if __name__ == "__main__":
    params = yaml.safe_load(open("params.yaml", "r", encoding="utf-8"))
    p = params
    PreprocessData(
        input_parquet=Path(p["data"]["interim_loaded"]),
        datetime_column=p["features"]["datetime_column"],
        target=p["features"]["target"]["preferred"],
        lags=p["features"]["lags"],
        test_size=p["split"]["test_size"],
        random_state=p["split"]["random_state"],
        out_train=Path(p["data"]["train"]),
        out_test=Path(p["data"]["test"]),
        out_cleaned=Path(p["data"]["cleaned"]),
    ).run()