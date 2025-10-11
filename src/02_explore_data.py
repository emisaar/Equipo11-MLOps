from pathlib import Path
import yaml
from classes import ExploreData

if __name__ == "__main__":
    params = yaml.safe_load(open("params.yaml", "r", encoding="utf-8"))
    ExploreData(
        input_parquet=Path(params["data"]["interim_loaded"]),
        report_dir=Path(params["reports"]["eda_dir"]),
        sample_rows=params["eda"]["sample_rows"],
    ).run()