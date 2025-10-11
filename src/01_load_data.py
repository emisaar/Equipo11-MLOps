from pathlib import Path
import yaml
from classes import LoadData

if __name__ == "__main__":
    params = yaml.safe_load(open("params.yaml", "r", encoding="utf-8"))
    LoadData(
        input_path=Path(params["data"]["raw_path"]),
        datetime_column=params["features"]["datetime_column"],
        output_path=Path(params["data"]["interim_loaded"]),
    ).run()