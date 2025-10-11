from pathlib import Path
import yaml
from classes import TrainModel

if __name__ == "__main__":
    params = yaml.safe_load(open("params.yaml", "r", encoding="utf-8"))
    TrainModel(
        train_parquet=Path(params["data"]["train"]),
        target=params["features"]["target"]["preferred"],
        model_out=Path(params["model"]["path"]),
        model_params=params["model"]["model_params"],
    ).run()