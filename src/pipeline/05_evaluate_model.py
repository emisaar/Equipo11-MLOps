from pathlib import Path
import yaml
from classes import EvaluateModel

if __name__ == "__main__":
    params = yaml.safe_load(open("params.yaml", "r", encoding="utf-8"))
    EvaluateModel(
        test_parquet=Path(params["data"]["test"]),
        model_path=Path(params["model"]["path"]),
        target=params["features"]["target"]["preferred"],
        metrics_path=Path(params["metrics"]["path"]),
        figures_path=Path(params["reports"]["figures_dir"]),
    ).run()