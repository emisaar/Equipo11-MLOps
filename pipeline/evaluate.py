#!/usr/bin/env python
"""
Pipeline Stage 5: EvaluaciÃ³n de modelos
EvalÃºa todos los modelos entrenados y genera mÃ©tricas comparativas.
Estructura parÃ¡metros desde DVC (params.yaml) e integra configuraciÃ³n de MLflow.
"""

from pathlib import Path
import yaml
import os
from src.modeling.evaluate import ModelEvaluator


if __name__ == "__main__":
    # Carga parÃ¡metros desde params.yaml
    params = yaml.safe_load(open("params.yaml", "r", encoding="utf-8"))

    # Ejecuta evaluaciÃ³n
    evaluator = ModelEvaluator(
        train_path=Path(params["data"]["train"]),
        test_path=Path(params["data"]["test"]),
        models_dir=Path(params["models"]["output_dir"]),
        metrics_output=Path(params["evaluation"]["metrics_output"]),
        figures_output=Path(params["evaluation"]["figures_output"]),
        n_steps=params["evaluation"]["n_steps"],
        mlflow_enabled=params["mlflow"]["mlflow_enabled"],
        mlflow_experiment=params["mlflow"]["experiment_name"],
        mlflow_tracking_uri=params["mlflow"]["tracking_uri"],
        champion_metric_name=params["mlflow"].get("champion_metric_name", "RMSE"),
        champion_higher_is_better=params["mlflow"].get("champion_higher_is_better", False),
    )

    # Exporta algunos valores al entorno para trazabilidad de MLflow
    run_name = params["mlflow"].get("run_name", None)
    if run_name:
        os.environ["MLFLOW_RUN_NAME"] = str(run_name)
    
    metrics_path = evaluator.run()

    print(f"\nEvaluaciÃ³n completada. MÃ©tricas guardadas en: {metrics_path}")
