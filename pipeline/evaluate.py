#!/usr/bin/env python
"""
Pipeline Stage 5: Evaluación de modelos
Evalúa todos los modelos entrenados y genera métricas comparativas.
"""

from pathlib import Path
import yaml
from src.modeling.evaluate import ModelEvaluator


if __name__ == "__main__":
    # Carga parámetros desde params.yaml
    params = yaml.safe_load(open("params.yaml", "r", encoding="utf-8"))

    # Ejecuta evaluación
    evaluator = ModelEvaluator(
        train_path=Path(params["data"]["train"]),
        test_path=Path(params["data"]["test"]),
        models_dir=Path(params["models"]["output_dir"]),
        metrics_output=Path(params["evaluation"]["metrics_output"]),
        figures_output=Path(params["evaluation"]["figures_output"]),
        n_steps=params["evaluation"]["n_steps"],
    )

    metrics_path = evaluator.run()

    print(f"\nEvaluación completada. Métricas guardadas en: {metrics_path}")
