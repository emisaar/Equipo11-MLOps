#!/usr/bin/env python
"""
Pipeline Stage 4: Entrenamiento de modelos
Entrena múltiples modelos (VAR, RF, XGBoost, LSTM) para todas las zonas.
"""

# IMPORTANT: Configure TensorFlow BEFORE any imports that use it
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'  # Fix for macOS mutex issues

from pathlib import Path
import yaml
from src.modeling.train import ModelTrainer


if __name__ == "__main__":
    print("Iniciando pipeline de entrenamiento...")

    # Carga parámetros desde params.yaml
    print("Cargando parámetros desde params.yaml...")
    params = yaml.safe_load(open("params.yaml", "r", encoding="utf-8"))

    # Prepara configuraciones de modelos
    print("Importando configuraciones de modelos...")
    from src.config import VAR_CONFIG, RF_CONFIG, XGB_CONFIG, LSTM_CONFIG

    # Actualiza configuraciones desde params.yaml si existen
    print("Actualizando configuraciones desde params.yaml...")
    if "var" in params["models"]:
        VAR_CONFIG.update(params["models"]["var"])
    if "random_forest" in params["models"]:
        RF_CONFIG.update(params["models"]["random_forest"])
    if "xgboost" in params["models"]:
        XGB_CONFIG.update(params["models"]["xgboost"])
    if "lstm" in params["models"]:
        LSTM_CONFIG.update(params["models"]["lstm"])

    # Ejecuta entrenamiento
    print(f"Creando ModelTrainer para {len(params['models']['types'])} tipos de modelos...")
    print(f"Modelos a entrenar: {', '.join(params['models']['types'])}")
    trainer = ModelTrainer(
        train_path=Path(params["data"]["train"]),
        models_output_dir=Path(params["models"]["output_dir"]),
        model_types=params["models"]["types"],
    )

    trained_models = trainer.run()

    print(f"\n{len(trained_models)} modelos entrenados exitosamente")
    for name, path in trained_models.items():
        print(f"   • {name}: {path}")
