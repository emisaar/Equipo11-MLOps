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
from typing import List
import yaml
from src.modeling.train import ModelTrainer


MODEL_ALIAS_MAP = {
    "var": "VAR",
    "random_forest": "RandomForest",
    "randomforest": "RandomForest",
    "xgboost": "XGBoost",
    "lstm": "LSTM",
}

DEFAULT_MODEL_SELECTION = ["random_forest"]


def resolve_model_types(model_section: dict) -> List[str]:
    """
    Normaliza los nombres de modelos definidos en params.yaml.

    Se aceptan alias en snake_case (p.ej. random_forest) y se
    convierten a los nombres esperados por ModelTrainer.
    """
    raw_selection = model_section.get("to_train") or model_section.get("types")

    if raw_selection is None:
        raw_selection = DEFAULT_MODEL_SELECTION

    if isinstance(raw_selection, str):
        raw_selection = [raw_selection]

    normalized = []
    for item in raw_selection:
        if item is None:
            continue

        normalized_key = str(item).strip().lower().replace("-", "_")
        mapped_name = MODEL_ALIAS_MAP.get(normalized_key)

        if mapped_name:
            normalized.append(mapped_name)
        else:
            # Mantener el valor original para no bloquear ejecuciones previas
            print(f"[WARN] Modelo '{item}' no reconocido, se usará tal cual.")
            normalized.append(str(item))

    if not normalized:
        normalized = [MODEL_ALIAS_MAP["random_forest"]]

    return normalized


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

    # Determina qué modelos entrenar según params.yaml
    selected_model_types = resolve_model_types(params["models"])
    print(f"Modelos seleccionados desde params.yaml: {', '.join(selected_model_types)}")
    print(f"Creando ModelTrainer para {len(selected_model_types)} tipos de modelos...")

    # Ejecuta entrenamiento
    trainer = ModelTrainer(
        train_path=Path(params["data"]["train"]),
        models_output_dir=Path(params["models"]["output_dir"]),
        model_types=selected_model_types,
    )

    trained_models = trainer.run()

    print(f"\n{len(trained_models)} modelos entrenados exitosamente")
    for name, path in trained_models.items():
        print(f"   • {name}: {path}")
