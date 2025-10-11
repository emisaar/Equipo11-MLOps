# power_tetouan_model.py
# Módulo que define la clase PowerTetouanCityModel para orquestar todo el flujo
# end-to-end de carga, EDA, preprocesamiento, entrenamiento y evaluación del modelo
# de consumo eléctrico en Tetouan City.
# Autor: Equipo 11 - MLOps
# ===========================

"""
Power Tetouan City.

Pipeline modular para carga de datos, EDA, preprocesamiento, entrenamiento y evaluación
de un modelo de regresión para consumo eléctrico en Tetouan City.

Clases:
- LoadData        : carga CSV/Parquet y normaliza columnas.
- DataExplorer    : EDA programático (resúmenes, correlaciones).
- PreprocessData  : limpieza, imputación, generación de variables (tiempo + lags).
- TrainModel      : pipeline scikit-learn (scaler + modelo) + MLflow logging.
- EvaluateModel   : métricas, gráficos + MLflow logging.
- PowerTetouanCityModel : orquestador que integra todo (fit/predict/evaluate).

Cada clase expone un método público `.run()` que ejecuta la etapa y retorna el
artefacto principal de salida (o None cuando la etapa sólo produce archivos en
carpeta). Todas las salidas y parámetros se inyectan desde `params.yaml`.
"""
# Librerías para anotaciones y dataclasses
from __future__ import annotations
from dataclasses import dataclass

# Librerías de utilidades
from pathlib import Path

# Librerías de manejo de datos
from typing import Optional, Tuple

# librerías del pipeline de power tetouan city
from classes import LoadData, DataExplorer, PreprocessData, TrainModel, EvaluateModel

@dataclass
class PowerTetouanCityModel:
    """Clase orquestadora que integra todo el flujo end-to-end."""
    raw_path: str
    processed_out: Optional[str] = None
    target_col: str = "zone_1_power_consumption"
    lags: Tuple[int, ...] = (1,2,24)
    alpha: float = 1.0
    test_size: float = 0.2
    experiment_name: str = "TetouanCityPower"
    tracking_uri: Optional[str] = None

    def fit_evaluate(self):
        # Carga
        df = LoadData(self.raw_path).run()

        # Preprocesamiento
        pre = PreprocessData(self.target_col, lags=self.lags)
        data = pre.run(df)

        # Entrenamiento + tracking
        trainer = TrainModel(
            target_col=self.target_col,
            test_size=self.test_size,
            alpha=self.alpha,
            experiment_name=self.experiment_name,
            tracking_uri=self.tracking_uri
        )
        pipe, train_results = trainer.run(data)

        # Evaluación final
        evaluator = EvaluateModel(target_col=self.target_col)
        eval_results = evaluator.run(pipe, data)

        # Export opcional del dataset procesado
        if self.processed_out:
            Path(self.processed_out).parent.mkdir(parents=True, exist_ok=True)
            data.to_parquet(self.processed_out, index=False)

        return {"train": train_results, "eval": eval_results}

if __name__ == "__main__":
    RAW = "./data/raw/power_tetouan_city_modified.csv"
    OUT = "./data/processed/power_tetouan_city_processed_2.parquet"

    app = PowerTetouanCityModel(
        raw_path=RAW,
        processed_out=OUT,
        target_col="zone_1_power_consumption",
        lags=(1,2,24),
        alpha=1.0,
        test_size=0.2,
        experiment_name="TetouanCityPower",
        tracking_uri=None  # usa el backend por defecto (./mlruns) o configura uno propio
    )
    results = app.fit_evaluate()
    print(results)