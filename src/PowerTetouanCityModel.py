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
- PowerTetouanCityModel : orquestador que integra todo.

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
from typing import Optional, Dict, Tuple

# librerías para las clases del pipeline
from classes.LoadData import LoadData
from classes.ExploreData import ExploreData
from classes.PreprocessData import PreprocessData
from classes.TrainModel import TrainModel
from classes.EvaluateModel import EvaluateModel

@dataclass
class PowerTetouanCityModel:
    """
    Clase orquestadora que integra todo el flujo end-to-end.
    Carga datos, realiza EDA, preprocesa, entrena y evalúa el modelo.
    Cada etapa usa las clases modulares definidas en `classes/`.
    
    Parameters
    -raw_path : str
        Ruta al CSV/Parquet con datos sin procesar.
    -processed_out : Optional[str]
        Ruta opcional para guardar el dataset procesado (Parquet).
    -target_col : str
        Nombre de la variable objetivo a modelar.   
    -lags : Tuple[int, ...]
        Desplazamientos para crear lags del target.
    -alpha : float   
        Parámetro alpha para el modelo (si aplica).
    -test_size : float   
        Proporción del conjunto de test (0 < test_size < 1).
    -experiment_name : str
        Nombre del experimento en MLflow.
    -tracking_uri : Optional[str]            
        URI del servidor de tracking de MLflow (si None usa el backend por defecto).
    
    Methods    
    -fit_evaluate(self) -> dict
        Ejecuta todo el flujo end-to-end y retorna un diccionario con
        resultados de entrenamiento y evaluación.
    """    
    raw_path: Path
    interim_path: Path
    report_path: Path
    cleaned_path: Path = None           
    train_path: Path = None,
    test_path: Path = None,
    model_path: Path = None,
    metrics_path: Path = None,
    figures_path: Path = None,
    source: str ="file",
    date_column: str ="DateTime",      
    target_col: str = "zone_1_power_consumption"
    model_params: Optional[Dict] = None
    lags: Tuple[int, ...] = (1,2,24)
    alpha: float = 1.0
    split_test_size: float = 0.2
    split_random_state: int = 42
    experiment_name: str = "TetouanCityPower"
    tracking_uri: Optional[str] = None

    def fit_evaluate(self):
        """
        Ejecuta todo el flujo end-to-end y retorna un diccionario con
        resultados de entrenamiento y evaluación.
        Pasos:
        1) Carga datos con LoadData.
        2) EDA con DataExplorer (resúmenes, correlaciones, gráficos).
        3) Preprocesamiento con PreprocessData (limpieza, lags, split).
        4) Entrenamiento con TrainModel (pipeline + MLflow logging).
        5) Evaluación con EvaluateModel (métricas, gráficos + MLflow logging).
        6) Guarda dataset procesado si se indica self.processed_out.
        """
        # Carga de datos
        output_path = LoadData(source=self.source, 
                               input_path=self.raw_path,datetime_column=self.date_column, 
                               output_path=self.interim_path).run()    
        print(f"Data loaded and saved to: {output_path}")      

        # Exploración de datos
        explorer = ExploreData(input_parquet=output_path, 
                               report_path=self.report_path,
                               sample_rows=5)
        explorer.run()                   
        
        # Preprocesamiento
        pre = PreprocessData(input_parquet=output_path,
                             datetime_column=self.date_column,
                             target=self.target_col,
                             lags=self.lags,
                             test_size=self.split_test_size,
                             random_state=self.split_random_state,
                             out_train=self.train_path,
                             out_test=self.test_path,
                             out_cleaned=self.cleaned_path)
        data = pre.run()      

        # Entrenamiento + tracking
        trainer = TrainModel(train_parquet=self.train_path,
                             target=self.target_col,
                             model_out=self.model_path,
                             model_params=self.model_params)        
        train_results = trainer.run()
        
        # Evaluación final
        evaluator = EvaluateModel(test_parquet= self.test_path,
                                  model_path= self.model_path,
                                  target=self.target_col,
                                  metrics_path=self.metrics_path, 
                                  figures_path= self.figures_path,
                                  experiment_name=self.experiment_name)
        eval_results = evaluator.run()                      

        return {"model": self.experiment_name, "eval": eval_results}        
        

if __name__ == "__main__":
    RAW_PATH = "./data/raw/power_tetouan_city_modified.csv"
    INTERMI_PATH = "./data/interim/loaded.parquet"
    REPORT_EDA_PATH = "./reports/eda/"
    CLEANED_PATH = "./data/processed/cleaned_sample.parquet"
    TRAIN_PATH = "./data/processed/train.parquet"
    TEST_PATH = "./data/processed/test.parquet"
    MODEL_PATH = "./models/power_tetouan_model.pkl"
    METRICS_PATH = "./metrics/metrics.json"
    FIGURES_PATH = "./reports/figures/"

    app = PowerTetouanCityModel(
        raw_path=Path(RAW_PATH),
        interim_path=Path(INTERMI_PATH),
        report_path=Path(REPORT_EDA_PATH),
        cleaned_path=Path(CLEANED_PATH),
        train_path =Path(TRAIN_PATH),
        test_path =Path(TEST_PATH),
        model_path=Path(MODEL_PATH),
        metrics_path=Path(METRICS_PATH),
        figures_path=Path(FIGURES_PATH),
        source="file",
        date_column="DateTime",
        target_col="zone_1_power_consumption",
        model_params= {"n_estimators": 800,
                       "learning_rate": 0.05,
                       "max_depth": 6,
                       "min_child_weight": 3,
                       "subsample": 0.8,
                       "colsample_bytree": 0.8,
                       "gamma": 0.1,
                       "reg_lambda": 1.0,
                       "reg_alpha": 0.1,
                       "random_state": 42,
                       "objective": 'reg:squarederror',                       
                       "n_jobs": -1},
        lags=(1,2,24),
        alpha=1.0,
        split_test_size=0.3,
        split_random_state=42,
        experiment_name="TetouanCityPower",
        tracking_uri=None  # usa el backend por defecto (./mlruns) o configura uno propio
    )
    results = app.fit_evaluate()
    print(results)