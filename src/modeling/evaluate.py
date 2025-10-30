# evaluate.py
# Evaluación y comparación de modelos de predicción
# ===========================

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any
import pandas as pd
import numpy as np
import json

from sklearn.metrics import mean_squared_error, mean_absolute_error

from src.modeling.train import load_all_models
from src.visualization.plots import plot_model_comparison
from src.config import POWER_COLS, N_STEPS_PREDICT


@dataclass
class ModelEvaluator:
    """
    Evalúa y compara el rendimiento de múltiples modelos de predicción.

    Esta clase coordina la evaluación de modelos VAR, Random Forest,
    XGBoost y LSTM, calculando métricas y generando visualizaciones
    comparativas.

    Attributes
    ----------
    test_path : Path
        Ruta al archivo Parquet con datos de prueba
    models_dir : Path
        Directorio con los modelos entrenados
    metrics_output : Path
        Ruta donde se guardará el archivo JSON con métricas
    figures_output : Path
        Directorio donde se guardarán las visualizaciones
    n_steps : int
        Número de pasos a predecir para evaluación
    """

    test_path: Path
    models_dir: Path
    metrics_output: Path
    figures_output: Path
    n_steps: int = N_STEPS_PREDICT

    def run(self) -> Path:
        """
        Ejecuta la evaluación completa de todos los modelos.

        El proceso incluye:
        1. Cargar modelos entrenados
        2. Cargar datos de prueba
        3. Generar predicciones para cada modelo
        4. Calcular métricas (RMSE, MAE, MAPE)
        5. Generar visualizaciones comparativas
        6. Guardar resultados en JSON

        Returns
        -------
        Path
            Ruta del archivo JSON con las métricas

        Examples
        --------
        >>> evaluator = ModelEvaluator(
        ...     test_path=Path('data/processed/test.parquet'),
        ...     models_dir=Path('models'),
        ...     metrics_output=Path('reports/metrics.json'),
        ...     figures_output=Path('reports/figures')
        ... )
        >>> metrics_path = evaluator.run()
        """
        print("\n" + "="*80)
        print("EVALUACIÓN DE MODELOS")
        print("="*80)

        # Asegura que los directorios existen
        self.metrics_output.parent.mkdir(parents=True, exist_ok=True)
        self.figures_output.mkdir(parents=True, exist_ok=True)

        # Carga modelos
        print(f"\nCargando modelos desde: {self.models_dir}")
        models = load_all_models(self.models_dir)

        if not models:
            raise RuntimeError(f"No se encontraron modelos en {self.models_dir}")

        # Carga datos de prueba
        print(f"Cargando datos de prueba desde: {self.test_path}")
        test_df = pd.read_parquet(self.test_path)

        # Establece datetime como índice si existe
        if 'datetime' in test_df.columns:
            test_df = test_df.set_index('datetime')

        print(f"   Datos de prueba: {test_df.shape}")
        print(f"   Rango temporal: {test_df.index[0]} a {test_df.index[-1]}")

        # Genera predicciones
        print(f"\nGenerando predicciones ({self.n_steps} pasos)...")
        predictions = self._generate_predictions(models, test_df)

        # Calcula métricas
        print("\nCalculando métricas...")
        metrics = self._calculate_metrics(test_df, predictions)

        # Genera visualizaciones
        print(f"\nGenerando visualizaciones en: {self.figures_output}")
        self._generate_plots(test_df, predictions)

        # Guarda métricas
        print(f"\nGuardando métricas en: {self.metrics_output}")
        self._save_metrics(metrics)

        # Muestra resumen
        self._print_summary(metrics)

        print("\n" + "="*80)
        print("EVALUACIÓN COMPLETADA")
        print("="*80 + "\n")

        return self.metrics_output

    def _generate_predictions(
        self,
        models: Dict[str, Any],
        test_df: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Genera predicciones para todos los modelos.

        Parameters
        ----------
        models : Dict[str, Any]
            Diccionario con modelos cargados
        test_df : pd.DataFrame
            DataFrame de prueba

        Returns
        -------
        Dict[str, Any]
            Diccionario con predicciones por modelo
        """
        predictions = {}

        # VAR (predicciones multivariadas)
        if 'VAR' in models:
            print("  • Generando predicciones VAR...")
            var_model = models['VAR']
            predictions['VAR'] = var_model.predict(test_df, n_steps=self.n_steps)

        # Modelos ML por zona
        for zone in POWER_COLS:
            zone_preds = {}

            # Random Forest
            rf_key = f'RF_{zone}'
            if rf_key in models:
                print(f"  • Generando predicciones RF para {zone}...")
                rf_model = models[rf_key]
                zone_preds['RF'] = rf_model.predict(test_df, n_steps=self.n_steps)

            # XGBoost
            xgb_key = f'XGB_{zone}'
            if xgb_key in models:
                print(f"  • Generando predicciones XGBoost para {zone}...")
                xgb_model = models[xgb_key]
                zone_preds['XGB'] = xgb_model.predict(test_df, n_steps=self.n_steps)

            # LSTM
            lstm_key = f'LSTM_{zone}'
            if lstm_key in models:
                print(f"  • Generando predicciones LSTM para {zone}...")
                lstm_model = models[lstm_key]
                zone_preds['LSTM'] = lstm_model.predict(test_df, n_steps=self.n_steps)

            if zone_preds:
                predictions[zone] = zone_preds

        return predictions

    def _calculate_metrics(
        self,
        test_df: pd.DataFrame,
        predictions: Dict[str, Any]
    ) -> Dict[str, Dict[str, Dict[str, float]]]:
        """
        Calcula métricas de evaluación para todos los modelos.

        Parameters
        ----------
        test_df : pd.DataFrame
            DataFrame con valores reales
        predictions : Dict[str, Any]
            Diccionario con predicciones

        Returns
        -------
        Dict[str, Dict[str, Dict[str, float]]]
            Métricas organizadas por zona y modelo
        """
        metrics = {}

        # Datos reales para los primeros n_steps
        test_data = test_df.head(self.n_steps)

        for zone in POWER_COLS:
            metrics[zone] = {}
            y_true = test_data[zone].values

            # Evalúa VAR
            if 'VAR' in predictions:
                y_pred_var = predictions['VAR'][zone].values
                metrics[zone]['VAR'] = self._compute_metrics(y_true, y_pred_var)

            # Evalúa modelos ML
            if zone in predictions:
                for model_name, y_pred in predictions[zone].items():
                    if isinstance(y_pred, pd.Series):
                        y_pred_values = y_pred.values
                    else:
                        y_pred_values = y_pred

                    metrics[zone][model_name] = self._compute_metrics(y_true, y_pred_values)

        return metrics

    def _compute_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calcula RMSE, MAE y MAPE.

        Parameters
        ----------
        y_true : np.ndarray
            Valores reales
        y_pred : np.ndarray
            Valores predichos

        Returns
        -------
        Dict[str, float]
            Diccionario con las métricas
        """
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)

        # MAPE (Mean Absolute Percentage Error)
        # Evita división por cero
        mask = y_true != 0
        if mask.sum() > 0:
            mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        else:
            mape = np.nan

        return {
            'RMSE': float(rmse),
            'MAE': float(mae),
            'MAPE': float(mape)
        }

    def _generate_plots(
        self,
        test_df: pd.DataFrame,
        predictions: Dict[str, Any]
    ) -> None:
        """
        Genera visualizaciones comparativas.

        Parameters
        ----------
        test_df : pd.DataFrame
            DataFrame con valores reales
        predictions : Dict[str, Any]
            Diccionario con predicciones
        """
        test_data = test_df.head(self.n_steps)

        for zone in POWER_COLS:
            # Construye diccionario de predicciones para esta zona
            zone_predictions = {}

            # VAR
            if 'VAR' in predictions:
                zone_predictions['VAR'] = predictions['VAR'][zone]

            # Modelos ML
            if zone in predictions:
                for model_name, preds in predictions[zone].items():
                    zone_predictions[model_name] = preds

            # Genera gráfica
            if zone_predictions:
                plot_model_comparison(
                    test_data=test_data,
                    predictions_dict=zone_predictions,
                    zone=zone,
                    output_dir=self.figures_output
                )

    def _save_metrics(self, metrics: Dict) -> None:
        """
        Guarda métricas en formato JSON.

        Parameters
        ----------
        metrics : Dict
            Diccionario con métricas
        """
        with open(self.metrics_output, 'w') as f:
            json.dump(metrics, f, indent=2)

    def _print_summary(self, metrics: Dict) -> None:
        """
        Imprime resumen de métricas en consola.

        Parameters
        ----------
        metrics : Dict
            Diccionario con métricas
        """
        print("\n" + "="*80)
        print("RESUMEN DE RESULTADOS")
        print("="*80)

        for zone in POWER_COLS:
            print(f"\n{zone.upper()}")
            print("-"*80)

            if zone in metrics:
                # Encuentra el mejor modelo
                best_model = None
                best_rmse = float('inf')

                for model_name, model_metrics in metrics[zone].items():
                    rmse = model_metrics['RMSE']
                    mae = model_metrics['MAE']
                    mape = model_metrics['MAPE']

                    print(f"{model_name:15s} | RMSE: {rmse:8.4f} | MAE: {mae:8.4f} | MAPE: {mape:6.2f}%")

                    if rmse < best_rmse:
                        best_rmse = rmse
                        best_model = model_name

                if best_model:
                    print(f"\n   Mejor modelo: {best_model} (RMSE={best_rmse:.4f})")
