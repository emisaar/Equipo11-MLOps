# train.py
# Orquestación del entrenamiento de múltiples modelos de predicción
# ===========================

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
import joblib

from src.modeling.models import (
    VARModel,
    RandomForestModel,
    XGBoostModel,
    LSTMModel
)
from src.config import (
    POWER_COLS,
    WEATHER_COLS,
    VAR_CONFIG,
    RF_CONFIG,
    XGB_CONFIG,
    LSTM_CONFIG
)


@dataclass
class ModelTrainer:
    """
    Entrena múltiples tipos de modelos para predicción de consumo de energía.

    Esta clase coordina el entrenamiento de modelos VAR, Random Forest,
    XGBoost y LSTM sobre los datos de consumo de energía de Tetouan City.

    Attributes
    ----------
    train_path : Path
        Ruta al archivo Parquet con datos de entrenamiento
    models_output_dir : Path
        Directorio donde se guardarán los modelos entrenados
    model_types : List[str]
        Lista de tipos de modelos a entrenar.
        Opciones: 'VAR', 'RandomForest', 'XGBoost', 'LSTM'
    """

    train_path: Path
    models_output_dir: Path
    model_types: Optional[List[str]] = None

    def __post_init__(self):
        """Inicializa tipos de modelos por defecto si no se especifican."""
        if self.model_types is None:
            self.model_types = ['VAR', 'RandomForest', 'XGBoost', 'LSTM']

    def run(self) -> Dict[str, Path]:
        """
        Ejecuta el entrenamiento de todos los modelos especificados.

        El proceso incluye:
        1. Cargar y preparar datos de entrenamiento
        2. Entrenar modelo VAR (multivariado) si está especificado
        3. Entrenar modelos ML univariados (RF, XGBoost, LSTM) para cada zona
        4. Guardar todos los modelos en disco

        Returns
        -------
        Dict[str, Path]
            Diccionario con las rutas de los modelos guardados.
            Claves: nombres descriptivos de modelos
            Valores: rutas absolutas a los archivos .pkl

        Examples
        --------
        >>> trainer = ModelTrainer(
        ...     train_path=Path('data/processed/train.parquet'),
        ...     models_output_dir=Path('models'),
        ...     model_types=['RandomForest', 'XGBoost']
        ... )
        >>> paths = trainer.run()
        """
        # Asegura que el directorio de modelos existe
        self.models_output_dir.mkdir(parents=True, exist_ok=True)

        # Carga datos de entrenamiento
        print("\n" + "="*80)
        print("ENTRENAMIENTO DE MODELOS")
        print("="*80)

        train_df = pd.read_parquet(self.train_path)

        # Establece datetime como índice si existe
        if 'datetime' in train_df.columns:
            train_df = train_df.set_index('datetime')

        print(f"\nDatos de entrenamiento cargados: {train_df.shape}")
        print(f"   Rango temporal: {train_df.index[0]} a {train_df.index[-1]}")

        trained_models = {}

        # Entrena VAR (multivariado - todas las zonas juntas)
        if 'VAR' in self.model_types:
            print("\n" + "-"*80)
            print("1. MODELO VAR (Multivariado)")
            print("-"*80)
            var_path = self._train_var(train_df)
            trained_models['VAR'] = var_path

        # Entrena modelos ML (uno por zona)
        for zone in POWER_COLS:
            print(f"\n" + "-"*80)
            print(f"2. MODELOS ML PARA ZONA: {zone}")
            print("-"*80)

            if 'RandomForest' in self.model_types:
                rf_path = self._train_rf(train_df, zone)
                trained_models[f'RF_{zone}'] = rf_path

            if 'XGBoost' in self.model_types:
                xgb_path = self._train_xgb(train_df, zone)
                trained_models[f'XGB_{zone}'] = xgb_path

            if 'LSTM' in self.model_types:
                lstm_path = self._train_lstm(train_df, zone)
                trained_models[f'LSTM_{zone}'] = lstm_path

        print("\n" + "="*80)
        print(f"ENTRENAMIENTO COMPLETADO - {len(trained_models)} modelos guardados")
        print("="*80 + "\n")

        return trained_models

    def _train_var(self, train_df: pd.DataFrame) -> Path:
        """
        Entrena modelo VAR multivariado.

        Parameters
        ----------
        train_df : pd.DataFrame
            DataFrame con todas las variables del sistema

        Returns
        -------
        Path
            Ruta del modelo guardado
        """
        # Selecciona solo las columnas de zonas de consumo
        var_data = train_df[POWER_COLS].copy()

        model = VARModel()
        model.train(var_data, **VAR_CONFIG)

        # Guarda modelo
        path = self.models_output_dir / "var_model.pkl"
        joblib.dump(model, path)

        print(f"Modelo guardado: {path}")
        return path

    def _train_rf(self, train_df: pd.DataFrame, zone: str) -> Path:
        """
        Entrena Random Forest para una zona específica.

        Parameters
        ----------
        train_df : pd.DataFrame
            DataFrame completo de entrenamiento
        zone : str
            Nombre de la columna objetivo (zona)

        Returns
        -------
        Path
            Ruta del modelo guardado
        """
        model = RandomForestModel()
        model.train(
            train_df,
            target_col=zone,
            weather_cols=WEATHER_COLS,
            **RF_CONFIG
        )

        # Guarda modelo
        path = self.models_output_dir / f"rf_{zone}.pkl"
        joblib.dump(model, path)

        print(f"Modelo guardado: {path}")
        return path

    def _train_xgb(self, train_df: pd.DataFrame, zone: str) -> Path:
        """
        Entrena XGBoost para una zona específica.

        Parameters
        ----------
        train_df : pd.DataFrame
            DataFrame completo de entrenamiento
        zone : str
            Nombre de la columna objetivo (zona)

        Returns
        -------
        Path
            Ruta del modelo guardado
        """
        model = XGBoostModel()
        model.train(
            train_df,
            target_col=zone,
            weather_cols=WEATHER_COLS,
            **XGB_CONFIG
        )

        # Guarda modelo
        path = self.models_output_dir / f"xgb_{zone}.pkl"
        joblib.dump(model, path)

        print(f"Modelo guardado: {path}")
        return path

    def _train_lstm(self, train_df: pd.DataFrame, zone: str) -> Path:
        """
        Entrena LSTM para una zona específica.

        Parameters
        ----------
        train_df : pd.DataFrame
            DataFrame completo de entrenamiento
        zone : str
            Nombre de la columna objetivo (zona)

        Returns
        -------
        Path
            Ruta del modelo guardado
        """
        model = LSTMModel()
        model.train(
            train_df,
            target_col=zone,
            weather_cols=WEATHER_COLS,
            **LSTM_CONFIG
        )

        # Guarda modelo
        path = self.models_output_dir / f"lstm_{zone}.pkl"
        joblib.dump(model, path)

        print(f"Modelo guardado: {path}")
        return path


# ===== Funciones de utilidad =====

def load_trained_model(model_path: Path):
    """
    Carga un modelo previamente entrenado.

    Parameters
    ----------
    model_path : Path
        Ruta al archivo .pkl del modelo

    Returns
    -------
    BaseModel
        Instancia del modelo cargado

    Examples
    --------
    >>> model = load_trained_model(Path('models/rf_zone_1_power_consumption.pkl'))
    >>> predictions = model.predict(train_df, n_steps=10)
    """
    return joblib.load(model_path)


def load_all_models(models_dir: Path) -> Dict[str, any]:
    """
    Carga todos los modelos desde un directorio.

    Parameters
    ----------
    models_dir : Path
        Directorio con los modelos entrenados

    Returns
    -------
    Dict[str, any]
        Diccionario con modelos cargados.
        Claves: nombres de modelos
        Valores: instancias de modelos

    Examples
    --------
    >>> models = load_all_models(Path('models'))
    >>> var_model = models['VAR']
    """
    models = {}

    # Carga VAR
    var_path = models_dir / "var_model.pkl"
    if var_path.exists():
        models['VAR'] = joblib.load(var_path)

    # Carga modelos por zona
    for zone in POWER_COLS:
        # Random Forest
        rf_path = models_dir / f"rf_{zone}.pkl"
        if rf_path.exists():
            models[f'RF_{zone}'] = joblib.load(rf_path)

        # XGBoost
        xgb_path = models_dir / f"xgb_{zone}.pkl"
        if xgb_path.exists():
            models[f'XGB_{zone}'] = joblib.load(xgb_path)

        # LSTM
        lstm_path = models_dir / f"lstm_{zone}.pkl"
        if lstm_path.exists():
            models[f'LSTM_{zone}'] = joblib.load(lstm_path)

    print(f"{len(models)} modelos cargados desde {models_dir}")
    return models