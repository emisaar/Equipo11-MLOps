"""
Generador de datos sintéticos para testing de la API.

Este módulo genera datos que simulan condiciones normales y con drift
para probar el sistema de monitoreo de la API.
"""

import numpy as np
from typing import Dict, List, Tuple
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class DataGenerator:
    """
    Generador de datos sintéticos para predicción de consumo eléctrico.

    Simula datos normales y puede introducir drift gradual o abrupto.
    """

    def __init__(self, seed: int = 42):
        """
        Inicializa el generador de datos.

        Parameters
        ----------
        seed : int
            Semilla para reproducibilidad
        """
        np.random.seed(seed)
        self.current_time = datetime(2024, 1, 1, 0, 0)

        # Parámetros de distribución normal (basados en datos de entrenamiento)
        self.normal_params = {
            'temperature': {'mean': 20.0, 'std': 5.0, 'min': 5.0, 'max': 35.0},
            'humidity': {'mean': 65.0, 'std': 15.0, 'min': 20.0, 'max': 95.0},
            'wind_speed': {'mean': 4.0, 'std': 2.0, 'min': 0.0, 'max': 15.0},
            'general_diffuse_flows': {'mean': 100.0, 'std': 30.0, 'min': 0.0, 'max': 300.0},
            'diffuse_flows': {'mean': 75.0, 'std': 25.0, 'min': 0.0, 'max': 250.0},
        }

        # Parámetros para consumo de energía (lags y rolling means)
        self.power_params = {
            'zone_1': {'mean': 25000.0, 'std': 5000.0},
            'zone_2': {'mean': 22000.0, 'std': 4500.0},
            'zone_3': {'mean': 20000.0, 'std': 4000.0},
        }

        self.drift_active = False
        self.drift_intensity = 0.0

    def _clip_value(self, value: float, param_name: str) -> float:
        """
        Recorta el valor dentro de los límites definidos.

        Parameters
        ----------
        value : float
            Valor a recortar
        param_name : str
            Nombre del parámetro para obtener límites

        Returns
        -------
        float
            Valor recortado
        """
        params = self.normal_params.get(param_name, {})
        min_val = params.get('min', -np.inf)
        max_val = params.get('max', np.inf)
        return np.clip(value, min_val, max_val)

    def generate_normal_features(self, zone: int = 1) -> Dict[str, float]:
        """
        Genera features con distribución normal (sin drift).

        Parameters
        ----------
        zone : int
            Zona para la cual generar features (1, 2, o 3)

        Returns
        -------
        Dict[str, float]
            Diccionario con features generadas
        """
        # Incrementar tiempo
        self.current_time += timedelta(minutes=10)

        # Features meteorológicas
        features = {}
        for param, stats in self.normal_params.items():
            value = np.random.normal(stats['mean'], stats['std'])
            features[param] = float(self._clip_value(value, param))

        # Features temporales
        features['hora'] = self.current_time.hour
        features['minuto'] = self.current_time.minute
        features['dia_de_semana'] = self.current_time.weekday()
        features['dia_del_ano'] = self.current_time.timetuple().tm_yday

        # Features de consumo (lags y rolling means)
        zone_key = f'zone_{zone}'
        power_stats = self.power_params.get(zone_key, self.power_params['zone_1'])

        base_power = np.random.normal(power_stats['mean'], power_stats['std'])

        # Añadir variación por hora del día (más consumo en horas pico)
        hour_factor = 1.0
        if 7 <= features['hora'] <= 9 or 18 <= features['hora'] <= 21:
            hour_factor = 1.2
        elif 0 <= features['hora'] <= 6:
            hour_factor = 0.8

        base_power *= hour_factor

        features[f'lag_zone_{zone}_power_consumption_1_hora'] = float(base_power * np.random.uniform(0.95, 1.05))
        features[f'lag_zone_{zone}_power_consumption_24_horas'] = float(base_power * np.random.uniform(0.90, 1.10))
        features[f'rolling_mean_zone_{zone}_power_consumption_1_hora'] = float(base_power * np.random.uniform(0.97, 1.03))
        features[f'rolling_mean_zone_{zone}_power_consumption_24_horas'] = float(base_power * np.random.uniform(0.95, 1.05))

        return features

    def generate_drift_features(self, zone: int = 1, drift_type: str = 'temperature',
                               intensity: float = 0.3) -> Dict[str, float]:
        """
        Genera features con drift introducido.

        Parameters
        ----------
        zone : int
            Zona para la cual generar features
        drift_type : str
            Tipo de drift: 'temperature', 'humidity', 'seasonal', 'all'
        intensity : float
            Intensidad del drift (0.0 a 1.0)

        Returns
        -------
        Dict[str, float]
            Diccionario con features con drift
        """
        # Comenzar con features normales
        features = self.generate_normal_features(zone)

        # Aplicar drift según el tipo
        if drift_type in ['temperature', 'all']:
            # Shift en temperatura (simula cambio climático)
            shift = intensity * 10.0  # Hasta +10°C de shift
            features['temperature'] = self._clip_value(
                features['temperature'] + shift,
                'temperature'
            )

        if drift_type in ['humidity', 'all']:
            # Cambio en humedad (simula cambio en patrones climáticos)
            shift = intensity * 20.0  # Hasta +20% de shift
            features['humidity'] = self._clip_value(
                features['humidity'] + shift,
                'humidity'
            )

        if drift_type in ['seasonal', 'all']:
            # Cambio en patrones de radiación solar
            scale = 1.0 + (intensity * 0.5)  # Hasta 50% más de radiación
            features['general_diffuse_flows'] = self._clip_value(
                features['general_diffuse_flows'] * scale,
                'general_diffuse_flows'
            )
            features['diffuse_flows'] = self._clip_value(
                features['diffuse_flows'] * scale,
                'diffuse_flows'
            )

        logger.debug(f"Drift aplicado: tipo={drift_type}, intensidad={intensity:.2f}")

        return features

    def activate_drift(self, intensity: float = 0.3):
        """
        Activa el drift con una intensidad específica.

        Parameters
        ----------
        intensity : float
            Intensidad del drift (0.0 a 1.0)
        """
        self.drift_active = True
        self.drift_intensity = intensity
        logger.info(f"Drift activado con intensidad: {intensity:.2f}")

    def deactivate_drift(self):
        """Desactiva el drift."""
        self.drift_active = False
        self.drift_intensity = 0.0
        logger.info("Drift desactivado")

    def generate_batch(self, n_samples: int, zone: int = 1,
                       drift_type: str = 'all',
                       start_drift_at: int = None) -> List[Dict[str, float]]:
        """
        Genera un batch de datos con drift gradual opcional.

        Parameters
        ----------
        n_samples : int
            Número de muestras a generar
        zone : int
            Zona para la cual generar datos
        drift_type : str
            Tipo de drift a introducir
        start_drift_at : int, optional
            Muestra en la cual comenzar el drift

        Returns
        -------
        List[Dict[str, float]]
            Lista de diccionarios con features
        """
        batch = []

        for i in range(n_samples):
            # Determinar si aplicar drift
            if start_drift_at is not None and i >= start_drift_at:
                # Drift gradual: aumenta con el tiempo
                progress = (i - start_drift_at) / (n_samples - start_drift_at)
                current_intensity = min(1.0, progress * self.drift_intensity)

                features = self.generate_drift_features(
                    zone=zone,
                    drift_type=drift_type,
                    intensity=current_intensity
                )
            else:
                features = self.generate_normal_features(zone)

            batch.append(features)

        logger.info(f"Batch generado: {n_samples} muestras, zona={zone}")
        if start_drift_at is not None:
            logger.info(f"  Drift comienza en muestra {start_drift_at}")

        return batch


class TrueValueSimulator:
    """
    Simulador de valores reales de consumo eléctrico.

    Genera valores reales basados en las predicciones con ruido añadido.
    """

    def __init__(self, noise_std: float = 500.0):
        """
        Inicializa el simulador.

        Parameters
        ----------
        noise_std : float
            Desviación estándar del ruido a añadir
        """
        self.noise_std = noise_std

    def simulate_true_value(self, predicted_value: float,
                           add_bias: bool = False,
                           bias_amount: float = 0.0) -> float:
        """
        Simula el valor real basado en la predicción.

        Parameters
        ----------
        predicted_value : float
            Valor predicho por el modelo
        add_bias : bool
            Si añadir bias sistemático
        bias_amount : float
            Cantidad de bias a añadir (proporción)

        Returns
        -------
        float
            Valor real simulado
        """
        # Añadir ruido gaussiano
        noise = np.random.normal(0, self.noise_std)

        # Añadir bias si se especifica
        bias = 0.0
        if add_bias:
            bias = predicted_value * bias_amount

        true_value = predicted_value + noise + bias

        # Asegurar que sea positivo (consumo no puede ser negativo)
        return max(0.0, true_value)
