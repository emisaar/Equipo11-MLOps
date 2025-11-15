"""
Módulo de visualización para los resultados de los tests.

Genera gráficas de predicciones, errores y drift.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

# Configurar estilo de seaborn
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 6)


class DriftVisualizer:
    """
    Visualizador de resultados de tests con drift.
    """

    def __init__(self):
        """Inicializa el visualizador."""
        self.colors = {
            'normal': '#2ecc71',
            'drift': '#e74c3c',
            'predicted': '#3498db',
            'true': '#2c3e50'
        }

    def plot_predictions(self, predictions: List[Dict[str, Any]],
                        output_dir: Path):
        """
        Grafica las predicciones vs valores reales a lo largo del tiempo.

        Parameters
        ----------
        predictions : List[Dict[str, Any]]
            Lista de predicciones realizadas
        output_dir : Path
            Directorio donde guardar las gráficas
        """
        if not predictions:
            logger.warning("No hay predicciones para graficar")
            return

        df = pd.DataFrame(predictions)

        fig, ax = plt.subplots(figsize=(16, 8))

        # Graficar predicciones
        ax.plot(df['index'], df['predicted'],
                label='Predicho',
                color=self.colors['predicted'],
                linewidth=2,
                alpha=0.8)

        # Graficar valores reales
        ax.plot(df['index'], df['true'],
                label='Real',
                color=self.colors['true'],
                linewidth=2,
                alpha=0.8,
                linestyle='--')

        # Marcar región de drift
        if 'drift_active' in df.columns:
            drift_start = df[df['drift_active']].index.min() if df['drift_active'].any() else None

            if drift_start is not None:
                ax.axvspan(drift_start, len(df),
                          alpha=0.2,
                          color=self.colors['drift'],
                          label='Período con Drift')

        ax.set_xlabel('Número de Predicción', fontsize=12, fontweight='bold')
        ax.set_ylabel('Consumo Eléctrico (kW)', fontsize=12, fontweight='bold')
        ax.set_title('Predicciones vs Valores Reales a lo Largo del Tiempo',
                    fontsize=14, fontweight='bold')
        ax.legend(fontsize=11, loc='best')
        ax.grid(True, alpha=0.3)

        # Guardar
        output_path = output_dir / 'predictions_timeline.png'
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Grafica de predicciones guardada: {output_path}")

    def plot_errors(self, predictions: List[Dict[str, Any]],
                   output_dir: Path):
        """
        Grafica los errores de predicción.

        Parameters
        ----------
        predictions : List[Dict[str, Any]]
            Lista de predicciones realizadas
        output_dir : Path
            Directorio donde guardar las gráficas
        """
        if not predictions:
            logger.warning("No hay predicciones para graficar")
            return

        df = pd.DataFrame(predictions)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))

        # Gráfica 1: Error absoluto a lo largo del tiempo
        colors = [self.colors['drift'] if d else self.colors['normal']
                 for d in df['drift_active']]

        ax1.scatter(df['index'], df['error'],
                   c=colors,
                   alpha=0.6,
                   s=50)

        # Línea de tendencia
        if len(df) > 1:
            z = np.polyfit(df['index'], df['error'], 2)
            p = np.poly1d(z)
            ax1.plot(df['index'], p(df['index']),
                    "r--",
                    alpha=0.8,
                    linewidth=2,
                    label='Tendencia')

        ax1.set_xlabel('Número de Predicción', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Error Absoluto (kW)', fontsize=12, fontweight='bold')
        ax1.set_title('Error de Predicción a lo Largo del Tiempo',
                     fontsize=14, fontweight='bold')
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)

        # Gráfica 2: Distribución de errores (normal vs drift)
        normal_errors = df[~df['drift_active']]['error']
        drift_errors = df[df['drift_active']]['error']

        if len(normal_errors) > 0:
            ax2.hist(normal_errors, bins=30,
                    alpha=0.6,
                    color=self.colors['normal'],
                    label=f'Sin Drift (n={len(normal_errors)})',
                    edgecolor='black')

        if len(drift_errors) > 0:
            ax2.hist(drift_errors, bins=30,
                    alpha=0.6,
                    color=self.colors['drift'],
                    label=f'Con Drift (n={len(drift_errors)})',
                    edgecolor='black')

        ax2.set_xlabel('Error Absoluto (kW)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Frecuencia', fontsize=12, fontweight='bold')
        ax2.set_title('Distribución de Errores: Normal vs Drift',
                     fontsize=14, fontweight='bold')
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3)

        # Guardar
        output_path = output_dir / 'prediction_errors.png'
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Grafica de errores guardada: {output_path}")


# Importar numpy para la línea de tendencia
import numpy as np
