#!/usr/bin/env python
"""
Aplicación de testing para la API de predicción de consumo eléctrico.

Esta aplicación simula un cliente real que:
1. Hace predicciones continuamente
2. Registra valores reales observados
3. Introduce drift gradual para probar el sistema de monitoreo
4. Monitorea el estado del drift
"""

import os
import sys
import time
import logging
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime
from dotenv import load_dotenv
from colorama import init, Fore, Back, Style

# Importar módulos locales
from data_generator import DataGenerator, TrueValueSimulator
from api_client import APIClient
from visualizer import DriftVisualizer

# Inicializar colorama para colores en consola
init(autoreset=True)

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('test_app.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class TestRunner:
    """
    Ejecutor de tests para la API.

    Coordina la generación de datos, predicciones, y monitoreo.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Inicializa el ejecutor de tests.

        Parameters
        ----------
        config : Dict[str, Any]
            Configuración de la aplicación
        """
        self.config = config
        self.api_client = APIClient(config['api_base_url'])
        self.data_generator = DataGenerator()
        self.true_value_simulator = TrueValueSimulator()
        self.visualizer = DriftVisualizer()

        self.predictions = []
        self.true_values = []
        self.drift_statuses = []

    def print_header(self, text: str):
        """Imprime un encabezado colorido."""
        print("\n" + "=" * 80)
        print(Fore.CYAN + Style.BRIGHT + text.center(80))
        print("=" * 80 + "\n")

    def print_success(self, text: str):
        """Imprime mensaje de éxito."""
        print(Fore.GREEN + "[OK] " + Style.RESET_ALL + text)

    def print_error(self, text: str):
        """Imprime mensaje de error."""
        print(Fore.RED + "[ERROR] " + Style.RESET_ALL + text)

    def print_warning(self, text: str):
        """Imprime mensaje de advertencia."""
        print(Fore.YELLOW + "[WARN] " + Style.RESET_ALL + text)

    def print_info(self, text: str):
        """Imprime mensaje informativo."""
        print(Fore.BLUE + "[INFO] " + Style.RESET_ALL + text)

    def check_api_health(self) -> bool:
        """
        Verifica que la API esté disponible.

        Returns
        -------
        bool
            True si la API está disponible
        """
        self.print_header("VERIFICANDO CONEXION CON LA API")

        try:
            health = self.api_client.health_check()

            self.print_success(f"API conectada: {self.config['api_base_url']}")
            self.print_info(f"  Status: {health.get('status')}")

            models = health.get('models_available', {})
            if models:
                self.print_info("  Modelos disponibles:")
                for model_name, info in models.items():
                    champion_v = info.get('champion_version', 'N/A')
                    self.print_info(f"    - {model_name}: champion v{champion_v}")

            return True

        except Exception as e:
            self.print_error(f"No se pudo conectar con la API: {e}")
            self.print_info(f"  URL: {self.config['api_base_url']}")
            self.print_info("  Asegurate de que la API este corriendo:")
            self.print_info("    docker ps | grep power-tetouan-api")
            return False

    def run_predictions(self, n_predictions: int, zone: int = 1,
                       drift_start_at: int = None,
                       drift_type: str = 'all'):
        """
        Ejecuta un ciclo de predicciones.

        Parameters
        ----------
        n_predictions : int
            Número de predicciones a realizar
        zone : int
            Zona para la cual hacer predicciones
        drift_start_at : int, optional
            Predicción en la cual comenzar el drift
        drift_type : str
            Tipo de drift a introducir
        """
        self.print_header(f"EJECUTANDO {n_predictions} PREDICCIONES (Zona {zone})")

        if drift_start_at is not None:
            self.print_warning(
                f"Drift se introducira gradualmente a partir de la prediccion {drift_start_at}"
            )
            self.print_info(f"  Tipo de drift: {drift_type}")
            self.print_info(f"  Intensidad maxima: {self.config['drift_intensity']:.1%}")

        print()

        # Activar drift en el generador si es necesario
        if drift_start_at is not None:
            self.data_generator.activate_drift(self.config['drift_intensity'])

        # Generar batch de datos
        batch = self.data_generator.generate_batch(
            n_samples=n_predictions,
            zone=zone,
            drift_type=drift_type,
            start_drift_at=drift_start_at
        )

        # Realizar predicciones
        interval = self.config['prediction_interval']

        for i, features in enumerate(batch):
            try:
                # Hacer predicción
                prediction = self.api_client.predict(features)
                predicted_value = prediction['prediction']
                model_name = prediction['model_name']

                # Simular valor real
                # Añadir bias si hay drift para simular degradación del modelo
                in_drift = drift_start_at is not None and i >= drift_start_at
                bias = 0.05 if in_drift else 0.0  # 5% de bias durante drift

                true_value = self.true_value_simulator.simulate_true_value(
                    predicted_value,
                    add_bias=in_drift,
                    bias_amount=bias
                )

                # Registrar valor real
                self.api_client.log_actual_value(zone, true_value)

                # Guardar para visualización
                self.predictions.append({
                    'index': i,
                    'zone': zone,
                    'predicted': predicted_value,
                    'true': true_value,
                    'error': abs(predicted_value - true_value),
                    'drift_active': in_drift,
                    'model_name': model_name
                })

                # Mostrar progreso
                if (i + 1) % 10 == 0 or i == 0:
                    drift_marker = Fore.YELLOW + " [DRIFT]" + Style.RESET_ALL if in_drift else ""
                    error_pct = abs(predicted_value - true_value) / true_value * 100

                    print(f"  [{i+1:3d}/{n_predictions}] " +
                          f"Pred: {predicted_value:8.2f} kW, " +
                          f"Real: {true_value:8.2f} kW, " +
                          f"Error: {error_pct:5.2f}%{drift_marker}")

                # Esperar antes de la siguiente predicción
                if i < len(batch) - 1:
                    time.sleep(interval)

            except Exception as e:
                self.print_error(f"Error en prediccion {i+1}: {e}")
                continue

        self.print_success(f"\n{len(self.predictions)} predicciones completadas")

    def check_drift_status(self, zones: List[int] = [1, 2, 3]):
        """
        Verifica el estado del drift para las zonas especificadas.

        Parameters
        ----------
        zones : List[int]
            Zonas a verificar
        """
        self.print_header("VERIFICANDO ESTADO DE DRIFT")

        for zone in zones:
            try:
                status = self.api_client.get_drift_status(zone)

                needs_check = status.get('needs_drift_check', False)
                last_check = status.get('last_check_time', 'Nunca')
                next_check_hours = status.get('next_check_in_hours', 0)

                print(f"\n  Zona {zone}:")
                print(f"    Necesita chequeo: {needs_check}")
                print(f"    Ultimo chequeo: {last_check}")
                print(f"    Proximo chequeo en: {next_check_hours:.1f} horas")

                summary = status.get('latest_report_summary')
                if summary:
                    print(f"    Alertas totales: {summary.get('total_alerts', 0)}")
                    print(f"    Requiere accion: {summary.get('requires_action', False)}")

                    if summary.get('has_critical_alerts'):
                        self.print_warning("      ¡ALERTAS CRITICAS DETECTADAS!")

                self.drift_statuses.append(status)

            except Exception as e:
                self.print_error(f"Error verificando drift zona {zone}: {e}")

    def trigger_drift_checks(self, zones: List[int] = [1, 2, 3]):
        """
        Ejecuta chequeos manuales de drift.

        Parameters
        ----------
        zones : List[int]
            Zonas a verificar
        """
        self.print_header("EJECUTANDO CHEQUEOS DE DRIFT")

        for zone in zones:
            try:
                self.print_info(f"Ejecutando chequeo para zona {zone}...")

                result = self.api_client.trigger_drift_check(zone)

                status = result.get('status')
                message = result.get('message')

                if status == 'success':
                    self.print_success(f"Zona {zone}: {message}")

                    summary = result.get('summary', {})
                    recommendations = result.get('recommendations', [])

                    print(f"    Alertas totales: {summary.get('total_alerts', 0)}")
                    print(f"    Alertas criticas: {summary.get('has_critical_alerts', False)}")
                    print(f"    Alertas altas: {summary.get('has_high_alerts', False)}")
                    print(f"    Requiere accion: {summary.get('requires_action', False)}")

                    if recommendations:
                        print(f"\n    Recomendaciones:")
                        for rec in recommendations:
                            print(f"      - {rec}")

                elif status == 'insufficient_data':
                    self.print_warning(f"Zona {zone}: Datos insuficientes")
                else:
                    self.print_error(f"Zona {zone}: {message}")

                print()

            except Exception as e:
                self.print_error(f"Error en chequeo de drift zona {zone}: {e}")

    def generate_report(self):
        """Genera un reporte visual de los resultados."""
        self.print_header("GENERANDO REPORTE VISUAL")

        if not self.predictions:
            self.print_warning("No hay predicciones para visualizar")
            return

        try:
            output_dir = Path("test_results")
            output_dir.mkdir(exist_ok=True)

            self.visualizer.plot_predictions(self.predictions, output_dir)
            self.visualizer.plot_errors(self.predictions, output_dir)

            self.print_success(f"Reportes generados en: {output_dir}/")
            self.print_info(f"  - predictions_timeline.png")
            self.print_info(f"  - prediction_errors.png")

        except Exception as e:
            self.print_error(f"Error generando reportes: {e}")

    def run_full_test(self):
        """Ejecuta el ciclo completo de tests."""
        self.print_header("INICIANDO TEST COMPLETO DE LA API")

        # Verificar conexión
        if not self.check_api_health():
            return

        time.sleep(2)

        # Configuración del test
        zones = [int(z) for z in self.config['simulate_zones'].split(',')]
        total_predictions = self.config['total_predictions']
        drift_start = self.config['drift_start_after']
        drift_type = self.config['drift_type']

        # Realizar predicciones para cada zona
        for zone in zones:
            self.run_predictions(
                n_predictions=total_predictions,
                zone=zone,
                drift_start_at=drift_start,
                drift_type=drift_type
            )
            time.sleep(3)

        # Verificar estado de drift
        time.sleep(2)
        self.check_drift_status(zones)

        # Ejecutar chequeos de drift
        time.sleep(2)
        self.trigger_drift_checks(zones)

        # Generar reportes
        time.sleep(2)
        self.generate_report()

        self.print_header("TEST COMPLETADO")

        # Resumen final
        total_preds = len(self.predictions)
        avg_error = sum(p['error'] for p in self.predictions) / total_preds if total_preds > 0 else 0

        self.print_success(f"Predicciones totales: {total_preds}")
        self.print_info(f"Error promedio: {avg_error:.2f} kW")

        drift_preds = sum(1 for p in self.predictions if p['drift_active'])
        if drift_preds > 0:
            self.print_warning(f"Predicciones con drift: {drift_preds} ({drift_preds/total_preds*100:.1f}%)")


def load_config() -> Dict[str, Any]:
    """
    Carga la configuración desde variables de entorno.

    Returns
    -------
    Dict[str, Any]
        Configuración de la aplicación
    """
    # Cargar .env si existe
    env_path = Path('.env')
    if env_path.exists():
        load_dotenv(env_path)
        logger.info("Configuracion cargada desde .env")
    else:
        logger.warning("Archivo .env no encontrado, usando valores por defecto")

    config = {
        'api_base_url': os.getenv('API_BASE_URL', 'http://localhost:8000'),
        'prediction_interval': float(os.getenv('PREDICTION_INTERVAL', '2')),
        'drift_start_after': int(os.getenv('DRIFT_START_AFTER', '50')),
        'drift_intensity': float(os.getenv('DRIFT_INTENSITY', '0.3')),
        'log_level': os.getenv('LOG_LEVEL', 'INFO'),
        'simulate_zones': os.getenv('SIMULATE_ZONES', '1'),
        'total_predictions': int(os.getenv('TOTAL_PREDICTIONS', '200')),
        'drift_type': os.getenv('DRIFT_TYPE', 'all'),
    }

    return config


def main():
    """Función principal."""
    # Cargar configuración
    config = load_config()

    # Configurar nivel de logging
    logging.getLogger().setLevel(config['log_level'])

    # Crear y ejecutar test
    runner = TestRunner(config)
    runner.run_full_test()


if __name__ == "__main__":
    main()
