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
from typing import List, Dict, Any, Optional
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
# Crear directorio de logs en el proyecto principal
log_dir = Path(__file__).parent.parent / "logs" / "test_app"
log_dir.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / 'test_app.log'),
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
        self._feature_zone: Optional[int] = None

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

    def _infer_champion_zone(self, models_info: Dict[str, Any]) -> Optional[int]:
        """
        Intenta deducir la zona asociada al modelo champion a partir
        de la lista de modelos disponibles.
        """
        if not models_info:
            return None

        for model_name in models_info.keys():
            for zone in (1, 2, 3):
                if f"zone_{zone}" in model_name:
                    return zone
        return None

    def check_api_health(self) -> bool:
        """
        Verifica que la API está disponible.

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

                inferred_zone = self._infer_champion_zone(models)
                if inferred_zone is not None:
                    self._feature_zone = inferred_zone
                    self.print_info(f"  Zona inferida para test: {inferred_zone}")
                else:
                    self.print_warning(
                        "  No se pudo inferir la zona del modelo, usando zona 1 por defecto"
                    )

            return True

        except Exception as e:
            self.print_error(f"No se pudo conectar con la API: {e}")
            self.print_info(f"  URL: {self.config['api_base_url']}")
            self.print_info("  Asegurate de que la API este corriendo:")
            self.print_info("    docker ps | grep power-tetouan-api")
            return False

    def run_predictions(self, n_predictions: int,
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
        self.print_header(f"EJECUTANDO {n_predictions} PREDICCIONES")

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
        feature_zone = self._feature_zone or 1
        batch = self.data_generator.generate_batch(
            n_samples=n_predictions,
            zone=feature_zone,
            drift_type=drift_type,
            start_drift_at=drift_start_at
        )

        # Realizar predicciones
        interval = self.config['prediction_interval']

        for i, features in enumerate(batch):
            try:
                # Hacer predicción
                prediction = self.api_client.predict(feature_zone, features)
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
                self.api_client.log_actual_value(feature_zone, true_value)

                # Guardar para visualización
                self.predictions.append({
                    'index': i,
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


    def check_drift_status(self):
        """Verifica el estado del drift usando la zona interna."""
        self.print_header("CONSULTANDO ESTADO DE DRIFT")
        zone = self._feature_zone or 1
        try:
            status = self.api_client.get_drift_status(zone)

            needs_check = status.get('needs_drift_check')
            next_check = status.get('next_check_in_hours')
            last_check = status.get('last_check_time')

            if needs_check:
                self.print_warning("Drift: se recomienda ejecutar chequeo manual")
            else:
                self.print_success("Drift bajo control")

            self.print_info(f"  Ultimo chequeo: {last_check}")
            self.print_info(f"  Proximo chequeo en: {next_check:.1f} horas")

            summary = status.get('latest_report_summary')
            if summary:
                alerts = summary.get('total_alerts', 0)
                requires_action = summary.get('requires_action')

                if alerts > 0 and requires_action:
                    self.print_warning(f"  Alertas de drift: {alerts} (requiere accion)")
                elif alerts > 0:
                    self.print_info(f"  Alertas menores: {alerts}")
                else:
                    self.print_success("  Sin alertas recientes")

            print()
            self.drift_statuses.append({'zone': zone, 'status': status})

        except Exception as e:
            self.print_error(f"Error consultando drift: {e}")

    def trigger_drift_checks(self):
        """Ejecuta un chequeo manual de drift usando la zona interna."""
        self.print_header("EJECUTANDO CHEQUEO MANUAL DE DRIFT")
        zone = self._feature_zone or 1
        try:
            result = self.api_client.trigger_drift_check(zone)
            status = result.get('status')
            message = result.get('message', '')

            if status == 'success':
                self.print_success("Chequeo completado")
                summary = result.get('summary', {})
                if summary:
                    alerts = summary.get('total_alerts', 0)
                    requires_action = summary.get('requires_action', False)
                    self.print_info(f"  Alertas: {alerts}")
                    self.print_info(f"  Requiere accion: {'SI' if requires_action else 'NO'}")
            elif status == 'insufficient_data':
                self.print_warning(message)
            else:
                self.print_error(message)

            print()

        except Exception as e:
            self.print_error(f"Error en chequeo de drift: {e}")

    def generate_report(self):
        """Genera un reporte visual de los resultados."""
        self.print_header("GENERANDO REPORTE VISUAL")

        if not self.predictions:
            self.print_warning("No hay predicciones para visualizar")
            return

        try:
            # Guardar reportes en el directorio reports del proyecto principal
            output_dir = Path(__file__).parent.parent / "reports" / "test_app"
            output_dir.mkdir(parents=True, exist_ok=True)

            self.visualizer.plot_predictions(self.predictions, output_dir)
            self.visualizer.plot_errors(self.predictions, output_dir)

            self.print_success(f"Reportes generados en: {output_dir.relative_to(Path.cwd().parent)}/")
            self.print_info(f"  - predictions_timeline.png")
            self.print_info(f"  - prediction_errors.png")

        except Exception as e:
            self.print_error(f"Error generando reportes: {e}")

    def run_full_test(self):
        """Ejecuta el ciclo completo de tests."""
        self.print_header("INICIANDO TEST COMPLETO DE LA API")

        # Verificar conexi?n
        if not self.check_api_health():
            return

        time.sleep(2)

        total_predictions = self.config['total_predictions']
        drift_start = self.config['drift_start_after']
        drift_type = self.config['drift_type']

        # Realizar predicciones
        self.run_predictions(
            n_predictions=total_predictions,
            drift_start_at=drift_start,
            drift_type=drift_type
        )
        time.sleep(3)

        # Verificar estado de drift
        time.sleep(2)
        self.check_drift_status()

        # Ejecutar chequeo de drift
        time.sleep(2)
        self.trigger_drift_checks()

        # Generar reportes
        time.sleep(2)
        self.generate_report()

        self.print_header("TEST COMPLETADO")

        total_preds = len(self.predictions)
        avg_error = (
            sum(p['error'] for p in self.predictions) / total_preds
            if total_preds > 0 else 0
        )

        self.print_success(f"Predicciones totales: {total_preds}")
        self.print_info(f"Error promedio: {avg_error:.2f} kW")

        drift_preds = sum(1 for p in self.predictions if p['drift_active'])
        if drift_preds > 0:
            self.print_warning(
                f"Predicciones con drift: {drift_preds} ({drift_preds/total_preds*100:.1f}%)"
            )


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












