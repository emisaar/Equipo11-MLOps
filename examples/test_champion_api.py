#!/usr/bin/env python
"""
Script de prueba para la API del modelo champion.

Este script demuestra cómo usar el nuevo endpoint de predicción
que utiliza automáticamente el modelo champion desplegado.
"""

import requests
import json
from datetime import datetime


def test_health_endpoint(api_url: str = "http://localhost:8000"):
    """
    Prueba el endpoint de health check.

    Parameters
    ----------
    api_url : str
        URL base de la API
    """
    print("\n" + "="*80)
    print("TEST 1: Health Check")
    print("="*80)

    try:
        response = requests.get(f"{api_url}/health")
        response.raise_for_status()

        data = response.json()
        print(f"[OK] Status: {data['status']}")
        print(f"[OK] Modelos disponibles:")
        for model, zones in data.get('models_available', {}).items():
            print(f"   - {model}: {zones}")

        return True

    except requests.exceptions.ConnectionError:
        print("[ERROR] No se pudo conectar a la API")
        print("  Asegurate de que la API este corriendo:")
        print("  uvicorn api.main:app --reload")
        return False

    except Exception as e:
        print(f"[ERROR] {e}")
        return False


def test_prediction_endpoint(api_url: str = "http://localhost:8000"):
    """
    Prueba el endpoint de predicción con el modelo champion.

    Parameters
    ----------
    api_url : str
        URL base de la API
    """
    print("\n" + "="*80)
    print("TEST 2: Prediccion con Modelo Champion")
    print("="*80)

    # Datos de ejemplo (ajustar según las features reales del modelo)
    payload = {
        "features": {
            "temperature": 23.5,
            "humidity": 65.2,
            "wind_speed": 5.3,
            "general_diffuse_flows": 120.5,
            "diffuse_flows": 80.3,
            "hora": 14,
            "minuto": 30,
            "dia_de_semana": 2,
            "dia_del_ano": 150,
            "lag_zone_1_power_consumption_1_hora": 25000.0,
            "lag_zone_1_power_consumption_24_horas": 26500.0,
            "rolling_mean_zone_1_power_consumption_1_hora": 25200.0,
            "rolling_mean_zone_1_power_consumption_24_horas": 24800.0
        }
    }

    print("\n[>>] Enviando request:")
    print(json.dumps(payload, indent=2))

    try:
        response = requests.post(
            f"{api_url}/predict",
            json=payload,
            headers={"Content-Type": "application/json"}
        )

        print(f"\n[<<] Status Code: {response.status_code}")

        if response.status_code == 200:
            data = response.json()
            print("\n[OK] Prediccion exitosa!")
            print(f"   Modelo usado:    {data['model_name']}")
            print(f"   Prediccion:      {data['prediction']:.2f} kW")
            print(f"   Timestamp:       {data['timestamp']}")
            print(f"   Features usadas: {len(data['features_used'])} features")
            return True
        else:
            print("\n[ERROR] Error en la prediccion:")
            print(json.dumps(response.json(), indent=2))
            return False

    except requests.exceptions.ConnectionError:
        print("[ERROR] No se pudo conectar a la API")
        return False

    except Exception as e:
        print(f"[ERROR] {e}")
        return False


def test_prediction_with_minimal_features(api_url: str = "http://localhost:8000"):
    """
    Prueba con features mínimas (para verificar manejo de errores).

    Parameters
    ----------
    api_url : str
        URL base de la API
    """
    print("\n" + "="*80)
    print("TEST 3: Prediccion con Features Minimas (debe fallar)")
    print("="*80)

    payload = {
        "features": {
            "temperature": 20.0,
            "humidity": 60.0
        }
    }

    print("\n[>>] Enviando request con features minimas:")
    print(json.dumps(payload, indent=2))

    try:
        response = requests.post(
            f"{api_url}/predict",
            json=payload,
            headers={"Content-Type": "application/json"}
        )

        print(f"\n[<<] Status Code: {response.status_code}")

        if response.status_code == 400:
            print("\n[OK] Error manejado correctamente (400 Bad Request)")
            print(json.dumps(response.json(), indent=2))
            return True
        elif response.status_code == 200:
            print("\n[WARN] Prediccion exitosa con features minimas (inesperado)")
            print(json.dumps(response.json(), indent=2))
            return True
        else:
            print("\n[ERROR] Error inesperado:")
            print(json.dumps(response.json(), indent=2))
            return False

    except Exception as e:
        print(f"[ERROR] {e}")
        return False


def main():
    """
    Ejecuta todos los tests.
    """
    print("\n" + "="*80)
    print("PRUEBAS DE LA API - MODELO CHAMPION")
    print("="*80)
    print("\nEste script prueba el nuevo endpoint de prediccion que usa")
    print("automaticamente el modelo champion desplegado.")
    print("\n[IMPORTANTE] Asegurate de que:")
    print("  1. La API este corriendo: uvicorn api.main:app --reload")
    print("  2. Exista un modelo champion: ls models/*_champion.pkl")
    print("\nPresiona Enter para continuar o Ctrl+C para cancelar...")
    input()

    api_url = "http://localhost:8000"
    results = []

    # Test 1: Health check
    results.append(("Health Check", test_health_endpoint(api_url)))

    # Test 2: Prediccion normal
    results.append(("Prediccion Normal", test_prediction_endpoint(api_url)))

    # Test 3: Features minimas
    results.append(("Features Minimas", test_prediction_with_minimal_features(api_url)))

    # Resumen
    print("\n" + "="*80)
    print("RESUMEN DE PRUEBAS")
    print("="*80)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "[OK]" if result else "[FAIL]"
        print(f"{status} {test_name}")

    print(f"\nResultado: {passed}/{total} pruebas exitosas")

    if passed == total:
        print("\n[OK] Todas las pruebas pasaron!")
        return 0
    else:
        print("\n[FAIL] Algunas pruebas fallaron")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
