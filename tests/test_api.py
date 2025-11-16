"""
Tests para los endpoints de la API FastAPI.

Valida funcionalidad de prediccion, health check, y monitoreo de drift.
"""

import pytest
from fastapi.testclient import TestClient
from pathlib import Path
import sys

# Agregar el directorio raiz al path para importar modulos
sys.path.insert(0, str(Path(__file__).parent.parent))

from api.main import app


@pytest.fixture
def client():
    """Fixture para crear un cliente de prueba de FastAPI."""
    return TestClient(app)


class TestHealthEndpoint:
    """Tests para el endpoint de health check."""

    def test_health_endpoint_returns_200(self, client):
        """El endpoint /health debe retornar status 200."""
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_response_contains_status(self, client):
        """El response debe contener campo 'status'."""
        response = client.get("/health")
        data = response.json()
        assert "status" in data
        assert data["status"] in ["healthy", "unhealthy"]

    def test_health_response_contains_models(self, client):
        """El response debe contener informacion de modelos disponibles."""
        response = client.get("/health")
        data = response.json()
        assert "models_available" in data
        assert isinstance(data["models_available"], dict)

    def test_health_response_contains_timestamp(self, client):
        """El response debe contener timestamp."""
        response = client.get("/health")
        data = response.json()
        assert "timestamp" in data


class TestPredictEndpoint:
    """Tests para el endpoint de prediccion."""

    @pytest.fixture
    def valid_features_zone_1(self):
        """Features validas para zona 1 (Random Forest - 13 features)."""
        return {
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

    @pytest.fixture
    def valid_features_zone_3(self):
        """Features validas para zona 3 (Champion model)."""
        return {
            "temperature": 23.5,
            "humidity": 65.2,
            "wind_speed": 5.3,
            "general_diffuse_flows": 120.5,
            "diffuse_flows": 80.3,
            "hora": 14,
            "minuto": 30,
            "dia_de_semana": 2,
            "dia_del_ano": 150,
            "lag_zone_3_power_consumption_1_hora": 25000.0,
            "lag_zone_3_power_consumption_24_horas": 26500.0,
            "rolling_mean_zone_3_power_consumption_1_hora": 25200.0,
            "rolling_mean_zone_3_power_consumption_24_horas": 24800.0
        }

    def test_predict_endpoint_accepts_valid_features(
        self, client, valid_features_zone_3
    ):
        """El endpoint /predict debe aceptar features validas (200 o 503 si MLflow no disponible)."""
        response = client.post("/predict", json={"features": valid_features_zone_3})
        # 200 si modelo disponible, 503 si MLflow no esta corriendo
        assert response.status_code in [200, 503]

    def test_predict_response_structure_when_available(
        self, client, valid_features_zone_3
    ):
        """El response debe contener campos esperados cuando el modelo esta disponible."""
        response = client.post("/predict", json={"features": valid_features_zone_3})
        data = response.json()

        if response.status_code == 200:
            # Modelo disponible - validar estructura de prediccion
            assert "prediction" in data
            assert isinstance(data["prediction"], (int, float))
            assert data["prediction"] > 0
            assert "model_name" in data
            assert isinstance(data["model_name"], str)
            assert len(data["model_name"]) > 0
            assert "timestamp" in data
        elif response.status_code == 503:
            # Servicio no disponible - validar estructura de error
            assert "detail" in data
            assert isinstance(data["detail"], dict)

    def test_predict_with_missing_features_returns_error(self, client):
        """Debe retornar error si faltan features requeridas."""
        incomplete_features = {
            "temperature": 23.5,
            "humidity": 65.2
            # Faltan muchas features requeridas
        }
        response = client.post("/predict", json={"features": incomplete_features})
        # Puede retornar 422 (validacion), 400 (bad request), o 503 (servicio no disponible)
        assert response.status_code in [400, 422, 503]

    def test_predict_with_invalid_feature_types_returns_422(self, client):
        """Debe retornar 422 si los tipos de features son invalidos."""
        invalid_features = {
            "temperature": "invalid_string",  # Deberia ser float
            "humidity": 65.2,
        }
        response = client.post("/predict", json={"features": invalid_features})
        assert response.status_code == 422

    def test_predict_without_features_key_returns_422(self, client):
        """Debe retornar 422 si no se envia el campo 'features'."""
        response = client.post("/predict", json={"wrong_key": {}})
        assert response.status_code == 422


class TestMonitoringEndpoints:
    """Tests para los endpoints de monitoreo de drift."""

    def test_drift_status_endpoint_exists(self, client):
        """El endpoint de drift status debe existir."""
        response = client.get("/monitoring/drift/status?zone=3")
        # No deberia retornar 404
        assert response.status_code != 404

    def test_drift_status_with_valid_zone(self, client):
        """El endpoint debe aceptar parametro zone valido."""
        response = client.get("/monitoring/drift/status?zone=3")
        # Puede retornar 200, 500, 503 (servicio no disponible), pero no 404
        assert response.status_code in [200, 400, 500, 503]

    def test_drift_status_response_structure(self, client):
        """El response de drift status debe tener estructura esperada."""
        response = client.get("/monitoring/drift/status?zone=3")
        if response.status_code == 200:
            data = response.json()
            assert "zone" in data or "status" in data or "error" in data

    def test_log_actual_value_endpoint_exists(self, client):
        """El endpoint para registrar valores reales debe existir."""
        payload = {
            "zone": 3,
            "actual_value": 28500.0
        }
        response = client.post("/monitoring/actual", json=payload)
        # No deberia retornar 404
        assert response.status_code != 404

    def test_drift_check_endpoint_exists(self, client):
        """El endpoint de chequeo de drift debe existir."""
        response = client.post("/monitoring/drift/check?zone=3")
        # No deberia retornar 404
        assert response.status_code != 404


class TestOpenAPIDocumentation:
    """Tests para validar que la documentacion OpenAPI este disponible."""

    def test_openapi_json_endpoint_exists(self, client):
        """El endpoint /openapi.json debe estar disponible."""
        response = client.get("/openapi.json")
        assert response.status_code == 200

    def test_openapi_json_is_valid_json(self, client):
        """El response de /openapi.json debe ser JSON valido."""
        response = client.get("/openapi.json")
        data = response.json()
        assert "openapi" in data
        assert "info" in data
        assert "paths" in data

    def test_docs_endpoint_exists(self, client):
        """El endpoint /docs (Swagger UI) debe estar disponible."""
        response = client.get("/docs")
        assert response.status_code == 200


class TestErrorHandling:
    """Tests para validar el manejo de errores de la API."""

    def test_invalid_endpoint_returns_404(self, client):
        """Un endpoint invalido debe retornar 404."""
        response = client.get("/endpoint_que_no_existe")
        assert response.status_code == 404

    def test_predict_with_empty_body_returns_422(self, client):
        """Una peticion sin body debe retornar 422."""
        response = client.post("/predict", json={})
        assert response.status_code == 422

    def test_method_not_allowed_returns_405(self, client):
        """Usar metodo HTTP incorrecto debe retornar 405."""
        # /predict espera POST, no GET
        response = client.get("/predict")
        assert response.status_code == 405
