import numpy as np
import pandas as pd
from src.features.engineering import create_ml_features
from src.config import PASOS_POR_HORA, PASOS_POR_DIA


def test_create_ml_features_adds_expected_lags_and_rollings() -> None:
    dates = pd.date_range("2024-01-01", periods=200, freq="10min")
    df = pd.DataFrame(
        {"zone_1_power_consumption": np.arange(len(dates))},
        index=dates,
    )

    features, target = create_ml_features(df, "zone_1_power_consumption")

    assert "lag_zone_1_power_consumption_1_hora" in features.columns
    assert "lag_zone_1_power_consumption_24_horas" in features.columns
    assert "rolling_mean_zone_1_power_consumption_1_hora" in features.columns
    assert "rolling_mean_zone_1_power_consumption_24_horas" in features.columns
    assert target.name == "zone_1_power_consumption"

    # Check lags offset equals horizons
    expected_lag_1h = (
        df["zone_1_power_consumption"]
        .shift(PASOS_POR_HORA)
        .reindex(features.index)
    )
    assert features["lag_zone_1_power_consumption_1_hora"].equals(expected_lag_1h)

    expected_lag_24h = (
        df["zone_1_power_consumption"]
        .shift(PASOS_POR_DIA)
        .reindex(features.index)
    )
    assert features["lag_zone_1_power_consumption_24_horas"].equals(expected_lag_24h)
