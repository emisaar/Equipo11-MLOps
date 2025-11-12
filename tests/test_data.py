from pathlib import Path

import pandas as pd
import pytest

from src.data.loaders import LoadData


# Verifica que LoadData lea un CSV, normalice columnas y escriba Parquet.
def test_load_data_csv_roundtrip(tmp_path: Path) -> None:
    csv_path = tmp_path / "raw.csv"
    pd.DataFrame(
        {
            "DateTime": ["2025-01-01 00:00", "2025-01-01 00:10"],
            "Temperature": [20.0, 20.5],
            "Zone 1 Power Consumption": [1000, 1010],
        }
    ).to_csv(csv_path, index=False)

    output_path = tmp_path / "processed.parquet"
    loader = LoadData(input_path=csv_path, output_path=output_path)
    result_path = loader.run()

    assert result_path == output_path
    df = pd.read_parquet(result_path)
    assert "datetime" in df.columns
    assert df["zone_1_power_consumption"].iloc[0] == 1000


# Comprueba que la inspección de cabeceras detecte columnas presentes y ausentes.
def test_csv_has_column_detects_headers(tmp_path: Path) -> None:
    csv_path = tmp_path / "data.csv"
    pd.DataFrame({"datetime": [0], "value": [1]}).to_csv(csv_path, index=False)
    loader = LoadData(input_path=csv_path, output_path=tmp_path / "out.parquet")
    assert loader._csv_has_column(csv_path, "datetime") is True
    assert loader._csv_has_column(csv_path, "missing") is False


# Confirma que extensiones no soportadas lanzan ValueError.
def test_load_data_unsupported_extension(tmp_path: Path) -> None:
    txt_path = tmp_path / "bad.txt"
    txt_path.write_text("unsupported")
    loader = LoadData(input_path=txt_path, output_path=tmp_path / "out.parquet")
    with pytest.raises(ValueError):
        loader._load_file()
