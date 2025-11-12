import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_series_equal

from src.preprocessing.normalization import clean_name, normalize_column_names
from src.preprocessing.imputation import generar_media_movil, imputar_nans_con_media_movil
from src.preprocessing.outliers import outliers


# Ensures clean_name strips and underscoredizes column labels.
def test_clean_name_trims_and_replaces_spaces() -> None:
    assert clean_name("  Mixed  Column  ") == "mixed_column"


# Validates normalization reduces whitespace and capital letters systematically.
def test_normalize_column_names_handle_spacing_and_case() -> None:
    example = pd.DataFrame(columns=["  DateTime  ", "Mixed   Case", "Simple"])
    normalized = normalize_column_names(example)
    assert list(normalized.columns) == ["datetime", "mixed_case", "simple"]


# Checks moving average honors window and min_periods parameters.
def test_generar_media_movil_respects_window_and_min_periods() -> None:
    example = pd.Series([1.0, 2.0, 3.0, 4.0])
    ma = generar_media_movil(example, window=2, center=False, min_periods=1)
    expected = pd.Series([1.0, 1.5, 2.5, 3.5])
    assert_series_equal(ma.reset_index(drop=True), expected)


# Verifies NaNs get filled from the corresponding moving average series.
def test_imputar_nans_con_media_movil_fills_missing_values() -> None:
    values = pd.Series([1.0, np.nan, 5.0])
    moving_average = pd.Series([1.0, 2.5, 3.0])
    imputed = imputar_nans_con_media_movil(values, moving_average)
    assert imputed.iloc[1] == 2.5
    assert imputed.isna().sum() == 0


# Ensures outlier removal drops the upper tail when replace=False.
def test_outliers_remove_upper_tail_when_replace_disabled() -> None:
    df = pd.DataFrame({"value": [1, 2, 3, 100, 4, 5]})
    outliers(df, "value", method="IQR", replace=False, limit_side="upper")
    assert 100 not in df["value"].values
    assert len(df) == 5


# Confirms a ValueError is raised for unknown columns during outlier detection.
def test_outliers_invalid_column_raises_value_error() -> None:
    df = pd.DataFrame({"value": [1, 2, 3]})
    with pytest.raises(ValueError):
        outliers(df, "missing", method="IQR")
