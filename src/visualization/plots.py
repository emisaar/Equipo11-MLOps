# plots.py
# Funciones para visualización de series temporales y modelos
# ===========================

from pathlib import Path
from typing import List, Dict, Any

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose


def plot_power_consumption_by_zone(
    df: pd.DataFrame,
    zones: List[str],
    datetime_col: str,
    output_dir: Path,
    plot_type: str = 'line'
) -> None:
    """
    Genera gráficas de consumo de energía para cada zona.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame con datos de consumo
    zones : List[str]
        Lista de nombres de columnas de zonas
    datetime_col : str
        Nombre de la columna datetime
    output_dir : Path
        Directorio donde guardar las figuras
    plot_type : str, default='line'
        Tipo de gráfica: 'line' o 'scatter'

    Examples
    --------
    >>> plot_power_consumption_by_zone(
    ...     df, ['zone_1_power_consumption', 'zone_2_power_consumption'],
    ...     'datetime', Path('reports/figures')
    ... )
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    for zone in zones:
        plt.figure(figsize=(10, 6))

        if plot_type == 'scatter':
            plt.scatter(
                df[datetime_col],
                df[zone],
                label=zone.replace('_', ' ').title(),
                alpha=0.6,
                s=10
            )
        else:  # line
            plt.plot(
                df[datetime_col],
                df[zone],
                label=zone.replace('_', ' ').title(),
                linewidth=1.2,
                color='red'
            )

        plt.xlabel('Datetime')
        plt.ylabel('Power Consumption')
        plt.title(f"Power Consumption - {zone.replace('_', ' ').title()}")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()

        # Guardar
        filename = f"power_consumption_{zone}_{plot_type}.png"
        plt.savefig(output_dir / filename, dpi=150, bbox_inches='tight')
        plt.close()

    print(f"Gráficas de consumo guardadas en {output_dir}")


def plot_seasonal_decomposition(
    df: pd.DataFrame,
    variable: str,
    datetime_col: str,
    output_dir: Path,
    period: int = 13,
    model: str = 'additive'
) -> None:
    """
    Genera descomposición estacional de una serie temporal.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame con datos
    variable : str
        Nombre de la variable a descomponer
    datetime_col : str
        Nombre de la columna datetime
    output_dir : Path
        Directorio donde guardar las figuras
    period : int, default=13
        Periodo para la descomposición estacional
    model : str, default='additive'
        Tipo de modelo: 'additive' o 'multiplicative'

    Examples
    --------
    >>> plot_seasonal_decomposition(
    ...     df, 'temperature', 'datetime',
    ...     Path('reports/figures'), period=13
    ... )
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Establece datetime como índice
    df_temp = df.set_index(datetime_col)[variable]

    # Realiza descomposición estacional
    result = seasonal_decompose(df_temp, model=model, period=period)

    # Gráfica de serie original
    plt.figure(figsize=(12, 3))
    plt.plot(df[datetime_col], df[variable], label='Original', color='blue')
    plt.title(f'Original Time Series - {variable}')
    plt.xlabel('DateTime')
    plt.ylabel(variable)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / f"seasonal_{variable}_original.png", dpi=150, bbox_inches='tight')
    plt.close()

    # Gráfica de tendencia
    plt.figure(figsize=(12, 3))
    plt.plot(df[datetime_col], result.trend, label='Trend', color='red')
    plt.title(f'Trend Component - {variable}')
    plt.xlabel('DateTime')
    plt.ylabel('Trend')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / f"seasonal_{variable}_trend.png", dpi=150, bbox_inches='tight')
    plt.close()

    # Gráfica de componente estacional
    plt.figure(figsize=(12, 3))
    plt.plot(df[datetime_col], result.seasonal, label='Seasonal', color='green')
    plt.title(f'Seasonal Component - {variable}')
    plt.xlabel('DateTime')
    plt.ylabel('Seasonal')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / f"seasonal_{variable}_seasonal.png", dpi=150, bbox_inches='tight')
    plt.close()

    # Gráfica de residuos
    plt.figure(figsize=(12, 3))
    plt.plot(df[datetime_col], result.resid, label='Residual', color='purple')
    plt.title(f'Residual Component - {variable}')
    plt.xlabel('DateTime')
    plt.ylabel('Residual')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / f"seasonal_{variable}_residual.png", dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Descomposición estacional de '{variable}' guardada en {output_dir}")


def plot_outliers_boxplot(
    df: pd.DataFrame,
    numeric_cols: List[str],
    output_dir: Path
) -> None:
    """
    Genera boxplots para visualizar outliers.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame con datos
    numeric_cols : List[str]
        Lista de columnas numéricas a graficar
    output_dir : Path
        Directorio donde guardar la figura

    Examples
    --------
    >>> plot_outliers_boxplot(
    ...     df, ['temperature', 'humidity'],
    ...     Path('reports/figures')
    ... )
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(len(numeric_cols), 1, figsize=(10, 18))
    plt.subplots_adjust(wspace=0.3)
    plt.tight_layout(pad=3.0)

    if len(numeric_cols) == 1:
        axes = [axes]
    else:
        axes = axes.ravel()

    for col, ax in zip(numeric_cols, axes):
        sns.boxplot(x=df[col], ax=ax, showmeans=True)
        ax.set(title=f'{col}', xlabel=None)

    plt.savefig(output_dir / "outliers_boxplots.png", dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Boxplots de outliers guardados en {output_dir}")


def plot_histograms_with_stats(
    df: pd.DataFrame,
    numeric_cols: List[str],
    output_dir: Path
) -> None:
    """
    Genera histogramas con líneas de media, mediana y moda.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame con datos
    numeric_cols : List[str]
        Lista de columnas numéricas a graficar
    output_dir : Path
        Directorio donde guardar la figura

    Examples
    --------
    >>> plot_histograms_with_stats(
    ...     df, ['temperature', 'humidity'],
    ...     Path('reports/figures')
    ... )
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(len(numeric_cols), 1, figsize=(10, 18))
    plt.subplots_adjust(wspace=0.3)
    plt.tight_layout(pad=3.0)

    if len(numeric_cols) == 1:
        axes = [axes]
    else:
        axes = axes.ravel()

    for col, ax in zip(numeric_cols, axes):
        sns.histplot(x=df[col], ax=ax, kde=True, bins=150)
        ax.set(title=f'{col}', xlabel=None)
        ax.ticklabel_format(useOffset=False, style='plain', axis='y')
        ax.ticklabel_format(useOffset=False, style='plain', axis='x')

        # Líneas de estadísticas
        ax.axvline(x=df[col].mean(), color='green', linestyle='--', label='Mean')
        ax.axvline(x=df[col].median(), color='black', linestyle='-', label='Median')
        mode_val = df[col].mode().values[0] if len(df[col].mode()) > 0 else df[col].median()
        ax.axvline(x=mode_val, color='red', linestyle='-', label='Mode')
        ax.legend()

    plt.savefig(output_dir / "histograms_with_stats.png", dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Histogramas con estadísticas guardados en {output_dir}")


def plot_model_comparison(
    test_data: pd.DataFrame,
    predictions_dict: Dict[str, Any],
    zone: str,
    output_dir: Path
) -> None:
    """
    Genera gráfica comparativa de múltiples modelos.

    Parameters
    ----------
    test_data : pd.DataFrame
        DataFrame con datos reales de test (debe tener DatetimeIndex)
    predictions_dict : Dict[str, Any]
        Diccionario con predicciones de cada modelo
        Formato: {'VAR': df_pred_var, 'RandomForest': df_pred_rf, ...}
    zone : str
        Nombre de la zona a graficar
    output_dir : Path
        Directorio donde guardar la figura

    Examples
    --------
    >>> predictions = {
    ...     'VAR': var_predictions,
    ...     'RandomForest': rf_predictions,
    ...     'XGBoost': xgb_predictions,
    ...     'LSTM': lstm_predictions
    ... }
    >>> plot_model_comparison(
    ...     test_df, predictions, 'zone_1_power_consumption',
    ...     Path('reports/figures')
    ... )
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(14, 7))

    # Datos reales
    plt.plot(
        test_data.index,
        test_data[zone],
        label='Datos Reales',
        color='black',
        linewidth=2.5
    )

    # Predicciones de cada modelo
    colors = ['blue', 'orange', 'green', 'purple', 'brown', 'pink']
    linestyles = ['--', ':', '-.', '--', ':', '-.']

    for i, (model_name, preds) in enumerate(predictions_dict.items()):
        color = colors[i % len(colors)]
        linestyle = linestyles[i % len(linestyles)]

        # Extrae predicciones de la zona
        if isinstance(preds, pd.DataFrame):
            y_pred = preds[zone] if zone in preds.columns else preds.iloc[:, 0]
        else:
            y_pred = preds

        plt.plot(
            y_pred.index if hasattr(y_pred, 'index') else test_data.index[:len(y_pred)],
            y_pred,
            label=f'{model_name}',
            linestyle=linestyle,
            color=color
        )

    plt.title(f'Comparación de Modelos - {zone}')
    plt.xlabel('Datetime')
    plt.ylabel('Power Consumption')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()

    # Guardar
    filename = f"model_comparison_{zone}.png"
    plt.savefig(output_dir / filename, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Comparación de modelos para '{zone}' guardada en {output_dir}")
