# Equipo11-MLOps

## Tetouan Power Consumption — Predicción Multivariada con MLOps

Proyecto de predicción de consumo eléctrico en Tetouan City utilizando múltiples modelos de machine learning (VAR, Random Forest, XGBoost, LSTM) con pipeline completo de MLOps.

### Estructura del Proyecto
```
.
├─ data/
│  ├─ raw/              # CSV originales (versionados con DVC)
│  ├─ interim/          # Datos cargados en formato Parquet
│  └─ processed/        # Datos limpios, preprocesados, train/test splits
│
├─ src/                 # Lógica de negocio (importable, testeable)
│  ├─ __init__.py
│  ├─ config.py         # Configuración centralizada y constantes
│  ├─ utils.py          # Utilidades de preprocesamiento
│  ├─ dataset.py        # Clases LoadData y DatasetCleaner
│  ├─ features.py       # Ingeniería de características (PreprocessData)
│  ├─ plots.py          # Visualizaciones y análisis exploratorio
│  └─ modeling/
│     ├─ models.py      # Implementación de VAR, RF, XGBoost, LSTM
│     ├─ train.py       # Orquestación de entrenamiento (ModelTrainer)
│     ├─ evaluate.py    # Evaluación y comparación (ModelEvaluator)
│     └─ predict.py     # Predicción recursiva para series temporales
│
├─ pipeline/            # Scripts de orquestación DVC
│  ├─ load_data.py      # Stage 1: Carga de datos
│  ├─ clean_data.py     # Stage 2: Limpieza
│  ├─ preprocess.py     # Stage 3: Preprocesamiento y splits
│  ├─ train.py          # Stage 4: Entrenamiento de modelos
│  └─ evaluate.py       # Stage 5: Evaluación y comparación
│
├─ models/              # Modelos entrenados (.pkl)
├─ metrics/             # Métricas de evaluación (JSON)
├─ reports/             # Figuras y visualizaciones
│  ├─ eda/
│  └─ figures/
├─ notebooks/           # Análisis individuales y experimentación
├─ tests/               # Tests unitarios (futuro)
│
├─ params.yaml          # Configuración de pipeline y hiperparámetros
├─ dvc.yaml             # Pipeline DVC (5 stages)
├─ requirements.txt     # Dependencias del proyecto
├─ README.md            # Documentación
└─ .gitignore
```

### Modelos Implementados

Este proyecto entrena y compara 4 tipos de modelos:

1. **VAR (Vector AutoRegression)**: Modelo estadístico multivariado para capturar interdependencias entre zonas
2. **Random Forest**: Modelo de ensemble con features temporales y meteorológicas
3. **XGBoost**: Gradient boosting optimizado con búsqueda de hiperparámetros
4. **LSTM**: Red neuronal recurrente para patrones temporales complejos

Cada modelo (excepto VAR) se entrena individualmente para cada una de las 3 zonas de consumo.

## Requisitos
```bash
python -m venv .venv 

# Linux/Mac:
# source .venv/bin/activate  

# Windows: 
.venv\Scripts\activate

# Actualizar pip
python.exe -m pip install --upgrade pip

# Instalar librerias desde requirements.txt
pip install -r requirements.txt

# En caso de reinstalar las libreria
# pip install --upgrade --force-reinstall -r requirements.txt
# (opcional) conda: conda env create -f environment.yml
```

## Pipeline DVC

El proyecto utiliza DVC para orquestar un pipeline de 5 etapas:

```
load_data → clean_data → preprocess_data → train_models → evaluate_models
```

### Descripción de Etapas

1. **load_data**: Carga datos CSV y convierte a Parquet
   - Entrada: `data/raw/power_tetouan_city_modified.csv`
   - Salida: `data/interim/loaded.parquet`

2. **clean_data**: Limpieza, detección de outliers, imputación
   - Entrada: `data/interim/loaded.parquet`
   - Salida: `data/processed/cleaned.parquet`

3. **preprocess_data**: Feature engineering, creación de lags, train/test split
   - Entrada: `data/processed/cleaned.parquet`
   - Salida: `data/processed/train.parquet`, `data/processed/test.parquet`

4. **train_models**: Entrenamiento de VAR, RF, XGBoost, LSTM
   - Entrada: `data/processed/train.parquet`
   - Salida: `models/*.pkl` (10 modelos: 1 VAR + 3x3 modelos por zona)

5. **evaluate_models**: Evaluación, comparación y visualización
   - Entrada: `data/processed/test.parquet`, `models/`
   - Salida: `metrics/metrics.json`, `reports/figures/`

## Primeros Pasos

### 1. Configuración Inicial
```bash
# Crear entorno virtual
python -m venv .venv

# Activar entorno
# Linux/Mac:
source .venv/bin/activate
# Windows:
.venv\Scripts\activate

# Actualizar pip
python -m pip install --upgrade pip

# Instalar dependencias
pip install -r requirements.txt
```

### 2. Inicializar DVC
```bash
# Inicializar Git y DVC (si no está hecho)
git init
dvc init

# Versionar datos crudos
dvc add data/raw/power_tetouan_city_modified.csv
git add data/raw/*.dvc .gitignore
git commit -m "Track Tetouan raw data with DVC"
```

### 3. Ejecutar Pipeline Completo
```bash
# Reproducir todo el pipeline
dvc repro

# O ejecutar etapas individuales
dvc repro load_data
dvc repro clean_data
dvc repro preprocess_data
dvc repro train_models
dvc repro evaluate_models
```

### 4. Visualizar Resultados
```bash
# Ver métricas de evaluación
dvc metrics show

# Ver el grafo del pipeline
dvc dag

# Inspeccionar métricas detalladas
cat metrics/metrics.json
```

## Configuración (params.yaml)

Todos los hiperparámetros y configuraciones se gestionan en `params.yaml`:

- **data**: Rutas de entrada/salida de datos
- **preprocessing**: Configuración de limpieza y outliers
- **split**: Proporción train/test y random state
- **models.types**: Lista de modelos a entrenar
- **models.var/random_forest/xgboost/lstm**: Hiperparámetros específicos por modelo
- **evaluation**: Configuración de evaluación y métricas

Para modificar hiperparámetros, edita `params.yaml` y ejecuta `dvc repro` para regenerar resultados.

## Almacenamiento Remoto DVC

```bash
# Configurar almacenamiento local
mkdir -p local-dvc-storage
dvc remote add -d localstorage local-dvc-storage

git add .dvc/config
git commit -m "Configure DVC remote storage"

# Subir datos y modelos al remoto
dvc push -r localstorage

# Descargar desde remoto (en otra máquina)
dvc pull -r localstorage
```

## Uso Individual de Scripts

También puedes ejecutar scripts individuales sin DVC:

```bash
# Ejecutar manualmente cada etapa del pipeline
python pipeline/load_data.py
python pipeline/clean_data.py
python pipeline/preprocess.py
python pipeline/train.py
python pipeline/evaluate.py
```

## Estructura de Métricas

El archivo `metrics/metrics.json` contiene métricas para cada zona y modelo:

```json
{
  "zone_1_power_consumption": {
    "VAR": {"RMSE": X.XX, "MAE": X.XX, "MAPE": X.XX},
    "RF": {"RMSE": X.XX, "MAE": X.XX, "MAPE": X.XX},
    "XGB": {"RMSE": X.XX, "MAE": X.XX, "MAPE": X.XX},
    "LSTM": {"RMSE": X.XX, "MAE": X.XX, "MAPE": X.XX}
  },
  ...
}
```

## Visualizaciones

Las gráficas comparativas se guardan en `reports/figures/`:
- `comparison_zone_1_power_consumption.png`
- `comparison_zone_2_power_consumption.png`
- `comparison_zone_3_power_consumption.png`

Cada gráfica muestra las predicciones de los 4 modelos vs. valores reales.