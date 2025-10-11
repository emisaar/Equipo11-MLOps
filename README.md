# Equipo11-MLOps

## Tetouan Power — EDA, Cleaning, Transform, and DVC Pipeline

### Estructura
```
.
├─ data/
│  ├─ raw/           # CSV originales (versionados con DVC)
│  ├─ interim/       # resultados intermedios de EDA y limpieza (prepare)
│  └─ processed/     # dataset final escalado para modelado (featurize)
├─ src/              # scripts del pipeline
├─ models/           # artefactos de modelos
├─ reports/          # métricas/figuras
├─ docs/             # documentación (roles, decisiones)
├─ notebooks/        # EDA adicional (Pruebas de cada integrante)
├─ params.yaml       # configuración de pipeline/modelos
└─ dvc.yaml          # pipeline DVC (stages eda → clean → transform)
```

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

## Primeros pasos
```bash
cd project

# inicializa Git y DVC
git init
dvc init

# versiona los datos crudos
dvc add data/raw/power_tetouan_city_modified.csv
git add data/raw/*.dvc .gitignore
git commit -m "Track Tetouan raw data with DVC"

# Reproducir pipeline completo
dvc repro

# Ver métricas
dvc metrics show

# Ver el DAG - Directed Acyclic Graph (Grafo Dirigido Acíclico).
dvc dag

# Configuracion de un carpeta local para el almacenamineto remoto
mkdir -p local-dvc-storage
dvc remote add -d localstorage local-dvc-storage

git add .dvc/config
git commit -m "chore(dvc): configure remote storage (local)"

# Sube los datos al almacenamiento remoto
dvc push -r localstorage

# Versiona los datos limpios, procesados y los sube
dvc add data/interim/loaded.parquet     # EDA y Limpieza
dvc add data/processed/test.parquet     # Dataset para Test
dvc add data/processed/train.parquet    # Dataset para Train
dvc push
```