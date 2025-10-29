# Makefile para automatizar comandos del proyecto MLOps
# Uso: make <comando>

.PHONY: help install clean data pipeline train evaluate all dvc-repro dvc-dag

# Comando por defecto: mostrar ayuda
help:
	@echo "Comandos disponibles:"
	@echo "  make install       - Instalar dependencias del proyecto"
	@echo "  make clean         - Limpiar archivos generados"
	@echo "  make data          - Ejecutar pipeline de datos (load + clean + preprocess)"
	@echo "  make train         - Entrenar todos los modelos"
	@echo "  make evaluate      - Evaluar modelos entrenados"
	@echo "  make pipeline      - Ejecutar pipeline completo (data + train + evaluate)"
	@echo "  make dvc-repro     - Reproducir pipeline DVC completo"
	@echo "  make dvc-dag       - Visualizar DAG del pipeline DVC"
	@echo "  make all           - Limpiar y ejecutar pipeline completo"

# Instalar dependencias
install:
	pip install -e .
	@echo "Paquete instalado en modo editable"

# Limpiar archivos generados
clean:
	rm -rf data/interim/* data/processed/* models/* reports/*.csv reports/figures/*
	@echo "Archivos de salida eliminados"

# Pipeline de datos
data:
	python3 pipeline/load_data.py
	python3 pipeline/clean_data.py
	python3 pipeline/preprocess.py
	@echo "Pipeline de datos completado"

# Entrenar modelos
train:
	python3 pipeline/train.py
	@echo "Modelos entrenados"

# Evaluar modelos
evaluate:
	python3 pipeline/evaluate.py
	@echo "Evaluación completada"

# Pipeline completo (manual)
pipeline: data train evaluate
	@echo "Pipeline completo ejecutado"

# Reproducir con DVC
dvc-repro:
	dvc repro
	@echo "Pipeline DVC reproducido"

# Visualizar DAG de DVC
dvc-dag:
	dvc dag

# Limpiar y ejecutar todo
all: clean pipeline
	@echo "Pipeline completo desde cero"
