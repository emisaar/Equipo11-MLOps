# Atajos
venv:
	python -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt

run:
	python src/prepare.py && python src/featurize.py && python src/train.py && python src/evaluate.py

clean:
	rm -rf data/interim/* data/processed/* models/* reports/*.csv reports/figures/*
