PYTHON ?= python3
PYTHONPATH := src
export PYTHONPATH

.PHONY: install download prepare test notebooks

install:
	$(PYTHON) -m pip install -r requirements.txt

download:
	$(PYTHON) -m house_prices.download

prepare: download
	$(PYTHON) -m house_prices.prepare

test:
	$(PYTHON) -m pytest

notebooks:
	$(PYTHON) -m nbconvert --to notebook --execute --inplace notebooks/01_exploratory_analysis.ipynb
	$(PYTHON) -m nbconvert --to notebook --execute --inplace notebooks/02_price_prediction.ipynb
