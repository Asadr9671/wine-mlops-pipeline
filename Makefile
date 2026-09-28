# Wine MLOps pipeline - uniform entry points for local dev and CI
PYTHON ?= python
VENV   ?= .venv

.PHONY: install lint test train evaluate clean all

install:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt

lint:
	$(PYTHON) -m flake8 src/ tests/ --max-line-length=100

test:
	$(PYTHON) -m pytest tests/ -v

train:
	$(PYTHON) -m src.train

evaluate:
	$(PYTHON) -m src.evaluate

all: install lint test train evaluate

clean:
	find . -type f -name "*.pyc" -not -path "./$(VENV)/*" -delete
	find . -type d -name "__pycache__" -not -path "./$(VENV)/*" -prune -exec rm -rf {} +
	rm -rf .pytest_cache reports
	rm -f *.tmp *.log
