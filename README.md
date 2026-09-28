# Wine MLOps Pipeline

![CI](https://github.com/Asadr9671/wine-mlops-pipeline/actions/workflows/ci.yml/badge.svg)

Reproducible MLOps pipeline for `sklearn.datasets.load_wine` (178 samples, 13 features, 3 classes):
Makefile automation, MLflow tracking + Model Registry, and a GitHub Actions quality gate.
All splits and model initialisations use `seed=42`.

## Quick start
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
make install     # upgrade pip, install pinned requirements
make lint        # flake8, max line length 100
make test        # pytest -v (unit tests + model quality gate)
make train       # 8 MLflow runs (4 RF + 4 GBM), registers WineClassifier@champion
make evaluate    # loads the champion and scores the held-out test split
make clean       # remove bytecode / caches
```
View experiments: `mlflow ui --backend-store-uri sqlite:///mlflow.db` then open http://127.0.0.1:5000

## Structure
| Path | Purpose |
|------|---------|
| `src/data.py` | Load Wine data, validation (no nulls, 13 features), stratified 80/20 split |
| `src/train.py` | RF + GBM grids, 5-fold stratified CV, MLflow runs, signatures, registry, alias |
| `src/evaluate.py` | Loads `models:/WineClassifier@champion`, computes test metrics |
| `tests/test_data.py` | Data pipeline unit tests |
| `tests/test_model_gate.py` | Quality gate: macro F1 >= 0.88, latency <= 30 ms, labels in {0,1,2} |
| `.github/workflows/ci.yml` | CI on PR to `main` and push to `main` (Python 3.10) |

## MLflow
* Experiment: `Wine-Cultivar-Classification` (SQLite backend `mlflow.db`)
* One run per hyperparameter configuration, with params, train/val macro F1, accuracy, log loss, tags
* Best run by validation macro F1 is registered as `WineClassifier` with alias `champion`

## Configuration
Cross-validation folds: 5
