# Git workflow & submission checklist (Milestone 5 + report)

Replace `<USER>` with your GitHub username. **Do everything below yourself so the history is genuinely yours**
(the integrity rule penalises copied histories). Commit in small steps with your own messages.

## 1. Setup (main stays clean)

```bash
git init -b main
git add .gitignore README.md Makefile requirements.txt data/.gitkeep
git commit -m "chore: add Makefile, requirements and gitignore"
git remote add origin https://github.com/Asadr9671/wine-mlops-pipeline.git
git push -u origin main
```

Then push the rest through feature branches and Pull Requests (each PR triggers CI):

```bash
git checkout -b feature/data-pipeline
git add src/__init__.py src/data.py tests/__init__.py tests/test_data.py
git commit -m "feat: data loading, validation and stratified split"
git push -u origin feature/data-pipeline        # open PR -> merge on GitHub once CI is green

git checkout main && git pull
git checkout -b feature/mlflow-tracking
git add src/train.py src/evaluate.py
git commit -m "feat: RF/GBM CV tuning with MLflow tracking and registry"
git push -u origin feature/mlflow-tracking      # PR -> merge

git checkout main && git pull
git checkout -b feature/ci-quality-gate
git add .github tests/test_model_gate.py
git commit -m "ci: GitHub Actions workflow and model quality gate"
git push -u origin feature/ci-quality-gate      # PR -> merge
```

Note: CI runs `make test`, which trains the pipeline inside the gate test, so the gate PR needs `src/` merged first.

## 2. Engineered merge conflict (CV_FOLDS line in src/train.py)

```bash
git checkout main && git pull
git checkout -b conflict-simulation
sed -i 's/^CV_FOLDS = 5$/CV_FOLDS = 10/' src/train.py     # macOS: sed -i '' ...
git commit -am "config: use 10 CV folds"

git checkout main
sed -i 's/^CV_FOLDS = 5$/CV_FOLDS = 3/' src/train.py
git commit -am "config: use 3 CV folds"

git merge conflict-simulation      # -> CONFLICT (content): Merge conflict in src/train.py
git status
```

Open `src/train.py`, delete the markers, keep the correct line:

```
<<<<<<< HEAD
CV_FOLDS = 3
=======
CV_FOLDS = 10
>>>>>>> conflict-simulation
```

becomes `CV_FOLDS = 5` (the assignment requires 5-fold CV). Then:

```bash
git add src/train.py
git commit -m "Merge conflict-simulation into main: resolve CV_FOLDS conflict (keep 5)"
git push origin main
git log --oneline --graph --all      # screenshot for the report
```

Copy the terminal output of the whole sequence (merge attempt, `git status`, resolution, log) into the report.

## 3. Report (MLOps_A01_RollNumber.pdf) checklist

- [ ] Table 1: hyperparameter results (from `reports/table1_hyperparameter_results.csv` after `make train`)
- [ ] MLflow UI: experiment overview, a comparison plot (select runs -> Compare), Models tab showing `WineClassifier` with alias `champion`
- [ ] Green GitHub Actions run screenshot
- [ ] `git log --oneline --graph --all` output + conflict terminal logs
- [ ] Analysis, max 150 words (use your own numbers; talking points: RF beat GBM on this small dataset; train F1 of 1.0 vs ~0.97 validation shows mild overfitting; deeper/larger RF gave the best validation log loss; GBM was more sensitive to learning-rate/depth; champion also scored well on the held-out test split)
