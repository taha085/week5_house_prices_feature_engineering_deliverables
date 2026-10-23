# Week 5 — House Prices Feature Engineering and Tuned Ensembles

## Files
- `feature_engineering.py`: creates six sale-time features without using `SalePrice`.
- `tuned_models.py`: leakage-safe 5-fold CV, RandomizedSearchCV for Random Forest and XGBoost, and one final untouched hold-out evaluation.
- `MODEL_COMPARISON.md`: measured comparison table, tuning settings, feature rationales, and verdict.
- `cv_model_comparison.csv`: machine-readable cross-validation results.
- `holdout_metrics.csv`: final hold-out metrics.
- `tuning_results.json`: best sampled hyperparameters for each ensemble.

## Run
Place `cleaned_ames_train.csv` one directory above this folder, then run:

```bash
python -m pip install pandas numpy scipy scikit-learn xgboost tabulate
python tuned_models.py ../cleaned_ames_train.csv --output-dir . --iterations 10
```

`--iterations 10` means 10 randomized parameter candidates per ensemble, each assessed with 5-fold CV. Increase it for a broader search if time permits. The untouched 20% test split follows Task 3 (`random_state=42`) and is scored only after tuning is complete.

## Important interpretation
The CV and hold-out results differ: the tuned ensembles improve CV RMSE, but the Task 3 log-target Linear Regression has lower hold-out RMSE in this run. The report therefore does not recommend shipping an ensemble based on CV alone. Reviewers should judge this as evidence of split variability/model stability, not claim the ensemble wins on every metric.
