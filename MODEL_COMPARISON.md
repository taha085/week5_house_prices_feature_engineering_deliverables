# Week 5 — Feature Engineering & Tuned Ensembles for Regression

## Dataset and validation
- Input: `cleaned_ames_train.csv`; rows: 1460; target: `SalePrice`; identifier columns removed.
- Task 3 split preserved: 80/20 train/hold-out, `random_state=42` (training rows: 1168; hold-out rows: 292).
- Five-fold shuffled CV on training partition only (`KFold(shuffle=True, random_state=42)`). Hold-out was not used for tuning.
- Target: `log1p(SalePrice)` during fitting; predictions inverse-transformed before scoring, so metrics are on the original SalePrice scale.
- Imputation, one-hot encoding, scaling, and feature engineering are in the estimator pipeline and refit within each CV training fold.
- scikit-learn: 1.8.0; XGBoost: 3.1.3.

## Engineered features and justification
All features use house attributes and sale/construction years, not `SalePrice`.
1. `TotalSF`: `GrLivArea + TotalBsmtSF`; summarizes above-ground and basement area.
2. `HouseAgeAtSale`: `YrSold - YearBuilt` (clipped at zero); captures age/depreciation at sale.
3. `TotalBathrooms`: full baths + 0.5 × half baths, including basement bathrooms; summarizes bathroom capacity.
4. `HasRemodeled`: 1 when `YearRemodAdd > YearBuilt`; captures post-construction remodeling.
5. `GarageAgeAtSale`: `YrSold - GarageYrBlt` (clipped at zero); approximates garage age at sale.
6. `TotalPorchDeckSF`: sum of porch and wood-deck areas; summarizes outdoor living area.
These are row-wise transformations without learned global statistics. Imputation and encoding are fitted only inside each fold. Extra features can add noise or redundancy, so their value is judged by CV rather than assumed.

## Model comparison
RMSE is mean ± sample standard deviation across five folds. Lower RMSE is better; higher R² is better.

| Model                                        |   Mean CV RMSE |   CV RMSE SD |   Mean CV R² |   Training/tuning time (sec) | Note                                                  |
|:---------------------------------------------|---------------:|-------------:|-------------:|-----------------------------:|:------------------------------------------------------|
| Tuned XGBoost                                |      26066.859 |     3483.602 |        0.886 |                       37.480 | RandomizedSearchCV; 10 draws, 5-fold CV.              |
| Tuned Random Forest                          |      29557.785 |     5413.058 |        0.851 |                      193.181 | RandomizedSearchCV; 10 draws, 5-fold CV.              |
| Task 3 log-target Linear Regression baseline |      48891.240 |    33759.732 |        0.483 |                        0.436 | Task 3 reference; log target, no engineered features. |
| Engineered log-target Linear Regression      |      49049.307 |    34040.799 |        0.478 |                        0.268 | Checks whether domain features help a simple model.   |

## Final untouched hold-out check
Models were refit on the training partition and evaluated once on the untouched 20% hold-out.

| Model                                        |   Hold-out RMSE |   Hold-out MAE |   Hold-out R² |
|:---------------------------------------------|----------------:|---------------:|--------------:|
| Task 3 log-target Linear Regression baseline |       22950.831 |      14808.453 |         0.931 |
| Engineered log-target Linear Regression      |       22966.798 |      14814.432 |         0.931 |
| Tuned XGBoost                                |       27018.486 |      16122.082 |         0.905 |
| Tuned Random Forest                          |       30132.279 |      17029.988 |         0.882 |

## Tuning setup
- `RandomizedSearchCV`, 5-fold CV, `random_state=42`, `n_iter=10` for each ensemble.
- Random Forest searched 6 hyperparameters: `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`, `max_features`, `bootstrap`.
- XGBoost searched 7 hyperparameters: `n_estimators`, `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`, `min_child_weight`, `reg_lambda`.
- Best sampled parameters:

```json
{
  "Tuned Random Forest": {
    "best_params": {
      "regressor__model__bootstrap": true,
      "regressor__model__max_depth": 24,
      "regressor__model__max_features": 0.6428245137802029,
      "regressor__model__min_samples_leaf": 2,
      "regressor__model__min_samples_split": 5,
      "regressor__model__n_estimators": 288
    },
    "best_cv_rmse": 29557.784614964872,
    "search_seconds": 193.18104409900002,
    "n_iter": 10
  },
  "Tuned XGBoost": {
    "best_params": {
      "regressor__model__colsample_bytree": 0.7443752583889521,
      "regressor__model__learning_rate": 0.07242124523564754,
      "regressor__model__max_depth": 4,
      "regressor__model__min_child_weight": 3,
      "regressor__model__n_estimators": 816,
      "regressor__model__reg_lambda": 8.365847086454306,
      "regressor__model__subsample": 0.813367012636793
    },
    "best_cv_rmse": 26066.859375,
    "search_seconds": 37.48024619700004,
    "n_iter": 10
  }
}
```

## Verdict
Best mean CV RMSE is **Tuned XGBoost** (26,067 ± 3,484). That is a 46.68% RMSE reduction versus the Task 3 log-target Linear Regression baseline (48,891). Treat small changes relative to fold-to-fold variability cautiously. The lowest hold-out RMSE is Task 3 log-target Linear Regression baseline (22,951); the hold-out was not used for tuning. Ship the simpler model unless ensemble CV and hold-out results show a practically meaningful, consistent gain that justifies added training cost.

## Reproducibility
Install dependencies (`pandas numpy scipy scikit-learn xgboost tabulate`) and run:
```bash
python tuned_models.py ../cleaned_ames_train.csv --output-dir . --iterations 10
```
