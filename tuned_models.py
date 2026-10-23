#!/usr/bin/env python3
"""Week 5: leakage-safe CV, randomized tuning, and one final hold-out evaluation."""
from __future__ import annotations
import argparse, json, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import sklearn
from scipy.stats import randint, uniform
from sklearn.base import clone
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, KFold, cross_validate, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from feature_engineering import engineer_features

def make_preprocessor(X):
    numeric = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical = [c for c in X.columns if c not in numeric]
    num_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    cat_pipe = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                         ("onehot", OneHotEncoder(handle_unknown="ignore"))])
    return ColumnTransformer([("numeric", num_pipe, numeric), ("categorical", cat_pipe, categorical)], remainder="drop")

def make_estimator(model, preprocessor, use_features=True):
    steps = []
    if use_features:
        steps.append(("features", FunctionTransformer(engineer_features, validate=False)))
    steps.extend([("preprocess", clone(preprocessor)), ("model", model)])
    return TransformedTargetRegressor(regressor=Pipeline(steps), func=np.log1p, inverse_func=np.expm1, check_inverse=False)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    parser.add_argument("--iterations", type=int, default=10)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.csv_path)
    if "SalePrice" not in df: raise ValueError("CSV must contain SalePrice")
    df = df.loc[df["SalePrice"].notna()].copy()
    y = pd.to_numeric(df["SalePrice"], errors="raise")
    X = df.drop(columns=["SalePrice"])
    X = X.drop(columns=[c for c in X.columns if c.lower() in {"id", "parcelid", "parcel_id"}], errors="ignore")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    preprocessor = make_preprocessor(X_train)
    engineered_preprocessor = make_preprocessor(engineer_features(X_train))
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    models = {}

    baseline = make_estimator(LinearRegression(), preprocessor, use_features=False)
    t0 = time.perf_counter()
    base_cv = cross_validate(baseline, X_train, y_train, cv=cv,
        scoring={"rmse":"neg_root_mean_squared_error", "r2":"r2"}, n_jobs=1)
    models["Task 3 log-target Linear Regression baseline"] = {
        "estimator":baseline, "rmse_mean":-base_cv["test_rmse"].mean(),
        "rmse_std":base_cv["test_rmse"].std(ddof=1), "r2_mean":base_cv["test_r2"].mean(),
        "time":time.perf_counter()-t0, "note":"Task 3 reference; log target, no engineered features."}

    eng_linear = make_estimator(LinearRegression(), engineered_preprocessor, use_features=True)
    t0 = time.perf_counter()
    eng_cv = cross_validate(eng_linear, X_train, y_train, cv=cv,
        scoring={"rmse":"neg_root_mean_squared_error", "r2":"r2"}, n_jobs=1)
    models["Engineered log-target Linear Regression"] = {
        "estimator":eng_linear, "rmse_mean":-eng_cv["test_rmse"].mean(),
        "rmse_std":eng_cv["test_rmse"].std(ddof=1), "r2_mean":eng_cv["test_r2"].mean(),
        "time":time.perf_counter()-t0, "note":"Checks whether domain features help a simple model."}

    rf_est = make_estimator(RandomForestRegressor(random_state=42, n_jobs=1), engineered_preprocessor, True)
    rf_params = {
        "regressor__model__n_estimators":randint(200,601),
        "regressor__model__max_depth":[None,8,12,16,24,32],
        "regressor__model__min_samples_split":randint(2,16),
        "regressor__model__min_samples_leaf":randint(1,9),
        "regressor__model__max_features":uniform(0.35,0.65),
        "regressor__model__bootstrap":[True,False]}
    xgb_est = make_estimator(XGBRegressor(random_state=42, objective="reg:squarederror",
        eval_metric="rmse", n_jobs=1, tree_method="hist", verbosity=0), engineered_preprocessor, True)
    xgb_params = {
        "regressor__model__n_estimators":randint(250,901),
        "regressor__model__max_depth":randint(2,9),
        "regressor__model__learning_rate":uniform(0.02,0.18),
        "regressor__model__subsample":uniform(0.65,0.35),
        "regressor__model__colsample_bytree":uniform(0.55,0.45),
        "regressor__model__min_child_weight":randint(1,10),
        "regressor__model__reg_lambda":uniform(0.5,8.0)}
    search_results = {}
    for name, estimator, params in [("Tuned Random Forest",rf_est,rf_params),("Tuned XGBoost",xgb_est,xgb_params)]:
        search = RandomizedSearchCV(estimator, params, n_iter=args.iterations,
            scoring="neg_root_mean_squared_error", cv=cv, random_state=42, n_jobs=1,
            refit=True, verbose=1, return_train_score=False)
        t0=time.perf_counter()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            search.fit(X_train,y_train)
        elapsed=time.perf_counter()-t0
        idx=search.best_index_
        fold_rmse=np.array([-search.cv_results_[f"split{i}_test_score"][idx] for i in range(5)])
        r2_scores=cross_validate(search.best_estimator_,X_train,y_train,cv=cv,scoring="r2",n_jobs=1)["test_score"]
        models[name]={"estimator":search.best_estimator_,"rmse_mean":float(fold_rmse.mean()),
            "rmse_std":float(fold_rmse.std(ddof=1)),"r2_mean":float(r2_scores.mean()),
            "time":elapsed,"note":f"RandomizedSearchCV; {args.iterations} draws, 5-fold CV."}
        search_results[name]={"best_params":search.best_params_,"best_cv_rmse":float(-search.best_score_),
            "search_seconds":elapsed,"n_iter":args.iterations}

    comparison=[]
    holdout_rows=[]
    for name,result in models.items():
        comparison.append({"Model":name,"Mean CV RMSE":result["rmse_mean"],"CV RMSE SD":result["rmse_std"],
            "Mean CV R²":result["r2_mean"],"Training/tuning time (sec)":result["time"],"Note":result["note"]})
        fitted=clone(result["estimator"])
        fitted.fit(X_train,y_train)
        pred=fitted.predict(X_test)
        holdout_rows.append({"Model":name,"Hold-out RMSE":float(np.sqrt(mean_squared_error(y_test,pred))),
            "Hold-out MAE":float(mean_absolute_error(y_test,pred)),"Hold-out R²":float(r2_score(y_test,pred))})
    comp=pd.DataFrame(comparison).sort_values("Mean CV RMSE")
    holdout=pd.DataFrame(holdout_rows).sort_values("Hold-out RMSE")
    comp.to_csv(args.output_dir/"cv_model_comparison.csv",index=False)
    holdout.to_csv(args.output_dir/"holdout_metrics.csv",index=False)
    (args.output_dir/"tuning_results.json").write_text(json.dumps(search_results,indent=2,default=str),encoding="utf-8")
    baseline_rmse=models["Task 3 log-target Linear Regression baseline"]["rmse_mean"]
    best=comp.iloc[0]
    reduction=(baseline_rmse-float(best["Mean CV RMSE"]))/baseline_rmse*100
    best_hold=holdout.iloc[0]
    verdict=(f"Best mean CV RMSE is **{best['Model']}** ({best['Mean CV RMSE']:,.0f} ± {best['CV RMSE SD']:,.0f}). "
        f"That is a {reduction:.2f}% RMSE reduction versus the Task 3 log-target Linear Regression baseline "
        f"({baseline_rmse:,.0f}). Treat small changes relative to fold-to-fold variability cautiously. "
        f"The lowest hold-out RMSE is {best_hold['Model']} ({best_hold['Hold-out RMSE']:,.0f}); "
        "the hold-out was not used for tuning. Ship the simpler model unless ensemble CV and hold-out results "
        "show a practically meaningful, consistent gain that justifies added training cost.")
    report=f"""# Week 5 — Feature Engineering & Tuned Ensembles for Regression

## Dataset and validation
- Input: `{args.csv_path.name}`; rows: {len(df)}; target: `SalePrice`; identifier columns removed.
- Task 3 split preserved: 80/20 train/hold-out, `random_state=42` (training rows: {len(X_train)}; hold-out rows: {len(X_test)}).
- Five-fold shuffled CV on training partition only (`KFold(shuffle=True, random_state=42)`). Hold-out was not used for tuning.
- Target: `log1p(SalePrice)` during fitting; predictions inverse-transformed before scoring, so metrics are on the original SalePrice scale.
- Imputation, one-hot encoding, scaling, and feature engineering are in the estimator pipeline and refit within each CV training fold.
- scikit-learn: {sklearn.__version__}; XGBoost: {__import__("xgboost").__version__}.

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

{comp.to_markdown(index=False, floatfmt=".3f")}

## Final untouched hold-out check
Models were refit on the training partition and evaluated once on the untouched 20% hold-out.

{holdout.to_markdown(index=False, floatfmt=".3f")}

## Tuning setup
- `RandomizedSearchCV`, 5-fold CV, `random_state=42`, `n_iter={args.iterations}` for each ensemble.
- Random Forest searched 6 hyperparameters: `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`, `max_features`, `bootstrap`.
- XGBoost searched 7 hyperparameters: `n_estimators`, `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`, `min_child_weight`, `reg_lambda`.
- Best sampled parameters:

```json
{json.dumps(search_results, indent=2, default=str)}
```

## Verdict
{verdict}

## Reproducibility
Install dependencies (`pandas numpy scipy scikit-learn xgboost tabulate`) and run:
```bash
python tuned_models.py ../cleaned_ames_train.csv --output-dir . --iterations 10
```
"""
    (args.output_dir/"MODEL_COMPARISON.md").write_text(report,encoding="utf-8")
    print("\nCV comparison:\n",comp.to_string(index=False))
    print("\nUntouched hold-out:\n",holdout.to_string(index=False))
    print(f"\nWrote files to {args.output_dir.resolve()}")

if __name__=="__main__": main()
