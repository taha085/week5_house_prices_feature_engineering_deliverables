#!/usr/bin/env python3
"""Leakage-safe, domain-driven feature engineering for Ames House Prices."""
from __future__ import annotations
import pandas as pd

def engineer_features(X: pd.DataFrame) -> pd.DataFrame:
    """Add row-wise features computable at sale time; never uses SalePrice."""
    X = X.copy()
    if {"TotalSF", "HouseAgeAtSale", "TotalBathrooms", "HasRemodeled"}.issubset(X.columns):
        return X
    def numeric(col):
        return pd.to_numeric(X[col], errors="coerce") if col in X.columns else pd.Series(0, index=X.index, dtype=float)
    X["TotalSF"] = numeric("GrLivArea").fillna(0) + numeric("TotalBsmtSF").fillna(0)
    X["HouseAgeAtSale"] = (numeric("YrSold") - numeric("YearBuilt")).clip(lower=0)
    X["TotalBathrooms"] = (numeric("FullBath").fillna(0) + 0.5*numeric("HalfBath").fillna(0)
        + numeric("BsmtFullBath").fillna(0) + 0.5*numeric("BsmtHalfBath").fillna(0))
    X["HasRemodeled"] = (numeric("YearRemodAdd") > numeric("YearBuilt")).astype(int)
    X["GarageAgeAtSale"] = (numeric("YrSold") - numeric("GarageYrBlt")).clip(lower=0)
    porch_cols = ["OpenPorchSF", "EnclosedPorch", "3SsnPorch", "ScreenPorch", "WoodDeckSF"]
    X["TotalPorchDeckSF"] = sum((numeric(c).fillna(0) for c in porch_cols), start=pd.Series(0.0, index=X.index))
    return X

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path")
    parser.add_argument("--output", default="engineered_ames_features.csv")
    args = parser.parse_args()
    df = pd.read_csv(args.csv_path)
    target = df["SalePrice"] if "SalePrice" in df else None
    engineered = engineer_features(df.drop(columns=["SalePrice"], errors="ignore"))
    if target is not None: engineered["SalePrice"] = target
    engineered.to_csv(args.output, index=False)
    print(f"Saved {engineered.shape[0]} rows and {engineered.shape[1]} columns to {args.output}")
