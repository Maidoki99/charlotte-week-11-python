"""Week 11 Class 3 student lab: implement a leakage-safe feature pipeline.
Run: python preprocess_iris.py --input iris_preprocessing_case_PUBLIC_DERIVED.csv --output-dir outputs
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC = ["sepal_length_cm", "sepal_width_cm", "petal_length_cm", "petal_width_cm"]
CATEGORICAL = ["collection_site", "instrument_type"]
LABEL = "species"
ID_COL = "sample_id"


def build_preprocessor() -> ColumnTransformer:
    """Return a ColumnTransformer with numeric imputation+scaling and category imputation+one-hot.
    Make unseen categories transformable without an exception.
    """
    numeric_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC),
            ("cat", categorical_pipeline, CATEGORICAL),
        ],
        sparse_threshold=0,
    )


def prepare_dataset(input_csv: Path | str) -> tuple[ColumnTransformer, pd.DataFrame, pd.DataFrame, dict[str, int]]:
    """Load CSV, split features/target before fitting, then return (transformer, train_df, test_df, report_dict).
    Split must be deterministic: test_size=0.2, random_state=42, stratify=species.
    Output DataFrames must have matching columns, numeric finite values, and no species/sample_id.
    Report keys: source_rows, training_rows, test_rows, transformed_features, missing_source_cells.
    """
    df = pd.read_csv(input_csv)
    feature_cols = NUMERIC + CATEGORICAL
    required_cols = [ID_COL, *feature_cols, LABEL]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required column(s): {missing_cols}")

    missing_source_cells = int(df[feature_cols].isna().sum().sum())

    X = df[feature_cols].copy()
    y = df[LABEL].copy()

    X_train, X_test, _, _ = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    transformer = build_preprocessor()
    X_train_transformed = np.asarray(transformer.fit_transform(X_train), dtype=float)
    X_test_transformed = np.asarray(transformer.transform(X_test), dtype=float)

    if not np.isfinite(X_train_transformed).all() or not np.isfinite(X_test_transformed).all():
        raise ValueError("Transformed features contain non-finite values (NaN or Inf).")

    feature_names = list(transformer.get_feature_names_out())
    train_df = pd.DataFrame(X_train_transformed, columns=feature_names)
    test_df = pd.DataFrame(X_test_transformed, columns=feature_names)

    report = {
        "source_rows": int(len(df)),
        "training_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "transformed_features": int(len(feature_names)),
        "missing_source_cells": missing_source_cells,
    }
    return transformer, train_df, test_df, report


def write_outputs(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    report: dict[str, int],
    output_dir: Path | str,
) -> None:
    """Write train_features.csv, test_features.csv, data_quality_summary.csv."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(out_path / "train_features.csv", index=False)
    test_df.to_csv(out_path / "test_features.csv", index=False)

    summary_df = pd.DataFrame([
        {"metric": key, "value": value} for key, value in report.items()
    ])
    summary_df.to_csv(out_path / "data_quality_summary.csv", index=False)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="iris_preprocessing_case_PUBLIC_DERIVED.csv")
    p.add_argument("--output-dir", default="outputs")
    args = p.parse_args()
    _, train, test, report = prepare_dataset(args.input)
    write_outputs(train, test, report, args.output_dir)
    print("Preprocessing completed")


if __name__ == "__main__":
    main()
