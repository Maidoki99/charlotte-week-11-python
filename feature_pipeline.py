"""Week 11 Class 5 Capstone — General and Industrial Data Wrangling & Feature Engineering."""
from __future__ import annotations

import json
from pathlib import Path
import lasio
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_DIR = Path(__file__).resolve().parent / "data"
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"

GENERAL_REQUIRED_COLS = {
    "sample_id",
    "mean_radius",
    "mean_texture",
    "mean_perimeter",
    "mean_area",
    "lab_site",
    "instrument",
}
GENERAL_NUMERIC_COLS = ["mean_radius", "mean_texture", "mean_perimeter", "mean_area"]
GENERAL_CAT_COLS = ["lab_site", "region", "instrument"]

SCADA_REQUIRED_COLS = {
    "timestamp",
    "well_id",
    "depth_m",
    "pressure_psi",
    "temperature_f",
    "flow_bpd",
}
PSI_TO_KPA = 6.894757


def _flag_iqr_outliers(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    """Flag rows outside [Q1 - 1.5*IQR, Q3 + 1.5*IQR] on valid positive observations."""
    valid_mask = ~(df["mean_area"] <= 0)
    outlier_any = pd.Series(False, index=df.index)
    for col in cols:
        s = df.loc[valid_mask, col].dropna()
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        col_out = df[col].notna() & ((df[col] < lower) | (df[col] > upper))
        outlier_any = outlier_any | col_out
    return outlier_any


def _compute_general_ratios(num_df: pd.DataFrame) -> pd.DataFrame:
    """Compute guarded geometric ratios from imputed positive measurements."""
    out = num_df.copy()
    radius_sq = out["mean_radius"] ** 2
    out["area_per_radius_sq"] = np.where(
        out["mean_radius"].notna() & (radius_sq > 0),
        out["mean_area"] / radius_sq,
        np.nan,
    )
    out["texture_perimeter_ratio"] = np.where(
        out["mean_perimeter"].notna() & (out["mean_perimeter"] > 0),
        out["mean_texture"] / out["mean_perimeter"],
        np.nan,
    )
    return out


def run_general(
    input_csv: Path | str = DATA_DIR / "general_measurements_PUBLIC_DERIVED.csv",
    metadata_json: Path | str = DATA_DIR / "site_metadata_SYNTHETIC.json",
    out_dir: Path | str = OUTPUT_DIR,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Validate, engineer, split 80/20, and fit leakage-free preprocessing on train only."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_csv)
    missing_cols = GENERAL_REQUIRED_COLS - set(df.columns)
    if missing_cols:
        raise ValueError(f"General CSV missing required columns: {sorted(missing_cols)}")

    df = df.copy()
    df["sample_id"] = df["sample_id"].astype(str).str.strip()
    df["lab_site"] = df["lab_site"].astype(str).str.strip()
    df["instrument"] = df["instrument"].astype(str).str.strip()
    for col in GENERAL_NUMERIC_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if df["sample_id"].isna().any() or df["sample_id"].duplicated().any():
        raise ValueError("General CSV contains null or duplicate sample_id keys.")

    raw_meta = json.loads(Path(metadata_json).read_text(encoding="utf-8"))
    sites_list = raw_meta.get("sites") if isinstance(raw_meta, dict) else raw_meta
    meta_df = pd.DataFrame(sites_list)
    if not {"lab_site", "region"}.issubset(meta_df.columns):
        raise ValueError("Site metadata JSON must define lab_site and region.")
    meta_df["lab_site"] = meta_df["lab_site"].astype(str).str.strip()
    meta_df = meta_df.drop_duplicates(subset=["lab_site"])

    merged = df.merge(meta_df[["lab_site", "region"]], on="lab_site", how="left", validate="many_to_one")
    merged["unmatched_site_flag"] = merged["region"].isna().astype(int)
    merged["region"] = merged["region"].fillna("UNMAPPED_SITE")

    invalid_area_mask = merged["mean_area"].notna() & (merged["mean_area"] <= 0)
    missing_any_mask = merged[GENERAL_NUMERIC_COLS].isna().any(axis=1)
    outlier_mask = _flag_iqr_outliers(merged, GENERAL_NUMERIC_COLS)

    quality_report = pd.DataFrame({
        "sample_id": merged["sample_id"],
        "lab_site": merged["lab_site"],
        "region": merged["region"],
        "status": np.where(invalid_area_mask, "excluded_invalid_mean_area", "included"),
        "invalid_mean_area_flag": invalid_area_mask.astype(int),
        "missing_value_flag": missing_any_mask.astype(int),
        "missing_columns": merged[GENERAL_NUMERIC_COLS].apply(
            lambda r: ";".join([c for c in GENERAL_NUMERIC_COLS if pd.isna(r[c])]) or "none",
            axis=1,
        ),
        "iqr_outlier_candidate_flag": outlier_mask.astype(int),
        "unmatched_site_flag": merged["unmatched_site_flag"],
    })

    valid_df = merged.loc[~invalid_area_mask].reset_index(drop=True)
    if (valid_df["mean_area"].dropna() <= 0).any():
        raise AssertionError("Non-positive mean_area remained in valid observations.")

    train_raw, test_raw = train_test_split(
        valid_df,
        test_size=0.20,
        random_state=42,
        shuffle=True,
    )
    train_raw = train_raw.reset_index(drop=True)
    test_raw = test_raw.reset_index(drop=True)

    # Fit numeric median imputer on TRAINING data only
    num_imputer = SimpleImputer(strategy="median")
    train_num_imp = pd.DataFrame(
        num_imputer.fit_transform(train_raw[GENERAL_NUMERIC_COLS]),
        columns=GENERAL_NUMERIC_COLS,
    )
    test_num_imp = pd.DataFrame(
        num_imputer.transform(test_raw[GENERAL_NUMERIC_COLS]),
        columns=GENERAL_NUMERIC_COLS,
    )

    train_num_feat = _compute_general_ratios(train_num_imp)
    test_num_feat = _compute_general_ratios(test_num_imp)

    engineered_num_cols = GENERAL_NUMERIC_COLS + ["area_per_radius_sq", "texture_perimeter_ratio"]
    train_pre_df = pd.concat([train_num_feat, train_raw[GENERAL_CAT_COLS].reset_index(drop=True)], axis=1)
    test_pre_df = pd.concat([test_num_feat, test_raw[GENERAL_CAT_COLS].reset_index(drop=True)], axis=1)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), engineered_num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), GENERAL_CAT_COLS),
        ],
        remainder="drop",
    )

    X_train = preprocessor.fit_transform(train_pre_df)
    X_test = preprocessor.transform(test_pre_df)
    feature_names = list(preprocessor.get_feature_names_out())

    train_out = pd.DataFrame(X_train, columns=feature_names)
    train_out.insert(0, "sample_id", train_raw["sample_id"].values)

    test_out = pd.DataFrame(X_test, columns=feature_names)
    test_out.insert(0, "sample_id", test_raw["sample_id"].values)

    assert list(train_out.columns) == list(test_out.columns), "Train/test feature columns do not match."
    assert not train_out.isna().any().any(), "Missing values detected in processed train matrix."
    assert not test_out.isna().any().any(), "Missing values detected in processed test matrix."

    schema_doc = {
        "provenance": {
            "measurements_source": "Public-derived subset (first 180 rows, 4 geometric features) from scikit-learn load_breast_cancer (UCI WDBC); clinical diagnosis labels intentionally omitted.",
            "synthetic_alterations": "Synthetic sample_id, lab_site, instrument, site_metadata_SYNTHETIC.json, and injected missing/non-positive values for educational data wrangling only.",
            "clinical_disclaimer": "Educational preprocessing artifact only; must not be used for medical diagnosis or clinical decisions.",
        },
        "split": {
            "strategy": "80/20 train_test_split",
            "random_state": 42,
            "total_input_rows": int(len(merged)),
            "excluded_invalid_rows": int(invalid_area_mask.sum()),
            "valid_rows": int(len(valid_df)),
            "train_rows": int(len(train_out)),
            "test_rows": int(len(test_out)),
        },
        "cleaning_decisions": {
            "invalid_mean_area": "Rows with mean_area <= 0 (SMP-0008, mean_area=-15.0) are excluded from train/test matrices and recorded in general_quality_report.csv.",
            "missing_values": "Valid rows with missing numeric measurements are retained and imputed using training-set medians only.",
            "outliers": "1.5*IQR outlier candidates are flagged in general_quality_report.csv and retained rather than blindly deleted.",
            "unmatched_site_rule": "Left join on lab_site; unmatched keys map to region='UNMAPPED_SITE' with unmatched_site_flag=1.",
        },
        "engineered_features": {
            "area_per_radius_sq": "mean_area / (mean_radius ** 2) after training-only median imputation, guarded for mean_radius > 0.",
            "texture_perimeter_ratio": "mean_texture / mean_perimeter after training-only median imputation, guarded for mean_perimeter > 0.",
        },
        "processed_columns": list(train_out.columns),
    }

    train_out.to_csv(out_path / "general_train_features.csv", index=False)
    test_out.to_csv(out_path / "general_test_features.csv", index=False)
    quality_report.to_csv(out_path / "general_quality_report.csv", index=False)
    (out_path / "general_feature_schema.json").write_text(json.dumps(schema_doc, indent=2), encoding="utf-8")

    return train_out, test_out, quality_report, schema_doc


def run_industrial(
    las_path: Path | str = DATA_DIR / "well_log_SYNTHETIC.las",
    scada_csv: Path | str = DATA_DIR / "scada_SYNTHETIC.csv",
    out_dir: Path | str = OUTPUT_DIR,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read LAS with lasio and SCADA CSV with pandas; align depths, convert units, and engineer features."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    las = lasio.read(str(las_path))
    curve_units = {c.mnemonic.strip().upper(): (c.unit or "").strip().upper() for c in las.curves}
    required_curves = {"DEPT", "GR", "RESD", "TEMP"}
    missing_curves = required_curves - set(curve_units.keys())
    if missing_curves:
        raise ValueError(f"LAS file missing required curves: {sorted(missing_curves)}")
    if curve_units["DEPT"] and curve_units["DEPT"] not in {"M", "METERS", "METRES"}:
        raise ValueError(f"Unexpected LAS depth unit: {curve_units['DEPT']}")

    well_item = las.well.get("WELL") if hasattr(las, "well") else None
    las_well_id = str(well_item.value).strip() if well_item is not None and str(well_item.value).strip() else "UNKNOWN_WELL"

    las_raw = las.df().reset_index()
    las_raw.columns = [str(c).strip().upper() for c in las_raw.columns]
    las_df = pd.DataFrame({
        "well_id": las_well_id,
        "depth_m": pd.to_numeric(las_raw["DEPT"], errors="coerce"),
        "las_gamma_api": pd.to_numeric(las_raw["GR"], errors="coerce"),
        "las_resistivity_ohmm": pd.to_numeric(las_raw["RESD"], errors="coerce"),
        "las_temp_degc": pd.to_numeric(las_raw["TEMP"], errors="coerce"),
    })
    if las_df[["well_id", "depth_m"]].isna().any().any() or las_df.duplicated(["well_id", "depth_m"]).any():
        raise ValueError("LAS table contains null or duplicate (well_id, depth_m) keys.")

    scada_df = pd.read_csv(scada_csv)
    missing_scada = SCADA_REQUIRED_COLS - set(scada_df.columns)
    if missing_scada:
        raise ValueError(f"SCADA CSV missing required columns: {sorted(missing_scada)}")

    scada_df = scada_df.copy()
    scada_df["well_id"] = scada_df["well_id"].astype(str).str.strip()
    scada_df["timestamp"] = pd.to_datetime(scada_df["timestamp"], errors="raise")
    for col in ["depth_m", "pressure_psi", "temperature_f", "flow_bpd"]:
        scada_df[col] = pd.to_numeric(scada_df[col], errors="coerce")

    if scada_df[["well_id", "timestamp", "depth_m"]].isna().any().any():
        raise ValueError("SCADA CSV contains null well_id, timestamp, or depth_m keys.")
    if scada_df.duplicated(["well_id", "timestamp"]).any():
        raise ValueError("SCADA CSV contains duplicate (well_id, timestamp) keys.")

    scada_sorted = scada_df.sort_values(["well_id", "timestamp"]).reset_index(drop=True)
    merged = scada_sorted.merge(
        las_df,
        on=["well_id", "depth_m"],
        how="left",
        validate="many_to_one",
    )

    merged["las_depth_matched_flag"] = merged["las_temp_degc"].notna().astype(int)
    merged["pressure_missing_flag"] = merged["pressure_psi"].isna().astype(int)
    merged["pressure_kpa"] = merged["pressure_psi"] * PSI_TO_KPA
    merged["temperature_c"] = (merged["temperature_f"] - 32.0) * (5.0 / 9.0)
    merged["temp_diff_from_las_c"] = merged["temperature_c"] - merged["las_temp_degc"]

    grouped = merged.groupby("well_id", sort=False)
    merged["pressure_kpa_var_3"] = grouped["pressure_kpa"].transform(
        lambda s: s.rolling(window=3, min_periods=2).var(ddof=0)
    )
    # Cumulative sum of instantaneous sampled flow_bpd readings (cumulative sampled indicator, NOT volume produced)
    merged["cumulative_sampled_flow_bpd"] = grouped["flow_bpd"].transform(
        lambda s: s.fillna(0.0).cumsum()
    )

    merged["timestamp"] = merged["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%S")

    unmatched_depths = sorted(merged.loc[merged["las_depth_matched_flag"] == 0, "depth_m"].unique().tolist())
    quality_report = pd.DataFrame([
        {"metric": "las_depth_rows", "value": str(len(las_df))},
        {"metric": "scada_input_rows", "value": str(len(scada_sorted))},
        {"metric": "matched_las_rows", "value": str(int(merged["las_depth_matched_flag"].sum()))},
        {"metric": "unmatched_las_rows", "value": str(int((merged["las_depth_matched_flag"] == 0).sum()))},
        {"metric": "unmatched_depths_m", "value": ";".join(str(d) for d in unmatched_depths) or "none"},
        {"metric": "missing_pressure_rows", "value": str(int(merged["pressure_missing_flag"].sum()))},
        {"metric": "output_rows", "value": str(len(merged))},
        {"metric": "output_columns", "value": str(len(merged.columns))},
    ])

    merged.to_csv(out_path / "industrial_features.csv", index=False)
    quality_report.to_csv(out_path / "industrial_quality_report.csv", index=False)
    return merged, quality_report


def main() -> None:
    train_df, test_df, gen_qa, _ = run_general()
    ind_df, ind_qa = run_industrial()
    print(f"Part A complete: train={train_df.shape}, test={test_df.shape}, audit_rows={len(gen_qa)}")
    print(f"Part B complete: industrial={ind_df.shape}")
    print(ind_qa.to_string(index=False))


if __name__ == "__main__":
    main()
