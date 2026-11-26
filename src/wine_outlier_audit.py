"""Week 11 Class 2 — Wine Chemistry Outlier Audit."""
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA = PROJECT_ROOT / "data" / "wine_quality_case_PUBLIC_DERIVED.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs"

REQUIRED_COLUMNS = ["sample_id", "alcohol", "malic_acid", "color_intensity", "proline", "target"]
TARGET_METRICS = ["malic_acid", "proline"]


def load_and_validate(path: Path | str) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required column(s): {missing}")
    return df


def compute_z_flags(series: pd.Series, threshold: float = 3.0) -> tuple[pd.Series, pd.Series]:
    values = pd.to_numeric(series, errors="coerce").astype(float)
    mean = values.mean()
    sd = values.std(ddof=0)
    if pd.isna(sd) or sd == 0.0:
        return pd.Series(0.0, index=series.index), pd.Series(False, index=series.index)
    z_scores = (values - mean) / sd
    flags = (z_scores.abs() > threshold).fillna(False)
    return z_scores, flags


def compute_iqr_flags(series: pd.Series, multiplier: float = 1.5) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").astype(float)
    q1 = values.quantile(0.25)
    q3 = values.quantile(0.75)
    iqr = q3 - q1
    if pd.isna(iqr) or iqr == 0.0:
        return pd.Series(False, index=series.index)
    lower = q1 - multiplier * iqr
    upper = q3 + multiplier * iqr
    return ((values < lower) | (values > upper)).fillna(False)


def audit_outliers(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required column(s): {missing}")

    audited = df.copy()
    for col in TARGET_METRICS:
        z_scores, z_flags = compute_z_flags(audited[col], threshold=3.0)
        audited[f"{col}_z_score"] = z_scores.round(4)
        audited[f"{col}_z_flag"] = z_flags
        audited[f"{col}_iqr_flag"] = compute_iqr_flags(audited[col], multiplier=1.5)

    malic_num = pd.to_numeric(audited["malic_acid"], errors="coerce")
    proline_num = pd.to_numeric(audited["proline"], errors="coerce")

    audited["malic_acid_invalid"] = malic_num.isna() | (malic_num < 0)
    audited["proline_invalid"] = proline_num.isna() | (proline_num <= 0)
    audited["domain_invalid"] = audited["malic_acid_invalid"] | audited["proline_invalid"]

    flag_cols = ["malic_acid_z_flag", "malic_acid_iqr_flag", "proline_z_flag", "proline_iqr_flag", "domain_invalid"]
    audited["review_needed"] = audited[flag_cols].any(axis=1)

    retained = audited.loc[~audited["domain_invalid"]].copy()
    quarantined = audited.loc[audited["domain_invalid"]].copy()

    summary = pd.DataFrame({
        "metric": [
            "total_input_rows", "retained_rows", "quarantined_rows", "review_needed_rows",
            "malic_acid_z_flags", "malic_acid_iqr_flags", "proline_z_flags", "proline_iqr_flags",
        ],
        "count": [
            int(len(audited)), int(len(retained)), int(len(quarantined)), int(audited["review_needed"].sum()),
            int(audited["malic_acid_z_flag"].sum()), int(audited["malic_acid_iqr_flag"].sum()),
            int(audited["proline_z_flag"].sum()), int(audited["proline_iqr_flag"].sum()),
        ],
    })
    return retained, quarantined, summary


def run_audit(data_path: Path | str = DEFAULT_DATA, output_dir: Path | str = DEFAULT_OUTPUT):
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    retained, quarantined, summary = audit_outliers(load_and_validate(data_path))
    retained.to_csv(out_path / "retained_flagged.csv", index=False)
    quarantined.to_csv(out_path / "quarantined_records.csv", index=False)
    summary.to_csv(out_path / "quality_summary.csv", index=False)
    return retained, quarantined, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Wine Chemistry Outlier Audit")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    _, _, summary = run_audit(data_path=args.data, output_dir=args.output)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
