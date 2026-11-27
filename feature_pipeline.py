"""Week 11 Class 4 — Industrial formats and feature engineering pipeline."""
from __future__ import annotations

from pathlib import Path
import lasio
import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent / "data"
OUTPUT_DIR = Path(__file__).parent / "outputs"

SCADA_REQUIRED_COLS = {"well_id", "depth_m", "pressure_psi", "temperature_c", "flow_m3_h"}
SCADA_NUMERIC_COLS = ["depth_m", "pressure_psi", "temperature_c", "flow_m3_h"]


def load_scada(path: Path | str) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing_cols = SCADA_REQUIRED_COLS - set(df.columns)
    if missing_cols:
        raise ValueError(f"SCADA input missing required column(s): {sorted(missing_cols)}")
    df = df.copy()
    df["well_id"] = df["well_id"].astype(str).str.strip()
    for col in SCADA_NUMERIC_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if df[["well_id", "depth_m"]].isna().any().any():
        raise ValueError("SCADA input contains null well_id or depth_m keys.")
    if int(df.duplicated(["well_id", "depth_m"]).sum()) > 0:
        raise ValueError("SCADA input contains duplicate (well_id, depth_m) keys.")
    return df.sort_values(["well_id", "depth_m"]).reset_index(drop=True)


def load_las(path: Path | str) -> pd.DataFrame:
    las = lasio.read(str(path))
    curve_units = {c.mnemonic.strip().upper(): (c.unit or "").strip().upper() for c in las.curves}
    missing_curves = {"DEPT", "GR", "RHOB"} - set(curve_units.keys())
    if missing_curves:
        raise ValueError(f"LAS input missing required curve(s): {sorted(missing_curves)}")
    if curve_units["DEPT"] and curve_units["DEPT"] not in {"M", "METERS", "METRES"}:
        raise ValueError(f"Unexpected depth unit in LAS file: {curve_units['DEPT']}")
    well_item = las.well.get("WELL") if hasattr(las, "well") else None
    well_id = str(well_item.value).strip() if well_item is not None and str(well_item.value).strip() else "UNKNOWN-WELL"
    raw_df = las.df().reset_index()
    raw_df.columns = [str(col).strip().upper() for col in raw_df.columns]
    out = pd.DataFrame({
        "well_id": well_id,
        "depth_m": pd.to_numeric(raw_df["DEPT"], errors="coerce"),
        "gamma_api": pd.to_numeric(raw_df["GR"], errors="coerce"),
        "density_g_cm3": pd.to_numeric(raw_df["RHOB"], errors="coerce"),
    })
    if out[["well_id", "depth_m"]].isna().any().any():
        raise ValueError("LAS input contains null well_id or depth_m keys.")
    if int(out.duplicated(["well_id", "depth_m"]).sum()) > 0:
        raise ValueError("LAS input contains duplicate (well_id, depth_m) keys.")
    return out.sort_values(["well_id", "depth_m"]).reset_index(drop=True)


def merge_inputs(las_df: pd.DataFrame, scada_df: pd.DataFrame) -> pd.DataFrame:
    if las_df.duplicated(["well_id", "depth_m"]).any():
        raise ValueError("Duplicate (well_id, depth_m) keys in LAS DataFrame.")
    if scada_df.duplicated(["well_id", "depth_m"]).any():
        raise ValueError("Duplicate (well_id, depth_m) keys in SCADA DataFrame.")
    merged = scada_df.merge(
        las_df, on=["well_id", "depth_m"], how="inner", validate="one_to_one", indicator=True
    )
    if len(merged) == 0 or len(merged) != len(scada_df) or len(merged) != len(las_df):
        raise ValueError(f"Incomplete merge coverage: scada={len(scada_df)}, las={len(las_df)}, matched={len(merged)}.")
    return merged.drop(columns=["_merge"]).sort_values(["well_id", "depth_m"]).reset_index(drop=True)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.sort_values(["well_id", "depth_m"]).reset_index(drop=True).copy()
    out["pressure_missing_flag"] = out["pressure_psi"].isna().astype(int)
    out["flow_missing_flag"] = out["flow_m3_h"].isna().astype(int)
    grouped = out.groupby("well_id", sort=False)
    out["pressure_delta_psi"] = grouped["pressure_psi"].diff()
    out["pressure_var_3"] = grouped["pressure_psi"].transform(
        lambda s: s.rolling(window=3, min_periods=3).var(ddof=0)
    )
    out["temperature_delta_c"] = grouped["temperature_c"].diff()
    out["cumulative_sampled_flow"] = grouped["flow_m3_h"].transform(lambda s: s.fillna(0.0).cumsum())
    out["gamma_density_ratio"] = np.where(
        out["density_g_cm3"].notna() & (out["density_g_cm3"] != 0),
        out["gamma_api"] / out["density_g_cm3"],
        np.nan,
    )
    return out


def build_quality_summary(scada_df, las_df, merged_df, featured_df) -> pd.DataFrame:
    return pd.DataFrame([
        {"metric": "scada_input_rows", "value": int(len(scada_df))},
        {"metric": "las_input_rows", "value": int(len(las_df))},
        {"metric": "merged_rows", "value": int(len(merged_df))},
        {"metric": "output_rows", "value": int(len(featured_df))},
        {"metric": "output_columns", "value": len(featured_df.columns)},
        {"metric": "scada_missing_pressure_psi", "value": int(scada_df["pressure_psi"].isna().sum())},
        {"metric": "scada_missing_temperature_c", "value": int(scada_df["temperature_c"].isna().sum())},
        {"metric": "scada_missing_flow_m3_h", "value": int(scada_df["flow_m3_h"].isna().sum())},
        {"metric": "las_missing_gamma_api", "value": int(las_df["gamma_api"].isna().sum())},
        {"metric": "las_missing_density_g_cm3", "value": int(las_df["density_g_cm3"].isna().sum())},
        {"metric": "output_pressure_missing_flags", "value": int(featured_df["pressure_missing_flag"].sum())},
        {"metric": "output_flow_missing_flags", "value": int(featured_df["flow_missing_flag"].sum())},
    ])


def run_pipeline(
    scada_path: Path | str = DATA_DIR / "scada_synthetic.csv",
    las_path: Path | str = DATA_DIR / "well_b_synthetic.las",
    output_dir: Path | str = OUTPUT_DIR,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    scada_df = load_scada(scada_path)
    las_df = load_las(las_path)
    merged_df = merge_inputs(las_df, scada_df)
    featured_df = engineer_features(merged_df)
    summary_df = build_quality_summary(scada_df, las_df, merged_df, featured_df)
    featured_df.to_csv(out_dir / "engineered_features.csv", index=False)
    summary_df.to_csv(out_dir / "quality_summary.csv", index=False)
    return featured_df, summary_df


def main() -> None:
    featured_df, summary_df = run_pipeline()
    print(f"Engineered features written: {len(featured_df)} rows x {featured_df.shape[1]} columns")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
