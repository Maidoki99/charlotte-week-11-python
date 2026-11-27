# Week 11 Class 5 Capstone — Student Submission

## Environment and exact commands
- **Python:** Python 3.11 on Windows (Git Bash)
- **Install dependencies:**
  ```bash
  python -m pip install -r requirements.txt
  ```
- **Run full capstone pipeline (generates all 6 required output files):**
  ```bash
  python feature_pipeline.py
  ```
- **Run automated unit tests:**
  ```bash
  python -m unittest discover -v
  ```

## Data provenance
- **General measurements (`data/general_measurements_PUBLIC_DERIVED.csv`):** Derived from the first 180 rows and four geometric columns (`mean_radius`, `mean_texture`, `mean_perimeter`, `mean_area`) of the public scikit-learn / UCI Wisconsin Diagnostic Breast Cancer dataset (`data/DATASET_CARD.md`). Diagnosis labels were intentionally removed, and synthetic fields (`sample_id`, `lab_site`, `instrument`) plus controlled missing and non-positive values were added for educational data wrangling only. No clinical claims or predictive medical models are made.
- **Site metadata (`data/site_metadata_SYNTHETIC.json`):** Wholly synthetic lookup table mapping `NORTH -> Region_1`, `EAST -> Region_2`, and `WEST -> Region_3`.
- **Industrial files (`data/well_log_SYNTHETIC.las` and `data/scada_SYNTHETIC.csv`):** Wholly synthetic LAS 2.0 borehole log (`DEMO_WELL_11`, `1450–1457 m`) and 4-hour SCADA-style telemetry (`18` records) created with controlled gaps and join exceptions for training purposes only.

## Data-quality decisions
- **Invalid non-positive `mean_area`:** Sample `SMP-0008` has `mean_area = -15.0`. Because physical area cannot be negative or zero, this record is excluded from the train/test feature matrices (`179` valid rows remain) and recorded as `excluded_invalid_mean_area` in `outputs/general_quality_report.csv`.
- **Missing numeric values (Part A):** Eight valid rows contain missing measurements (`mean_radius`: 1, `mean_texture`: 4, `mean_area`: 3). Rather than dropping valid rows, missingness is flagged in `general_quality_report.csv` and imputed using training-set medians only.
- **Outliers (Part A):** Plausible extreme measurements are flagged using the `1.5 * IQR` rule (`iqr_outlier_candidate_flag`) in `general_quality_report.csv` and retained instead of blindly deleted.
- **Unmatched site keys (Part A):** Left-joining `lab_site` to `site_metadata_SYNTHETIC.json` maps unmatched keys to `region = "UNMAPPED_SITE"` with `unmatched_site_flag = 1` (0 unmatched rows in the provided dataset).
- **Unmatched depths & missing sensor readings (Part B):** Left-joining SCADA rows (`18` records) to LAS depths (`8` depths from `1450–1457 m`) identifies one unmatched SCADA depth (`1499 m` at `2026-09-03T04:00:00`, flagged with `las_depth_matched_flag = 0`) and one missing pressure reading (`1455 m` at `2026-09-01T20:00:00`, flagged with `pressure_missing_flag = 1`). Both are reported in `outputs/industrial_quality_report.csv`.

## General track
- **Engineered features:**
  - `area_per_radius_sq = mean_area / (mean_radius ** 2)`: dimensionless compactness ratio comparing measured area against squared radius, guarded against non-positive denominators.
  - `texture_perimeter_ratio = mean_texture / mean_perimeter`: relative surface variation normalized by boundary length, guarded against non-positive denominators.
  - These ratios capture shape and texture proportions useful for general tabular feature engineering without implying any clinical validity.
- **Leakage-free 80/20 preprocessing:** Valid records (`179` rows) are split first into `143` training rows (80%) and `36` test rows (20%) using `train_test_split(..., test_size=0.20, random_state=42)`. Numeric median imputation (`SimpleImputer(strategy="median")`), feature scaling (`StandardScaler`), and categorical one-hot encoding (`OneHotEncoder(handle_unknown="ignore")`) are fitted **strictly on the training split** and then applied to the test split.
- **Outputs:** `outputs/general_train_features.csv` (`143 x 15`), `outputs/general_test_features.csv` (`36 x 15`), `outputs/general_quality_report.csv` (`180` audit rows), and `outputs/general_feature_schema.json`.

## Industrial track
- **LAS & SCADA ingestion and join:** `well_log_SYNTHETIC.las` is parsed with `lasio.read` (verifying `DEPT.M`, `GR.API`, `RESD.OHMM`, and `TEMP.DEGC` curves) and joined to `scada_SYNTHETIC.csv` on `["well_id", "depth_m"]` with `validate="many_to_one"` after sorting SCADA readings chronologically by `["well_id", "timestamp"]`.
- **Unit conversions & engineered signals:**
  - `pressure_kpa = pressure_psi * 6.894757` (converts `psi` to kilopascals `kPa`).
  - `temperature_c = (temperature_f - 32) * 5 / 9` (converts `°F` to `°C`).
  - `temp_diff_from_las_c = temperature_c - las_temp_degc` (thermal difference in `°C` between flowing SCADA fluid temperature and static LAS formation temperature).
  - `pressure_kpa_var_3`: 3-record trailing population rolling variance (`rolling(window=3, min_periods=2).var(ddof=0)`) in `kPa²` grouped by `well_id` in timestamp order.
  - `cumulative_sampled_flow_bpd`: running sum (`cumsum()`) of instantaneous `flow_bpd` readings per well. **Important engineering note:** this is a cumulative sampled rate indicator, **not** physical barrels produced, because true produced volume requires integrating flow rate over elapsed time (`Δt`).
- **Outputs:** `outputs/industrial_features.csv` (`18 x 16`) and `outputs/industrial_quality_report.csv`.

## How to reproduce / test
1. Run `python feature_pipeline.py` from the repository root to regenerate all six files inside `outputs/`.
2. Run `python -m unittest discover -v` to execute all 4 automated checks verifying file generation, schema alignment, zero missing processed values, exclusion of `SMP-0008`, absence of test-to-train scaler leakage, unmatched depth reporting, unit conversions, and rolling variance calculations.

## Limitations and next improvements
- **Depth alignment in real wells:** Real LAS files are indexed by measured depth while SCADA streams are indexed by time; exact `(well_id, depth_m)` joins only work in simplified examples and would require directional survey / well-completion depth mapping or nearest-depth interpolation (`merge_asof`) in production.
- **Volume integration:** Replacing `cumulative_sampled_flow_bpd` with trapezoidal or rectangular time integration across actual timestamp intervals (`Δt`) would yield true physical fluid volume.
- **AI assistance disclosure:** Used an AI coding assistant to structure boilerplate functions, unit tests, and documentation; all data-quality rules, schemas, and pipeline outputs were verified locally against the capstone brief and dataset card.

## GitHub PR link
- **Branch:** `feature/week11-class05-capstone`
- **Pull Request:** https://github.com
