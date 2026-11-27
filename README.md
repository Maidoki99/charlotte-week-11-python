# Week 11 Class 4 — Feature Engineering from LAS and SCADA-Style CSV

## Data sources & limitations
- **Synthetic teaching datasets (`data/DATASET_CARD.md`):** Both `data/scada_synthetic.csv` (9 rows for `WELL-B`, depths `1000`–`1008 m`) and `data/well_b_synthetic.las` (9 depth-stepped log records for `WELL-B`) are synthetic instructional datasets and must not be represented as real field telemetry.
- **Assumed physical units:** `depth_m` (`DEPT.M` in meters `m`), `pressure_psi` (`psi`), `temperature_c` (`°C`), `flow_m3_h` (`m³/h`), `gamma_api` (`GR.API` in `API`), and `density_g_cm3` (`RHOB.G/C3` in `g/cm³`).
- **Alignment limitation:** Real LAS depth logs and SCADA time-series streams do not align automatically; the shared `(well_id, depth_m)` key is deliberately aligned for educational feature engineering.

## Running the project
```bash
python -m pip install -r requirements.txt
python feature_pipeline.py
python -m unittest discover -v
```

## Feature definitions

| Feature | Formula | Grouping / Order | Missing-value policy |
|---|---|---|---|
| `pressure_missing_flag` | `1` if `pressure_psi` is `NaN`, else `0` | Row-level after sorting by `well_id, depth_m` | Binary indicator preserving missing sensor readings |
| `flow_missing_flag` | `1` if `flow_m3_h` is `NaN`, else `0` | Row-level after sorting by `well_id, depth_m` | Binary indicator preserving missing sensor readings |
| `pressure_delta_psi` | `pressure_psi[t] - pressure_psi[t-1]` (`diff()`) | Grouped by `well_id`, ordered by `depth_m` | First row per well and rows adjacent to `NaN` remain `NaN` |
| `pressure_var_3` | Trailing 3-point population variance `rolling(3, min_periods=3).var(ddof=0)` | Grouped by `well_id`, ordered by `depth_m` (no future look-ahead) | Requires 3 non-null consecutive readings; incomplete windows remain `NaN` |
| `temperature_delta_c` | `temperature_c[t] - temperature_c[t-1]` (`diff()`) | Grouped by `well_id`, ordered by `depth_m` | First row per well remains `NaN` |
| `cumulative_sampled_flow` | `cumsum()` of `flow_m3_h` (`fillna(0.0)` inside sum) | Grouped by `well_id`, ordered by `depth_m` | Missing rate treated as `0.0` increment while `flow_missing_flag` tracks the gap. **Important:** running sum of sampled *rates* (`m³/h`), **not** physically integrated fluid volume (which requires multiplying each rate by time step `Δt`) |
| `gamma_density_ratio` | `gamma_api / density_g_cm3` | Row-level | Returns `NaN` if `density_g_cm3` is missing or zero |

## Validation evidence
- **Input & output counts (`outputs/quality_summary.csv`):** `scada_input_rows = 9` (`1` missing `pressure_psi` at `1005 m`, `1` missing `flow_m3_h` at `1007 m`), `las_input_rows = 9`, `merged_rows = 9` (`100%` `one_to_one` key match), `output_rows = 9`, `output_columns = 16`.
- **Automated tests (`python -m unittest discover -v`):** 4 unit tests verify input presence, duplicate-key failure, well-boundary isolation + trailing-window math, and end-to-end CSV outputs.

## AWS connection
In an Amazon S3 analytical data lake, **CSV** is convenient for small human-readable file drops, whereas **Parquet** is preferable for analytical workloads and AWS Glue/Athena queries because it stores columnar data with embedded schema types, compression, and column pruning—reducing S3 scan costs and preventing type-coercion bugs.

## GitHub PR
- **Branch:** `feature/week11-class04-feature-engineering`
- **Pull Request:** https://github.com
- **Summary:** Implemented LAS (`lasio`) and SCADA CSV ingestion, strict `one_to_one` join validation, order-aware per-well feature engineering, CSV quality reporting, and unit tests.
