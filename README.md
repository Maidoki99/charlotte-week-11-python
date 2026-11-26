# Week 11 Class 2 — Wine Chemistry Outlier Audit

**Source:** Public UCI Wine dataset distributed with `scikit-learn`, modified for educational instruction with three injected anomalies (`WINE-013`, `WINE-062`, `WINE-131`).

## Run & Test Commands
```bash
python -m pip install -r requirements.txt
python src/wine_outlier_audit.py
python -m unittest discover -s tests -v
```

## Validation and Decision Policy
- **Why statistical outliers are flagged rather than automatically dropped:** A statistical flag (`|Z| > 3` or `1.5×IQR`) only indicates numerical distance from the bulk of the sample—not that an observation is erroneous. In wine chemistry, rare high readings can reflect genuine cultivar variation, late-harvest grape concentration, or distinct fermentation conditions. Automatically deleting statistical outliers removes informative extreme cases and biases downstream models.
- **Quarantine vs. Retention:** Only records violating explicit domain sanity checks (`proline <= 0` or missing; `malic_acid < 0` or missing) are routed to `outputs/quarantined_records.csv` (`WINE-013` with `proline = -45.0`). Statistically extreme but physically plausible observations (`WINE-062` with `malic_acid = 99.0` and `WINE-131` with `proline = 9999.0`) stay in `outputs/retained_flagged.csv` with `review_needed = True` and their original raw values preserved.

## Results and Limitations
- `total_input_rows`: **178** | `retained_rows`: **177** | `quarantined_rows`: **1** (`WINE-013`) | `review_needed_rows`: **6**
- `malic_acid_z_flags`: **1** | `malic_acid_iqr_flags`: **4** | `proline_z_flags`: **1** | `proline_iqr_flags`: **1**
- **Trade-offs & Limitations:** Population Z-score (`ddof=0`) is sensitive to extreme values inflating the mean and standard deviation, whereas `1.5×IQR` resists extreme tails but flags more points in skewed distributions across cultivars. Domain rules (`proline <= 0`, `malic_acid < 0`) are instructional sanity checks rather than universal chemical bounds.

## GitHub PR
- **Branch:** `feature/week11-class02-outliers`
- **Pull Request:** https://github.com
