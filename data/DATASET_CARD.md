# Dataset card and provenance

## General measurements (public-derived; instructional modifications)
Source: scikit-learn `load_breast_cancer` dataset, originally Wisconsin Diagnostic Breast Cancer (UCI). This package uses the first 180 source rows and only four original numeric measurement columns (`mean radius`, `mean texture`, `mean perimeter`, `mean area`). Outcome/diagnosis labels are intentionally **excluded**; do not make clinical claims or predictive medical decisions. `sample_id`, `lab_site`, and `instrument` were created synthetically for this exercise. A small number of cells were intentionally set missing or to implausible values. Lab site metadata in JSON is wholly synthetic.
Reference: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html

## Industrial files (fully synthetic)
`well_log_SYNTHETIC.las` is a constructed LAS 2.0-style borehole log, not field data. `scada_SYNTHETIC.csv` is constructed SCADA-like readings; SCADA itself is not a file format. Both contain controlled gaps and join exceptions.

## Licence & safe use
Public dataset provenance is attributed to scikit-learn/UCI; original dataset licence/terms should be checked at the source before redistribution outside training. All added columns, metadata, missingness and mistakes are instructional. Not for clinical decisions, reservoir decisions, or operational control.
