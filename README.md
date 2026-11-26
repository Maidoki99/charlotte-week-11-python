# Week 11 Class 3 | Feature preprocessing

## Dataset source and changes
- **Public measured portion:** Classic UCI Iris flower dataset (`load_iris` from `scikit-learn`: https://uci.edu and https://scikit-learn.org), containing 150 observations across 3 species (`setosa`, `versicolor`, `virginica`) and 4 botanical measurements (`sepal_length_cm`, `sepal_width_cm`, `petal_length_cm`, `petal_width_cm`).
- **Teaching modifications (`DATASET_CARD.md`):** Added synthetic identifier `sample_id` (`S-001`–`S-150`) and two synthetic categorical context fields (`collection_site` and `instrument_type`), plus 11 deliberately blanked predictor cells (7 numeric and 4 `collection_site` values) to practice missing-data imputation. Synthetic fields are for preprocessing instruction only and do not represent real botanical metadata.

## Run instructions
- **Python & dependencies:** Python 3.11+ with `numpy>=1.24`, `pandas>=2.0`, and `scikit-learn>=1.3`.
```bash
python -m pip install -r requirements.txt
python preprocess_iris.py --input iris_preprocessing_case_PUBLIC_DERIVED.csv --output-dir outputs
python -m unittest test_student_smoke.py
```

## Feature choices
- **Numeric predictors (`sepal_length_cm`, `sepal_width_cm`, `petal_length_cm`, `petal_width_cm`):** Imputed with the training median (`SimpleImputer(strategy="median")`) to resist extreme values, then standardized (`StandardScaler()`) so features measured on different scales contribute comparably to distance-based or regularized downstream models.
- **Categorical predictors (`collection_site`, `instrument_type`):** Nominal labels with no inherent numeric rank. Imputed with the most frequent training category (`SimpleImputer(strategy="most_frequent")`) and encoded with `OneHotEncoder(handle_unknown="ignore", sparse_output=False)` so arbitrary integer ordering is avoided.
- **Excluded columns:** `sample_id` is a row identifier with no predictive generalization value, and `species` is the target label (`y`), which must never enter the predictor matrix (`X`).

## Why no leakage?
- **Split before fit:** `train_test_split(..., test_size=0.2, random_state=42, stratify=y)` separates the 120 training rows from the 30 held-out test rows **before** any imputation, scaling, or category vocabulary fitting occurs.
- **Training-only `fit`, test-only `transform`:** `ColumnTransformer.fit_transform(X_train)` computes medians, means, standard deviations, modes, and category vocabularies strictly from the 120 training records. `ColumnTransformer.transform(X_test)` reuses those exact training parameters on the 30 held-out records so test-set distributions never influence preprocessing statistics.

## Results and known limitations
- **Output counts (`outputs/data_quality_summary.csv`):** `source_rows = 150`, `training_rows = 120`, `test_rows = 30`, `transformed_features = 9` (4 standardized numeric columns + 3 `collection_site` indicators + 2 `instrument_type` indicators), and `missing_source_cells = 11`.
- **Known limitations:**
  - `handle_unknown="ignore"` prevents runtime crashes on unseen categories at test/inference time by emitting all-zero indicators for that feature, which silently maps unknown categories to the baseline state unless monitored for category drift.
  - Global median and most-frequent imputation ignore potential differences across species or collection sites, and `StandardScaler` does not bound extreme test values to `[0, 1]`.

## Pull request link
- **Branch:** `feature/week11-class03-preprocessing`
- **Pull Request:** https://github.com
