import sys
import tempfile
import unittest
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.wine_outlier_audit import (
    DEFAULT_DATA, audit_outliers, compute_iqr_flags, compute_z_flags, load_and_validate, run_audit,
)


class TestOutlierAudit(unittest.TestCase):
    def test_missing_required_column_raises_value_error(self):
        bad_df = pd.DataFrame({
            "sample_id": ["W-1"],
            "alcohol": [13.5],
            "malic_acid": [1.8],
            "color_intensity": [5.0],
            "target": [0 for _ in range(1)],
        })
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "bad.csv"
            bad_df.to_csv(csv_path, index=False)
            with self.assertRaises(ValueError):
                load_and_validate(csv_path)

    def test_zero_variance_columns_handled_safely(self):
        constant_df = pd.DataFrame({
            "sample_id": ["W-1", "W-2", "W-3", "W-4"],
            "alcohol": [13.0 for _ in range(4)],
            "malic_acid": [2.0 for _ in range(4)],
            "color_intensity": [4.0 for _ in range(4)],
            "proline": [500.0 for _ in range(4)],
            "target": [0 for _ in range(4)],
        })
        z_scores, z_flags = compute_z_flags(constant_df["proline"])
        self.assertTrue((z_scores == 0.0).all())
        self.assertFalse(z_flags.any())
        self.assertFalse(compute_iqr_flags(constant_df["proline"]).any())
        retained, quarantined, _ = audit_outliers(constant_df)
        self.assertEqual(len(retained), 4)
        self.assertEqual(len(quarantined), 0)

    def test_domain_quarantine_vs_statistical_retention(self):
        df = pd.DataFrame({
            "sample_id": [f"N-{i}" for i in range(10)] + ["STAT-HIGH", "BAD-NEG"],
            "alcohol": [13.0 for _ in range(12)],
            "malic_acid": [1.8 for _ in range(10)] + [50.0, 1.8],
            "color_intensity": [4.5 for _ in range(12)],
            "proline": [800.0 for _ in range(11)] + [-10.0],
            "target": [0 for _ in range(12)],
        })
        retained, quarantined, _ = audit_outliers(df)
        self.assertEqual(quarantined["sample_id"].tolist(), ["BAD-NEG"])
        stat_row = retained.loc[retained["sample_id"] == "STAT-HIGH"].iloc[0]
        self.assertTrue(bool(stat_row["review_needed"]))
        self.assertEqual(float(stat_row["malic_acid"]), 50.0)

    def test_end_to_end_dataset_audit_and_outputs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            retained, quarantined, _ = run_audit(DEFAULT_DATA, tmpdir)
            self.assertEqual(len(retained) + len(quarantined), 178)
            self.assertEqual(quarantined["sample_id"].tolist(), ["WINE-013"])
            for fname in ["retained_flagged.csv", "quarantined_records.csv", "quality_summary.csv"]:
                self.assertTrue((Path(tmpdir) / fname).exists())


if __name__ == "__main__":
    unittest.main()
