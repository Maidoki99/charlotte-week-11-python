"""Automated tests for Week 11 Class 5 Capstone (General + Industrial pipelines)."""
from pathlib import Path
import tempfile
import unittest
import numpy as np
import pandas as pd
from feature_pipeline import DATA_DIR, run_general, run_industrial


class TestWeek11Capstone(unittest.TestCase):
    def test_general_pipeline_outputs_and_exclusions(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            train_df, test_df, qa_df, schema = run_general(
                DATA_DIR / "general_measurements_PUBLIC_DERIVED.csv",
                DATA_DIR / "site_metadata_SYNTHETIC.json",
                tmp_path,
            )
            for fname in [
                "general_train_features.csv",
                "general_test_features.csv",
                "general_quality_report.csv",
                "general_feature_schema.json",
            ]:
                self.assertTrue((tmp_path / fname).exists(), f"Missing {fname}")

            self.assertEqual(list(train_df.columns), list(test_df.columns))
            self.assertFalse(train_df.isna().any().any())
            self.assertFalse(test_df.isna().any().any())

            # SMP-0008 has mean_area = -15.0 and must be excluded from train/test
            self.assertNotIn("SMP-0008", set(train_df["sample_id"]))
            self.assertNotIn("SMP-0008", set(test_df["sample_id"]))
            excluded = qa_df.loc[qa_df["sample_id"] == "SMP-0008", "status"].iloc[0]
            self.assertEqual(excluded, "excluded_invalid_mean_area")
            self.assertEqual(len(train_df) + len(test_df), 179)

    def test_general_no_test_leakage_into_scaler(self):
        with tempfile.TemporaryDirectory() as tmpdir1, tempfile.TemporaryDirectory() as tmpdir2:
            csv_path = DATA_DIR / "general_measurements_PUBLIC_DERIVED.csv"
            json_path = DATA_DIR / "site_metadata_SYNTHETIC.json"
            train1, test1, _, _ = run_general(csv_path, json_path, tmpdir1)

            # Inflate a test-set sample's mean_area wildly; training outputs must stay identical
            tampered_df = pd.read_csv(csv_path)
            test_sample_id = test1["sample_id"].iloc[0]
            tampered_df.loc[tampered_df["sample_id"] == test_sample_id, "mean_area"] = 999999.0
            tampered_csv = Path(tmpdir2) / "tampered.csv"
            tampered_df.to_csv(tampered_csv, index=False)

            train2, _, _, _ = run_general(tampered_csv, json_path, tmpdir2)
            pd.testing.assert_frame_equal(train1, train2)

    def test_industrial_pipeline_outputs_and_unmatched_depth(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            ind_df, qa_df = run_industrial(
                DATA_DIR / "well_log_SYNTHETIC.las",
                DATA_DIR / "scada_SYNTHETIC.csv",
                tmp_path,
            )
            self.assertTrue((tmp_path / "industrial_features.csv").exists())
            self.assertTrue((tmp_path / "industrial_quality_report.csv").exists())
            self.assertEqual(len(ind_df), 18)

            qa_map = dict(zip(qa_df["metric"], qa_df["value"]))
            self.assertEqual(qa_map["unmatched_las_rows"], "1")
            self.assertEqual(qa_map["unmatched_depths_m"], "1499")
            self.assertEqual(qa_map["missing_pressure_rows"], "1")

    def test_industrial_conversions_and_rolling_math(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ind_df, _ = run_industrial(
                DATA_DIR / "well_log_SYNTHETIC.las",
                DATA_DIR / "scada_SYNTHETIC.csv",
                tmpdir,
            )
            first = ind_df.iloc[0]
            # 3032 psi * 6.894757 = 20904.903224 kPa
            self.assertAlmostEqual(float(first["pressure_kpa"]), 3032.0 * 6.894757, places=4)
            # (184.0 F - 32) * 5 / 9 = 84.444444 C; LAS temp at 1450m is 82.0 C -> diff = +2.444444 C
            self.assertAlmostEqual(float(first["temperature_c"]), (184.0 - 32.0) * 5.0 / 9.0, places=5)
            self.assertAlmostEqual(float(first["temp_diff_from_las_c"]), (184.0 - 32.0) * 5.0 / 9.0 - 82.0, places=5)
            # First row has only 1 sample -> rolling variance (min_periods=2) must be NaN
            self.assertTrue(pd.isna(first["pressure_kpa_var_3"]))
            # Second row has 2 samples -> rolling variance (min_periods=2) must be finite
            self.assertTrue(np.isfinite(float(ind_df.iloc[1]["pressure_kpa_var_3"])))
            # Cumulative sampled flow at row 2 = 400 + 403 + 406 = 1209
            self.assertAlmostEqual(float(ind_df.iloc[2]["cumulative_sampled_flow_bpd"]), 1209.0, places=5)


if __name__ == "__main__":
    unittest.main()
