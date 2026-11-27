"""Automated tests for Week 11 Class 4 LAS + SCADA feature engineering pipeline."""
from pathlib import Path
import tempfile
import unittest
import numpy as np
import pandas as pd
from feature_pipeline import DATA_DIR, engineer_features, load_las, load_scada, merge_inputs, run_pipeline


class TestFeaturePipeline(unittest.TestCase):
    def test_inputs_exist(self):
        self.assertTrue((DATA_DIR / "scada_synthetic.csv").exists())
        self.assertTrue((DATA_DIR / "well_b_synthetic.las").exists())

    def test_duplicate_keys_fail_validation(self):
        scada = load_scada(DATA_DIR / "scada_synthetic.csv")
        las = load_las(DATA_DIR / "well_b_synthetic.las")
        dup_scada = pd.concat([scada, scada.iloc[[0]]], ignore_index=True)
        with self.assertRaises(ValueError):
            merge_inputs(las, dup_scada)

    def test_well_boundary_isolation_and_trailing_window(self):
        sample = pd.DataFrame({
            "well_id": ["W1", "W1", "W1", "W2", "W2"],
            "depth_m": [100.0, 101.0, 102.0, 100.0, 101.0],
            "pressure_psi": [100.0, 110.0, 120.0, 500.0, 505.0],
            "temperature_c": [50.0, 51.0, 52.0, 80.0, 82.0],
            "flow_m3_h": [10.0, np.nan, 12.0, 20.0, 25.0],
            "gamma_api": [40.0, 42.0, 44.0, 60.0, 62.0],
            "density_g_cm3": [2.0, 2.0, 2.0, 2.5, 2.5],
        })
        out = engineer_features(sample)
        w2_first = out.loc[(out["well_id"] == "W2") & (out["depth_m"] == 100.0)].iloc[0]
        self.assertTrue(pd.isna(w2_first["pressure_delta_psi"]))
        self.assertTrue(pd.isna(w2_first["pressure_var_3"]))
        w1_third = out.loc[(out["well_id"] == "W1") & (out["depth_m"] == 102.0)].iloc[0]
        self.assertAlmostEqual(float(w1_third["pressure_var_3"]), 200.0 / 3.0, places=5)
        self.assertEqual(int(out.loc[out["flow_m3_h"].isna(), "flow_missing_flag"].iloc[0]), 1)

    def test_end_to_end_pipeline_outputs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            featured, summary = run_pipeline(output_dir=tmpdir)
            self.assertEqual(len(featured), 9)
            self.assertEqual(int(featured["pressure_missing_flag"].sum()), 1)
            self.assertEqual(int(featured["flow_missing_flag"].sum()), 1)
            self.assertTrue((Path(tmpdir) / "engineered_features.csv").exists())
            self.assertTrue((Path(tmpdir) / "quality_summary.csv").exists())


if __name__ == "__main__":
    unittest.main()
