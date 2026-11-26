"""Smoke checks: expected to fail until you implement starter tasks."""
import unittest
from pathlib import Path
from preprocess_iris import prepare_dataset


class StudentSmokeTest(unittest.TestCase):
    def test_usable_outputs(self):
        _tr, train, test, report = prepare_dataset(
            Path(__file__).with_name("iris_preprocessing_case_PUBLIC_DERIVED.csv")
        )
        self.assertEqual(len(train), 120)
        self.assertEqual(len(test), 30)
        self.assertEqual(list(train.columns), list(test.columns))
        self.assertNotIn("species", train.columns)
        self.assertNotIn("sample_id", train.columns)
        self.assertEqual(report["source_rows"], 150)
        self.assertEqual(report["training_rows"], 120)
        self.assertEqual(report["test_rows"], 30)
        self.assertEqual(report["transformed_features"], 9)
        self.assertEqual(report["missing_source_cells"], 11)


if __name__ == "__main__":
    unittest.main()
