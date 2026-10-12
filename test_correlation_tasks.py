import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
import pandas as pd

from correlation_tasks import analyze_brain_correlations
from grader_contracts.correlation_tasks import BrainDataInput

class TestCorrelationTasks(unittest.TestCase):

    def get_result(self):
        path = Path(__file__).with_name("brainsize.txt")
        return analyze_brain_correlations(BrainDataInput(str(path)))

    def test_group_counts(self):
        result = self.get_result()
        self.assertEqual(result.men_count, 20)
        self.assertEqual(result.women_count, 20)

    def test_correlation_values(self):
        result = self.get_result()

        self.assertAlmostEqual(
            result.men_mri_correlation["PIQ"], 0.568237, places=5
        )
        self.assertAlmostEqual(
            result.women_mri_correlation["Weight"],
            0.446271, places=5
        )
        self.assertAlmostEqual(
            result.men_mri_correlation["Weight"],
            -0.0768746, places=5
        )

    def test_all_features(self):
        result = self.get_result()

        features = {"FSIQ", "VIQ", "PIQ", "Weight", "Height"}

        self.assertEqual(
            set(result.men_mri_correlation), features
        )
        self.assertEqual(
            set(result.women_mri_correlation), features
        )
        self.assertEqual(result.strongest_mri_feature, "PIQ")

    def test_missing_values_and_negative_correlation(self):
        table = pd.DataFrame({
            "Gender": ["Male"] * 4 + ["Female"] * 4,
            "MRI_Count": [100, 200, 300, 400, 150, 250, 350, 450],
            "FSIQ": [4, 3, 2, 1, 1, 3, 2, 4],
            "VIQ": [1, 3, 4, 2, 2, 3, 4, 1],
            "PIQ": [2, 1, 4, 3, 4, 2, 1, 3],
            "Weight": [1, None, 2, 4, 2, 4, None, 3],
            "Height": [3, 2, 4, 1, 3, 1, 4, 2]
        })

        with TemporaryDirectory() as folder:
            path = Path(folder) / "sample.txt"
            table.to_csv(path, sep="\t", index=False, na_rep="NA")
            result = analyze_brain_correlations(
                BrainDataInput(str(path))
            )

        self.assertEqual(result.men_count, 4)
        self.assertEqual(result.women_count, 4)
        self.assertAlmostEqual(
            result.men_mri_correlation["FSIQ"], -1.0
        )
        self.assertEqual(result.strongest_mri_feature, "FSIQ")
        self.assertAlmostEqual(
            result.men_mri_correlation["Weight"],
            0.9285714286
        )

if __name__ == "__main__":
    unittest.main()