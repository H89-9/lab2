
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
from pandas_tasks import analyze_titanic
from grader_contracts.pandas_tasks import TitanicInput


class TestPandasTasks(unittest.TestCase):

    def check_rows(self, rows):
        with TemporaryDirectory() as folder:
            path = Path(folder) / "sample.csv"
            pd.DataFrame(rows).to_csv(path, index=False)
            return analyze_titanic(TitanicInput(str(path)))

    def test_original_titanic(self):
        path = Path(__file__).with_name("titanic.csv")
        result = analyze_titanic(TitanicInput(str(path)))

        self.assertEqual(result.row_count, 891)
        self.assertEqual(result.adults_over_30_count, 305)
        self.assertEqual(result.missing_by_column["Age"], 177)
        self.assertEqual(result.missing_by_column["Cabin"], 687)
        self.assertEqual(result.missing_by_column["Embarked"], 2)
        self.assertEqual(set(result.mean_age_by_pclass), {1, 2, 3})
        self.assertAlmostEqual(
            result.survival_rate_by_pclass[1], 0.6296296296
        )
        self.assertEqual(
            result.highest_fares,
            [512.3292, 512.3292, 512.3292, 263.0, 263.0]
        )

    def test_age_threshold(self):
        result = self.check_rows({
            "Pclass": [1, 1, 2, 2],
            "Age": [30, 31, None, 40],
            "Survived": [1, 0, 1, 0],
            "Fare": [100, 80, 20, 5]
        })

        self.assertEqual(result.row_count, 4)
        self.assertEqual(result.adults_over_30_count, 2)
        self.assertEqual(result.missing_by_column["Age"], 1)
        self.assertAlmostEqual(result.mean_age_by_pclass[1], 30.5)
        self.assertAlmostEqual(result.mean_age_by_pclass[2], 40)
        self.assertAlmostEqual(result.survival_rate_by_pclass[1], 0.5)
        self.assertEqual(result.highest_fares, [100, 80, 20, 5])

    def test_repeated_fares(self):
        result = self.check_rows({
            "Pclass": [3, 3, 3],
            "Age": [20, 25, 29],
            "Survived": [0, 1, 0],
            "Fare": [10, 5, 10]
        })

        self.assertEqual(result.adults_over_30_count, 0)
        self.assertEqual(result.highest_fares, [10, 10, 5])
        self.assertAlmostEqual(
            result.survival_rate_by_pclass[3], 1 / 3
        )

    def test_missing_age(self):
        result = self.check_rows({
            "Pclass": [1, 1],
            "Age": [None, 40],
            "Survived": [0, 1],
            "Fare": [10, 20]
        })

        self.assertEqual(result.missing_by_column["Age"], 1)
        self.assertEqual(result.mean_age_by_pclass[1], 40)
        self.assertEqual(result.adults_over_30_count, 1)

if __name__ == "__main__":
    unittest.main()