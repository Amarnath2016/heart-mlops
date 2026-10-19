import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src import config, data
from tests.helpers import synthetic_heart


class TestData(unittest.TestCase):
    def test_load_raw_parses_question_marks_as_nan(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "raw.data"
            p.write_text("63,1,1,145,233,1,2,150,0,2.3,3,0,6,0\n"
                         "67,1,4,160,286,0,2,108,1,1.5,2,3,?,2\n")
            df = data.load_raw(p)
        self.assertEqual(df.shape, (2, 14))
        self.assertTrue(np.isnan(df.loc[1, "thal"]))

    def test_clean_binarises_target_and_drops_duplicates(self):
        df = synthetic_heart(50, with_missing=False)
        df["target"] = np.arange(len(df)) % 5      # 0..4 like the raw data
        df = pd.concat([df, df.iloc[[0]]], ignore_index=True)  # one duplicate row
        out = data.clean(df)
        self.assertEqual(len(out), 50)
        self.assertSetEqual(set(out["target"].unique()), {0, 1})

    def test_clean_keeps_missing_for_pipeline_imputation(self):
        out = data.clean(synthetic_heart(100))
        self.assertGreater(out[["ca", "thal"]].isna().sum().sum(), 0)

    def test_prepare_writes_csv(self):
        with tempfile.TemporaryDirectory() as d:
            raw, out = Path(d) / "r.data", Path(d) / "c.csv"
            synthetic_heart(30).to_csv(raw, header=False, index=False)
            df = data.prepare(raw, out)
            self.assertTrue(out.exists())
            self.assertEqual(list(df.columns), config.COLUMNS)


if __name__ == "__main__":
    unittest.main()
