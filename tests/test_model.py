import unittest

import numpy as np
from sklearn.model_selection import cross_val_score

from src import config
from src.data import clean
from src.features import build_preprocessor
from src.models import candidates
from src.predict import ValidationError, predict_one, validate
from src.train import split, evaluate_holdout
from tests.helpers import synthetic_heart

SAMPLE = {"age": 63, "sex": 1, "cp": 4, "trestbps": 145, "chol": 233, "fbs": 1,
          "restecg": 2, "thalach": 150, "exang": 0, "oldpeak": 2.3, "slope": 3,
          "ca": 0, "thal": 6}


class TestModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = clean(synthetic_heart(250))
        cls.X_tr, cls.X_te, cls.y_tr, cls.y_te = split(cls.df)
        pipe, _ = candidates()["logistic_regression"]
        cls.model = pipe.fit(cls.X_tr, cls.y_tr)

    def test_preprocessor_output_has_no_nan_and_is_scaled(self):
        prep = build_preprocessor()
        Z = prep.fit_transform(self.df[config.FEATURES])
        Z = Z.toarray() if hasattr(Z, "toarray") else Z
        self.assertFalse(np.isnan(Z).any())
        self.assertAlmostEqual(float(Z[:, 0].mean()), 0.0, places=6)  # age standardised

    def test_unseen_category_does_not_crash(self):
        row = self.X_te.iloc[[0]].copy()
        row["thal"] = 99.0
        self.assertEqual(self.model.predict_proba(row).shape, (1, 2))

    def test_split_is_stratified_and_disjoint(self):
        self.assertAlmostEqual(self.y_tr.mean(), self.y_te.mean(), delta=0.05)
        self.assertEqual(len(set(self.X_tr.index) & set(self.X_te.index)), 0)

    def test_model_beats_chance_in_cv(self):
        pipe, _ = candidates()["random_forest"]
        auc = cross_val_score(pipe, self.X_tr, self.y_tr, cv=3, scoring="roc_auc").mean()
        self.assertGreater(auc, 0.6)

    def evaluate_holdout_in_range(self):
        for k, v in evaluate_holdout(self.model, self.X_te, self.y_te).items():
            self.assertTrue(0.0 <= v <= 1.0, k)

    def test_predict_one_schema(self):
        out = predict_one(self.model, SAMPLE)
        self.assertIn(out["prediction"], (0, 1))
        self.assertTrue(0.5 <= out["confidence"] <= 1.0)

    def test_validation_rules(self):
        with self.assertRaises(ValidationError):
            validate({k: v for k, v in SAMPLE.items() if k != "age"})
        with self.assertRaises(ValidationError):
            validate({**SAMPLE, "age": 500})
        with self.assertRaises(ValidationError):
            validate({**SAMPLE, "cp": 9})
        with self.assertRaises(ValidationError):
            validate({**SAMPLE, "chol": "abc"})
        self.assertIsNone(validate({**SAMPLE, "ca": None})["ca"])


if __name__ == "__main__":
    unittest.main()
