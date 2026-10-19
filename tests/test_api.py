import unittest

from src.api import create_app
from src.data import clean
from src.models import candidates
from src.train import split
from tests.helpers import synthetic_heart
from tests.test_model import SAMPLE


class TestApi(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        X_tr, _, y_tr, _ = split(clean(synthetic_heart(250)))
        model = candidates()["logistic_regression"][0].fit(X_tr, y_tr)
        cls.client = create_app(model=model, metadata={"model_name": "test"}).test_client()

    def test_health(self):
        self.assertEqual(self.client.get("/health").get_json(), {"status": "ok"})

    def test_predict_ok(self):
        r = self.client.post("/predict", json=SAMPLE)
        self.assertEqual(r.status_code, 200)
        body = r.get_json()
        self.assertEqual(set(body), {"prediction", "label", "confidence", "probability_heart_disease"})

    def test_predict_bad_input_is_400(self):
        self.assertEqual(self.client.post("/predict", json={"age": 5}).status_code, 400)
        self.assertEqual(self.client.post("/predict", data="nope").status_code, 400)

    def test_metrics_exposed_after_traffic(self):
        self.client.post("/predict", json=SAMPLE)
        text = self.client.get("/metrics").get_data(as_text=True)
        self.assertIn("api_requests_total", text)
        self.assertIn("predict_latency_seconds_count", text)


if __name__ == "__main__":
    unittest.main()
