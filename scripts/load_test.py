"""Send N requests so the monitoring dashboard has data: python scripts/load_test.py [url] [n]"""
import json
import random
import sys
import urllib.request


def main():
    url = (sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000") + "/predict"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    for _ in range(n):
        body = {"age": random.randint(35, 75), "sex": random.randint(0, 1), "cp": random.randint(1, 4),
                "trestbps": random.randint(100, 180), "chol": random.randint(150, 350),
                "fbs": random.randint(0, 1), "restecg": random.randint(0, 2),
                "thalach": random.randint(90, 190), "exang": random.randint(0, 1),
                "oldpeak": round(random.uniform(0, 4), 1), "slope": random.randint(1, 3),
                "ca": random.randint(0, 3), "thal": random.choice([3, 6, 7])}
        req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
        urllib.request.urlopen(req).read()
    print("sent", n, "requests")


if __name__ == "__main__":
    main()
