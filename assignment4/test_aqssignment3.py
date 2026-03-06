import pytest
import joblib
import subprocess
import time
import requests
from score import score


# ---------- UNIT TEST ----------
def test_score():

    # Load model inside function (important!)
    model = joblib.load("best_model.pkl")

    # Smoke test
    pred, prop = score("Hello world", model, 0.5)

    # Format checks
    assert isinstance(pred, bool)
    assert isinstance(prop, float)

    # Sanity checks
    assert 0 <= prop <= 1
    assert pred in [True, False]

    # Edge case threshold = 0 → always 1
    pred_zero, _ = score("Any random text", model, 0)
    assert pred_zero is True

    # Edge case threshold = 1 → always 0
    pred_one, _ = score("Any random text", model, 1)
    assert pred_one is False

    spam_text = "Congratulations! You won free money!!!"
    pred_spam, prop_spam = score(spam_text, model, 0.5)

    assert isinstance(pred_spam, bool)
    assert 0 <= prop_spam <= 1
   


# ---------- INTEGRATION TEST ----------
def test_flask():

    # Start Flask app
    process = subprocess.Popen(["python", "app.py"])

    time.sleep(3)  # give server time to start

    try:
        response = requests.post(
            "http://127.0.0.1:5000/score",
            json={"text": "Free lottery prize!!!", "threshold": 0.5}
        )

        assert response.status_code == 200

        data = response.json()

        assert "prediction" in data
        assert "propensity" in data
        assert data["prediction"] in [0, 1]
        assert 0 <= data["propensity"] <= 1

    finally:
        process.terminate()
        process.wait()


def test_docker():

    # Build Docker image
    subprocess.run(["docker", "build", "-t", "spam-flask", "."], check=True)

    # Run Docker container
    container = subprocess.Popen(
        ["docker", "run", "-p", "5001:5000", "--name", "spam-test-container", "spam-flask"]
    )

    # Wait until container is ready
    time.sleep(8)

    try:
        # Retry request if server not ready yet
        for _ in range(10):
            try:
                response = requests.post(
                    "http://127.0.0.1:5001/score",
                    json={"text": "Free money!!!", "threshold": 0.5}
                )
                break
            except requests.exceptions.ConnectionError:
                time.sleep(1)

        assert response.status_code == 200

        data = response.json()

        assert "prediction" in data
        assert "propensity" in data
        assert data["prediction"] in [0, 1]
        assert 0 <= data["propensity"] <= 1

    finally:
        # Stop container
        subprocess.run(["docker", "stop", "spam-test-container"])
        subprocess.run(["docker", "rm", "spam-test-container"])
        container.terminate()

import pytest
import joblib

def test_invalid_inputs():
    model = joblib.load("best_model.pkl")

    with pytest.raises(ValueError):
        score(123, model, 0.5)

    with pytest.raises(ValueError):
        score("hello", model, 2)