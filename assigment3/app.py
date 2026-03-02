# app.py

from flask import Flask, request, jsonify
import joblib
from score import score

app = Flask(__name__)

# Load best model
model = joblib.load("best_model.pkl")


@app.route("/score", methods=["POST"])
def score_endpoint():
    data = request.get_json()

    if "text" not in data:
        return jsonify({"error": "Missing text field"}), 400

    text = data["text"]
    threshold = data.get("threshold", 0.5)

    prediction, propensity = score(text, model, threshold)

    return jsonify({
        "prediction": int(prediction),
        "propensity": propensity
    })


if __name__ == "__main__":
    app.run(port=5000)