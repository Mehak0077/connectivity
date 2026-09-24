from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model
model = joblib.load("model.pkl")

# Iris class names
classes = ["setosa", "versicolor", "virginica"]


@app.route("/")
def home():
    return jsonify({
        "message": "ML Model REST API is running"
    })


@app.route("/predict", methods=["POST"])
def predict():

    data = request.get_json()

    # Get features from JSON request
    features = np.array(data["features"]).reshape(1, -1)

    # Make prediction
    prediction = model.predict(features)[0]

    # Return prediction
    return jsonify({
        "prediction": classes[prediction]
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)