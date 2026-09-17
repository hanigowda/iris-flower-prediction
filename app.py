from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load the saved ML files
model = joblib.load("model.joblib")
scaler = joblib.load("scaler.joblib")
encoder = joblib.load("encoder.joblib")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    # Get input values from the HTML form
    sepal_length = float(request.form["sepal_length"])
    sepal_width = float(request.form["sepal_width"])
    petal_length = float(request.form["petal_length"])
    petal_width = float(request.form["petal_width"])

    # Create input array
    features = np.array([[
        sepal_length,
        sepal_width,
        petal_length,
        petal_width
    ]])

    # Normalize the input
    scaled_features = scaler.transform(features)

    # Predict using Logistic Regression
    prediction = model.predict(scaled_features)

    # Decode the prediction
    species = encoder.inverse_transform(prediction)[0]

    return render_template(
        "index.html",
        prediction=species
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)