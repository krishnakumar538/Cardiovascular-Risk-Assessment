"""
Flask API bridge for the Cardiovascular Risk Assessment frontend.

Expects a CSV named `cardio_train.csv` (the standard Kaggle Cardiovascular
Disease dataset, semicolon-separated) in the same folder as this file.
Trains a model once, saves it to disk, and reuses it on future runs.

Run with:
    pip install flask flask-cors scikit-learn pandas joblib
    python app.py
"""
import pandas as pd

df = pd.read_csv("cardio_train.csv")

# Remove rows where target value is missing
df = df.dropna(subset=["cardio"])

# Features and target
X = df.drop(columns=["cardio"])
y = df["cardio"]

from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy="median")
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Load dataset
df = pd.read_csv("cardio_train.csv")

# Remove rows with missing target
df = df.dropna(subset=["cardio"])

# Separate X and y
X = df.drop(columns=["cardio"])
y = df["cardio"]

# Fill missing feature values
imputer = SimpleImputer(strategy="median")
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)

print("Model trained successfully!")
from flask import Flask

app = Flask(__name__)

@app.route("/")
def home():
    return "Cardio Prediction API is running!"

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
df = df.dropna(subset=["cardio"])

from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

app = Flask(__name__)
CORS(app)  # allows the HTML page (opened via file:// or a different port) to call this API

MODEL_PATH = "cardio_model.pkl"
DATA_PATH = "cardio_train.csv"

FEATURE_ORDER = [
    "age", "gender", "height", "weight",
    "ap_hi", "ap_lo", "cholesterol", "gluc",
    "smoke", "alco", "active"
]


def train_model():
    """Trains a classifier from cardio_train.csv and saves it to disk."""
    df = pd.read_csv(DATA_PATH, sep=";")

    X = df[FEATURE_ORDER]
    y = df["cardio"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # <<< ================= INTEGRATE YOUR MODEL HERE ================= >>>
    # This is the line to change if you already trained your own model
    # (e.g. in a notebook) and just want to plug it in instead of
    # training a fresh RandomForest here. Replace the two lines below
    # with your own model object + its .fit(...) call.
    clf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
    clf.fit(X_train, y_train)
    # <<< ================================================================ >>>

    acc = accuracy_score(y_test, clf.predict(X_test))
    print(f"Model trained. Test accuracy: {acc:.4f}")

    joblib.dump(clf, MODEL_PATH)
    return clf


def load_or_train_model():
    if os.path.exists(MODEL_PATH):
        print("Loading existing model from", MODEL_PATH)
        return joblib.load(MODEL_PATH)
    print("No saved model found, training a new one...")
    return train_model()


model = load_or_train_model()


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        # Build the feature row in the exact order the model expects
        row = [[data[feat] for feat in FEATURE_ORDER]]
        X = pd.DataFrame(row, columns=FEATURE_ORDER)

        prediction = int(model.predict(X)[0])
        probability = float(model.predict_proba(X)[0][1])  # prob of class 1 (disease)

        return jsonify({
            "prediction": prediction,
            "probability": probability
        })

    except KeyError as e:
        return jsonify({"error": f"Missing field: {e}"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
