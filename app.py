from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib

app = Flask(__name__)

# Load the trained model and encoders
model = tf.keras.models.load_model("asd_model.h5")
label_encoders = joblib.load("label_encoders.pkl")

# Define feature names
NUMERIC_FEATURES = [
    "A1_Score", "A2_Score", "A3_Score", "A4_Score", "A5_Score",
    "A6_Score", "A7_Score", "A8_Score", "A9_Score", "A10_Score",
    "age"
]
CATEGORICAL_FEATURES = [
    "gender", "ethnicity", "jundice", "austim",
    "contry_of_res", "used_app_before", "relation"
]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "ASD Detection API is Running!"})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        features = []

        NUMERIC_FEATURES = [
            "A1_Score", "A2_Score", "A3_Score", "A4_Score", "A5_Score",
            "A6_Score", "A7_Score", "A8_Score", "A9_Score", "A10_Score",
            "age", "result"
        ]
        CATEGORICAL_FEATURES = [
            "gender", "ethnicity", "jundice", "austim",
            "contry_of_res", "used_app_before", "relation"
        ]

        # 1️⃣ CHECK & APPEND NUMERIC FEATURES
        for feature in NUMERIC_FEATURES:
            if feature not in data:
                return jsonify({"error": f"Missing numeric value: {feature}"}), 400
            features.append(float(data[feature]))  # Ensure float type

        # 2️⃣ CHECK & APPEND CATEGORICAL FEATURES
        for col in CATEGORICAL_FEATURES:
            if col in data:
                if data[col] in label_encoders[col].classes_:
                    features.append(label_encoders[col].transform([data[col]])[0])
                else:
                    return jsonify({"error": f"Invalid value for {col}: {data[col]}"}), 400
            else:
                return jsonify({"error": f"Missing categorical value: {col}"}), 400

        # 3️⃣ CHECK FINAL FEATURE COUNT
        print(f"Expected features: 19, Received features: {len(features)}")  # Debugging

        # 4️⃣ CONVERT TO NUMPY ARRAY & CHECK SHAPE
        features = np.array(features).reshape(1, -1)
        print(f"Final input shape: {features.shape}")  # Expected (1, 19)

        # 5️⃣ RUN PREDICTION
        prediction = model.predict(features)[0][0]

        return jsonify({"ASD_Risk": float(prediction)})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

