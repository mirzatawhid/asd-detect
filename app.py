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
        if not data:
            return jsonify({"error": "Empty request received"}), 400

        features = []

        # Validate and process numeric features
        for feature in NUMERIC_FEATURES:
            if feature in data:
                try:
                    features.append(float(data[feature]))  # Convert to float
                except ValueError:
                    return jsonify({"error": f"Invalid numeric value for {feature}: {data[feature]}"}), 400
            else:
                return jsonify({"error": f"Missing numeric feature: {feature}"}), 400

        # Validate and encode categorical features
        for col in CATEGORICAL_FEATURES:
            if col in data:
                if data[col] in label_encoders[col].classes_:
                    features.append(label_encoders[col].transform([data[col]])[0])
                else:
                    return jsonify({"error": f"Invalid category for {col}: {data[col]}"}), 400
            else:
                return jsonify({"error": f"Missing categorical feature: {col}"}), 400

        # Convert to NumPy array and reshape
        X_input = np.array(features).reshape(1, -1)

        # Predict ASD risk
        prediction = model.predict(X_input)[0][0]

        # Format response
        return jsonify({
            "ASD_Risk_Score": round(float(prediction), 4),
            "ASD_Prediction": "High Risk" if prediction > 0.5 else "Low Risk"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000, debug=True)
