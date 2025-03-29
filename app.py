from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
import joblib

app = Flask(__name__)

# Load the trained model and encoders
model = tf.keras.models.load_model("asd_model.h5")
label_encoders = joblib.load("label_encoders.pkl")

@app.route("/")
def home():
    return "ASD Detection API is Running!"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        features = []

        # Numeric inputs (ASD scores, age, result)
        numeric_features = [
            "A1_Score", "A2_Score", "A3_Score", "A4_Score", "A5_Score",
            "A6_Score", "A7_Score", "A8_Score", "A9_Score", "A10_Score", "age", "result"
        ]
        for feature in numeric_features:
            if feature in data:
                features.append(float(data[feature]))  # Ensure float values
            else:
                return jsonify({"error": f"Missing numeric value: {feature}"}), 400

        # Encode categorical values
        categorical_features = ["gender", "ethnicity", "jundice", "austim",
                                "contry_of_res", "used_app_before", "relation"]
        for col in categorical_features:
            if col in data:
                if data[col] in label_encoders[col].classes_:
                    features.append(label_encoders[col].transform([data[col]])[0])
                else:
                    return jsonify({"error": f"Invalid value for {col}: {data[col]}"}), 400
            else:
                return jsonify({"error": f"Missing categorical value: {col}"}), 400

        # Convert to NumPy array and reshape for model input
        features = np.array(features).reshape(1, -1)
        prediction = model.predict(features)[0][0]

        return jsonify({"ASD_Risk": float(prediction)})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
