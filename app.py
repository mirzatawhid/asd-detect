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
    data = request.json
    features = []

    # Encode categorical values
    for col in label_encoders.keys():
        if col in data:
            features.append(label_encoders[col].transform([data[col]])[0])
        else:
            return jsonify({"error": f"Missing value: {col}"}), 400

    features = np.array(features).reshape(1, -1)
    prediction = model.predict(features)

    return jsonify({"ASD_Risk": float(prediction[0][0])})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
