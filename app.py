from flask import Flask, request, jsonify
import joblib
import os
import pandas as pd

app = Flask(__name__)

script_dir = os.path.dirname(os.path.abspath(__file__))
model = joblib.load(os.path.join(script_dir, "../recommender-models/career_xgb.pkl"))
scaler = joblib.load(os.path.join(script_dir, "../recommender-models/scaler.pkl"))
label_encoder = joblib.load(os.path.join(script_dir, "../recommender-models/label_encoder.pkl"))

# Expected input fields (only subject scores now)
expected_features = [
    "math_score", "history_score", "physics_score",
    "chemistry_score", "biology_score", "english_score", "geography_score"
]

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        features = pd.DataFrame([data])

        # Ensure all expected fields are present
        for feature in expected_features:
            if feature not in features:
                features[feature] = 0

        features = features[expected_features]
        features_scaled = scaler.transform(features)

        predicted_label = model.predict(features_scaled)[0]
        predicted_career = label_encoder.inverse_transform([predicted_label])[0]

        return jsonify({"career": predicted_career})

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
