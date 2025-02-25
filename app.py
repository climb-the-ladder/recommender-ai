from flask import Flask, request, jsonify
import joblib
import os
import pandas as pd

app = Flask(__name__)

# Load trained model, scaler, and label encoder
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, "../recommender-models/career_xgb.pkl")  # use career_xgb.pkl or career_rf.pkl
scaler_path = os.path.join(script_dir, "../recommender-models/scaler.pkl")
encoder_path = os.path.join(script_dir, "../recommender-models/label_encoder.pkl")

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)
label_encoder = joblib.load(encoder_path)

expected_features = [
    "GPA", "Extracurriculars", "InternshipExperience", "Projects",
    "Leadership_Positions", "Courses", "Research_Experience", "Coding_Skills",
    "Communication_Skills", "Problem_Solving_Skills", "Teamwork_Skills",
    "AnalyticalSkills", "Presentation_Skills", "Networking_Skills",
    "Certifications"
]

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        features = pd.DataFrame([data])

        # Convert categorical variables
        categorical_columns = ["Extracurriculars", "InternshipExperience", "Courses", "Certifications"]
        for col in categorical_columns:
            if col in features.columns:
                features[col] = features[col].map({"Yes": 1, "No": 0})

        for feature in expected_features:
            if feature not in features:
                features[feature] = 0

        features = features[expected_features]

        # to scale features
        features_scaled = scaler.transform(features)

        # predict numeric label
        predicted_label = model.predict(features_scaled)[0]

        # Convert numeric label to career name abck again
        predicted_career = label_encoder.inverse_transform([predicted_label])[0]

        return jsonify({"career": predicted_career})

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
