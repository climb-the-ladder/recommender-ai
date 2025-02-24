from flask import Flask, request, jsonify
import joblib
import os
import pandas as pd

app = Flask(__name__)

script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, "../recommender-models/career_recommender.pkl")
scaler_path = os.path.join(script_dir, "../recommender-models/scaler.pkl")

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)  # Load the scaler for feature scaling

#expected feature names 
expected_features = [
    "GPA", "Extracurriculars", "InternshipExperience", "Projects",
    "Leadership_Positions", "Courses", "Research_Experience", "Coding_Skills",
    "Communication_Skills", "Problem_Solving_Skills", "Teamwork_Skills",
    "AnalyticalSkills", "Presentation_Skills", "Networking_Skills",
    "Certifications"
]

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json  # Get user input for the frontebd
    print("Received data:", data)  

    # Convert input to DataFrame
    features = pd.DataFrame([data])

    # Ensure all required features are present (fill missing ones with 0)
    for feature in expected_features:
        if feature not in features:
            features[feature] = 0

    features = features[expected_features]

    # Convert categorical values ("Yes"/"No") to numbers
    for column in ["Extracurriculars", "InternshipExperience", "Courses", "Certifications"]:
        features[column] = features[column].map({"Yes": 1, "No": 0}).fillna(0)

    #Apply scaling to match the training data
    features_scaled = scaler.transform(features)

    # Make a prediction
    probabilities = model.predict_proba(features_scaled)[0]
    career_classes = model.classes_

    # Sort predictions by probability (disabled currently)
    career_predictions = sorted(zip(career_classes, probabilities), key=lambda x: x[1], reverse=True)

    # Return the top career predictions
    top_careers = [{"career": career, "probability": round(prob * 100, 2)} for career, prob in career_predictions[:3]]

    print("Returning predictions:", top_careers)  # Debugging
    return jsonify({"predictions": top_careers})

if __name__ == "__main__":
    app.run(port=5001, debug=True)
