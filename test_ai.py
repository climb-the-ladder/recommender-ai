import joblib
import os
import pandas as pd

script_dir = os.path.dirname(os.path.abspath(__file__))
model = joblib.load(os.path.join(script_dir, "../recommender-models/career_xgb.pkl"))
scaler = joblib.load(os.path.join(script_dir, "../recommender-models/scaler.pkl"))
label_encoder = joblib.load(os.path.join(script_dir, "../recommender-models/label_encoder.pkl"))

test_data = {
    "GPA": 2.5,
    "Extracurriculars": "No",
    "InternshipExperience": "Yes",
    "Projects": 5,
    "Leadership_Positions": 2,
    "Courses": "Yes",
    "Research_Experience": 4,
    "Coding_Skills": 5,
    "Communication_Skills": 1,
    "Problem_Solving_Skills": 5,
    "Teamwork_Skills": 3,
    "AnalyticalSkills": 5,
    "Presentation_Skills": 1,
    "Networking_Skills": 1,
    "Certifications": "Yes"
}

features = pd.DataFrame([test_data])

categorical_columns = ["Extracurriculars", "InternshipExperience", "Courses", "Certifications"]
for col in categorical_columns:
    features[col] = features[col].map({"Yes": 1, "No": 0})

features_scaled = scaler.transform(features)
predicted_label = model.predict(features_scaled)[0]

# to decode label to career name
predicted_career = label_encoder.inverse_transform([predicted_label])[0]

print(f"🎯 Recommended Career: {predicted_career}")
