import joblib
import os
import pandas as pd

# Load trained model and scaler
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, "../recommender-models/career_recommender.pkl")
scaler_path = os.path.join(script_dir, "../recommender-models/scaler.pkl")

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)  # Load scaler for feature scaling

# 🔥 Define expected feature names (must match training dataset)
expected_features = [
    "GPA", "Extracurriculars", "InternshipExperience", "Projects",
    "Leadership_Positions", "Courses", "Research_Experience", "Coding_Skills",
    "Communication_Skills", "Problem_Solving_Skills", "Teamwork_Skills",
    "AnalyticalSkills", "Presentation_Skills", "Networking_Skills",
    "Certifications"
]

# Define a sample test input
test_data = {
    "GPA": 4.0,
    "Extracurriculars": "No",
    "InternshipExperience": "Yes",
    "Projects": 5,
    "Leadership_Positions": 2,
    "Courses": "Yes",
    "Research_Experience": 4,
    "Coding_Skills": 5,
    "Communication_Skills": 3,
    "Problem_Solving_Skills": 5,
    "Teamwork_Skills": 3,
    "AnalyticalSkills": 5,
    "Presentation_Skills": 2,
    "Networking_Skills": 2,
    "Certifications": "Yes"
}

# Convert categorical values ("Yes"/"No") to numbers
for key in test_data:
    if test_data[key] == "Yes":
        test_data[key] = 1
    elif test_data[key] == "No":
        test_data[key] = 0

# Convert test data into DataFrame
features = pd.DataFrame([test_data])

# 🔥 Ensure all expected features are present (fill missing ones with 0)
for feature in expected_features:
    if feature not in features:
        features[feature] = 0

# Ensure correct column order
features = features[expected_features]

# 🔥 Apply feature scaling
features_scaled = scaler.transform(features)

# Make a prediction
probabilities = model.predict_proba(features_scaled)[0]
career_classes = model.classes_

# Get the most likely career (highest probability)
most_likely_career = career_classes[probabilities.argmax()]

# Display only the most likely career
print(f"\n🎯 Most Likely Career: {most_likely_career}")
