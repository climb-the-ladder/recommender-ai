import joblib
import os
import pandas as pd

script_dir = os.path.dirname(os.path.abspath(__file__))
model = joblib.load(os.path.join(script_dir, "../recommender-models/career_xgb.pkl"))
scaler = joblib.load(os.path.join(script_dir, "../recommender-models/scaler.pkl"))
label_encoder = joblib.load(os.path.join(script_dir, "../recommender-models/label_encoder.pkl"))

# Example test data engineered to lean towards "Software Engineer"
test_data = {
    "math_score": 95,
    "history_score": 70,
    "physics_score": 96,
    "chemistry_score": 85,
    "biology_score": 78,
    "english_score": 88,
    "geography_score": 75
}

features = pd.DataFrame([test_data])
features_scaled = scaler.transform(features)
predicted_label = model.predict(features_scaled)[0]

predicted_career = label_encoder.inverse_transform([predicted_label])[0]
print(f"🎯 Recommended Career: {predicted_career}")
