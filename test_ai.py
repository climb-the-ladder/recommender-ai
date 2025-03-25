import joblib
import os
import pandas as pd

script_dir = os.path.dirname(os.path.abspath(__file__))
model = joblib.load(os.path.join(script_dir, "../recommender-models/career_xgb.pkl"))
scaler = joblib.load(os.path.join(script_dir, "../recommender-models/scaler.pkl"))
label_encoder = joblib.load(os.path.join(script_dir, "../recommender-models/label_encoder.pkl"))

# Example test data engineered to lean towards "Software Engineer"
test_data = {

    #software engineer:
    # "math_score": 95,
    # "history_score": 70,
    # "physics_score": 96,
    # "chemistry_score": 85,
    # "biology_score": 78,
    # "english_score": 88,
    # "geography_score": 75

    #doctor:
    # "math_score": 80,
    # "history_score": 68,
    # "physics_score": 88,
    # "chemistry_score": 95,
    # "biology_score": 94,  # High bio
    # "english_score": 82,
    # "geography_score": 65

    #real estate developer:
    # "math_score": 50,
    # "history_score": 92,
    # "physics_score": 45,
    # "chemistry_score": 60,
    # "biology_score": 70,
    # "english_score": 95,
    # "geography_score": 88

    #lawyer
    # "math_score": 70,
    # "history_score": 90,   # Strong in history
    # "physics_score": 60,
    # "chemistry_score": 65,
    # "biology_score": 70,
    # "english_score": 95,   # Strong English
    # "geography_score": 80

    #scientist - correct
    # "math_score": 92,
    # "history_score": 75,
    # "physics_score": 94,
    # "chemistry_score": 96,
    # "biology_score": 89,
    # "english_score": 80,
    # "geography_score": 78

}

features = pd.DataFrame([test_data])
features_scaled = scaler.transform(features)
predicted_label = model.predict(features_scaled)[0]

predicted_career = label_encoder.inverse_transform([predicted_label])[0]
print(f"🎯 Recommended Career: {predicted_career}")
