from flask import Flask, request, jsonify
import joblib
import os
import pandas as pd

app = Flask(__name__)

# Load trained model
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, "../recommender-models/career_recommender.pkl")
model = joblib.load(model_path)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json  # User input
    features = pd.DataFrame([data])  # Convert to DataFrame

    # Get probability predictions for all careers
    probabilities = model.predict_proba(features)[0]
    career_classes = model.classes_

    # Sort careers by probability
    career_predictions = sorted(
        zip(career_classes, probabilities), key=lambda x: x[1], reverse=True
    )

    # Return the top 3 career predictions
    top_careers = [{"career": career, "probability": round(prob * 100, 2)} for career, prob in career_predictions[:3]]

    return jsonify({"predictions": top_careers})

if __name__ == "__main__":
    app.run(port=5001, debug=True)
