from flask import Flask, request, jsonify
import joblib
import os
import pandas as pd
from flask_cors import CORS
from chatbot import CareerChatbot  
from gpt_chatbot import handle_chat
from career_details import get_career_details
from career_roadmap import generate_career_roadmap
from alternative_careers import AlternativeCareersAnalyzer

app = Flask(__name__)

# Configure CORS
CORS(app, resources={
    r"/*": {
        "origins": ["http://localhost:3000"],
        "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"]
    }
})

script_dir = os.path.dirname(os.path.abspath(__file__))
model = joblib.load(os.path.join(script_dir, "../recommender-models/career_xgb.pkl"))
scaler = joblib.load(os.path.join(script_dir, "../recommender-models/scaler.pkl"))
label_encoder = joblib.load(os.path.join(script_dir, "../recommender-models/label_encoder.pkl"))

# Initialize services
chatbot = CareerChatbot()
alternative_careers_analyzer = AlternativeCareersAnalyzer()

# Expected input fields
expected_features = [
    "math_score", "history_score", "physics_score",
    "chemistry_score", "biology_score", "english_score", "geography_score"
]

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        print("\n📊 Received prediction request with data:", data)
        
        # Ensure all expected fields are present and convert to float
        features = {}
        for feature in expected_features:
            if feature not in data:
                return jsonify({"error": f"Missing required field: {feature}"}), 400
            try:
                features[feature] = float(data[feature])
            except (ValueError, TypeError):
                return jsonify({"error": f"Invalid value for {feature}. Expected a number."}), 400

        # Create DataFrame with the features
        features_df = pd.DataFrame([features])
        features_df = features_df[expected_features]
        
        # Scale the features
        features_scaled = scaler.transform(features_df)
        print("🔢 Scaled features:", features_scaled)

        # Predict the career
        predicted_label = model.predict(features_scaled)[0]
        predicted_career = label_encoder.inverse_transform([predicted_label])[0]
        print("🎯 Predicted career:", predicted_career)

        return jsonify({
            "career": predicted_career,
            "confidence_score": 0.85  # Example confidence score
        })

    except Exception as e:
        print("❌ Error in prediction:", str(e))
        return jsonify({"error": str(e)}), 500

@app.route("/chatbot-recommend", methods=["POST"])
def chatbot_recommend():
    """Get initial similar careers list."""
    try:
        data = request.json
        career = data.get('career')

        if not career:
            return jsonify({"error": "Career is required"}), 400

        similar_careers = alternative_careers_analyzer.get_similar_careers(career)
        return jsonify({
            "similar_careers": similar_careers[:5]  # Limit to top 5 alternatives
        })
    
    except Exception as e:
        print(f"Error in chatbot recommend: {str(e)}")
        return jsonify({"error": str(e)}), 400

@app.route("/analyze-careers", methods=["POST"])
def analyze_careers():
    """Analyze careers with academic scores."""
    try:
        data = request.json
        careers = data.get('careers', [])
        academic_scores = data.get('academic_scores', {})
        predicted_career = data.get('predicted_career')

        if not careers or not academic_scores or not predicted_career:
            return jsonify({"error": "Missing required data"}), 400

        analyzed_careers = []
        for career in careers:
            analysis = alternative_careers_analyzer.analyze_career_match(
                career, 
                academic_scores, 
                predicted_career
            )
            analyzed_careers.append({
                "career": career,
                "matching_score": analysis["matching_score"],
                "explanation": analysis["explanation"],
                "key_skills": analysis["key_skills"]
            })

        # Sort by matching score
        analyzed_careers.sort(key=lambda x: x["matching_score"], reverse=True)
        
        return jsonify({
            "success": True,
            "analyzed_careers": analyzed_careers
        })

    except Exception as e:
        print(f"Error in analyze careers: {str(e)}")
        return jsonify({"error": str(e)}), 400

#new route for our chatbot
@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.json
        user_input = data.get("message", "")
        if not user_input:
            return jsonify({"error": "Message is required"}), 400

        history = []  # in-memory; store in session/db for longer chat
        response = handle_chat(user_input, history)
        return jsonify({"response": response})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/career-details", methods=["POST"])
def career_details():
    """Handles requests for detailed career information."""
    try:
        data = request.json
        career = data.get('career')
        
        print(f"Career details requested in AI service for: {career}")

        if not career:
            return jsonify({"error": "Career is required", "success": False}), 400

        # Get detailed information about the career
        print(f"Calling OpenAI for career details: {career}")
        details = get_career_details(career)
        
        print(f"Career details success: {details.get('success', False)}")
        
        return jsonify(details)
    
    except Exception as e:
        error_msg = f"Error getting career details: {str(e)}"
        print(f"Error in career details endpoint: {error_msg}")
        return jsonify({"error": error_msg, "success": False}), 400

@app.route("/career-roadmap", methods=["POST"])
def career_roadmap():
    """Handles requests for career roadmap generation."""
    try:
        data = request.json
        career = data.get('career')
        subject_grades = data.get('subject_grades', {})
        gpa = data.get('gpa')
        
        print(f"Career roadmap requested in AI service for: {career}")

        if not career:
            return jsonify({"error": "Career is required", "success": False}), 400

        # Generate roadmap for the career
        print(f"Calling OpenAI for career roadmap: {career}")
        roadmap = generate_career_roadmap(career, subject_grades, gpa)
        
        print(f"Career roadmap success: {roadmap.get('success', False)}")
        
        return jsonify(roadmap)
    
    except Exception as e:
        error_msg = f"Error generating career roadmap: {str(e)}"
        print(f"Error in career roadmap endpoint: {error_msg}")
        return jsonify({"error": error_msg, "success": False}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
