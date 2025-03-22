from flask import Flask, request, jsonify
import os
from flask_cors import CORS
import dotenv

# Load environment variables
dotenv.load_dotenv()

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Fallback data for testing when models aren't available
fallback_careers = ["Software Engineer", "Data Scientist", "Product Manager", "UX Designer"]

# Try to import optional dependencies
model = None
joblib_available = False
pandas_available = False

try:
    import joblib
    joblib_available = True
except ImportError:
    print("Warning: joblib not available")

try:
    import pandas as pd
    pandas_available = True
except ImportError:
    print("Warning: pandas not available")

# Only attempt to load models if dependencies are available
if joblib_available and pandas_available:
    try:
        # Update paths to match the Docker Compose volume mount
        model_path = os.path.join("/app/models", "career_recommender.pk1")
        scaler_path = os.path.join("/app/models", "scaler.pkl")
        encoder_path = os.path.join("/app/models", "label_encoder.pkl")
        
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            print("Model loaded successfully")
        else:
            print(f"Warning: Model file not found at {model_path}")
            
        scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None
        label_encoder = joblib.load(encoder_path) if os.path.exists(encoder_path) else None
    except Exception as e:
        print(f"Warning: Error loading models: {str(e)}")
        model = None
        scaler = None
        label_encoder = None

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
        
        # If dependencies aren't available or model isn't loaded, use fallback behavior
        if not joblib_available or not pandas_available or model is None:
            import random
            return jsonify({
                "career": random.choice(fallback_careers), 
                "note": "Using fallback model (models not available)"
            })
            
        features = pd.DataFrame([data])

        # Map frontend field names to expected feature names if needed
        field_mapping = {
            "gpa": "GPA",
            "extracurriculars": "Extracurriculars",
            "internships": "InternshipExperience",
            "projects": "Projects",
            "leadership": "Leadership_Positions",
            "fieldCourses": "Courses",
            "research": "Research_Experience",
            "coding": "Coding_Skills",
            "communication": "Communication_Skills",
            "problemSolving": "Problem_Solving_Skills",
            "teamwork": "Teamwork_Skills",
            "analytical": "AnalyticalSkills",
            "presentation": "Presentation_Skills",
            "networking": "Networking_Skills",
            "certifications": "Certifications"
        }
        
        # Convert field names
        transformed_data = {}
        for frontend_field, model_field in field_mapping.items():
            if frontend_field in data:
                transformed_data[model_field] = data[frontend_field]
                
        features = pd.DataFrame([transformed_data])

        # Convert categorical variables
        categorical_columns = ["Extracurriculars", "InternshipExperience", "Courses", "Certifications"]
        for col in categorical_columns:
            if col in features.columns:
                features[col] = features[col].map({"Yes": 1, "No": 0})

        for feature in expected_features:
            if feature not in features:
                features[feature] = 0

        features = features[expected_features]

        # Scale features if scaler is available
        if scaler:
            features_scaled = scaler.transform(features)
            predicted_label = model.predict(features_scaled)[0]
        else:
            predicted_label = model.predict(features)[0]

        # Convert numeric label to career name if encoder is available
        if label_encoder:
            predicted_career = label_encoder.inverse_transform([predicted_label])[0]
        else:
            predicted_career = str(predicted_label)

        return jsonify({"career": predicted_career})

    except Exception as e:
        print(f"Prediction error: {str(e)}")
        import random
        return jsonify({
            "career": random.choice(fallback_careers), 
            "error": str(e),
            "note": "Using fallback due to error"
        })

@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({
        "status": "ok",
        "dependencies": {
            "joblib": joblib_available,
            "pandas": pandas_available
        },
        "model_loaded": model is not None
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    # Check if we're in production or development
    debug_mode = os.environ.get("FLASK_ENV", "development") != "production"
    app.run(host="0.0.0.0", port=port, debug=debug_mode)
