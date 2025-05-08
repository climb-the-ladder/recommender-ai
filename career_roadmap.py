import os
from openai import OpenAI
from dotenv import load_dotenv
import json

# Load environment variables and API key
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def generate_career_roadmap(career, subject_grades, gpa=None):
    """
    Generate a career roadmap for a specific career based on user's academic performance
    
    Args:
        career: The predicted career path
        subject_grades: Dictionary of subject grades
        gpa: The user's overall GPA (optional)
    
    Returns:
        Dict containing structured roadmap data
    """
    # Format subject grades information
    grades_info = ""
    strengths = []
    areas_to_improve = []
    
    if subject_grades:
        grades_info = "Subject Grades:\n"
        # Find strengths and weaknesses
        for subject, grade in subject_grades.items():
            grade_val = float(grade) if grade else 0
            grades_info += f"- {subject.replace('_score', '')}: {grade_val}/100\n"
            
            if grade_val >= 70:
                strengths.append(subject.replace('_score', ''))
            elif grade_val < 50:
                areas_to_improve.append(subject.replace('_score', ''))
    
    # Build the prompt
    prompt = f"""You are a career roadmap expert. Provide detailed, structured career roadmaps to help people achieve their professional goals.

Generate a detailed career roadmap for someone pursuing a career as a {career}.

USER INFORMATION:
{grades_info}
Overall GPA: {gpa if gpa else 'Not specified'}/100

Strengths: {', '.join(strengths) if strengths else 'Not enough information'}
Areas to improve: {', '.join(areas_to_improve) if areas_to_improve else 'Not enough information'}

Your task is to create a structured career roadmap with the following sections:
1. Short-term goals (0-2 years)
2. Mid-term goals (2-5 years)
3. Long-term goals (5+ years)
4. Education requirements
5. Skills to develop
6. Experience needed
7. Industry certifications
8. Personal development recommendations
9. Networking suggestions
10. Timeline milestones (include specific years and durations)

For each section, provide specific, actionable advice tailored to this person's academic profile.
Format the response as a JSON object with these sections as keys and arrays of step-by-step guidance as values.
Use the exact keys: "short-term goals", "mid-term goals", "long-term goals", "education requirements", "skills to develop", "experience needed", "industry certifications", "personal development recommendations", "networking suggestions", "timeline_milestones".
Each key should have an array of strings as its value.
For timeline_milestones, each entry should be in the format: "Year X: [milestone description]"
Keep each step brief and actionable, and ensure the entire response is JSON-parsable.
"""

    try:
        # Call the OpenAI API
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=1500
        )
        
        # The response will be a JSON string
        roadmap_data = response.choices[0].message.content
        
        # Try to parse the JSON to ensure it's valid
        try:
            parsed_data = json.loads(roadmap_data)
            
            # Create a fallback structure if any required keys are missing
            required_keys = [
                "short-term goals", "mid-term goals", "long-term goals", 
                "education requirements", "skills to develop", "experience needed", 
                "industry certifications", "personal development recommendations", 
                "networking suggestions", "timeline_milestones"
            ]
            
            # Check for missing keys and add placeholders
            for key in required_keys:
                if key not in parsed_data:
                    if key == "timeline_milestones":
                        parsed_data[key] = [
                            "Year 1: Complete foundational courses",
                            "Year 2: Gain internship experience",
                            "Year 3: Complete degree requirements",
                            "Year 4: Secure entry-level position",
                            "Year 5: Pursue advanced certifications"
                        ]
                    else:
                        parsed_data[key] = ["Information not available"]
                elif not isinstance(parsed_data[key], list):
                    parsed_data[key] = [str(parsed_data[key])]
                
            roadmap_data = json.dumps(parsed_data)
            
        except json.JSONDecodeError as e:
            print(f"JSON parse error in roadmap generation: {e}")
            
            # Create a basic structure as fallback
            fallback_data = {
                "short-term goals": ["Begin by focusing on coursework in relevant subjects"],
                "mid-term goals": ["Pursue a bachelor's degree in a related field"],
                "long-term goals": ["Seek specialization and career advancement"],
                "education requirements": ["Bachelor's degree in relevant field"],
                "skills to develop": ["Critical thinking", "Problem-solving"],
                "experience needed": ["Internships in related field"],
                "industry certifications": ["Relevant professional certifications"],
                "personal development recommendations": ["Develop time management skills"],
                "networking suggestions": ["Join professional organizations"],
                "timeline_milestones": [
                    "Year 1: Complete foundational courses",
                    "Year 2: Gain internship experience",
                    "Year 3: Complete degree requirements",
                    "Year 4: Secure entry-level position",
                    "Year 5: Pursue advanced certifications"
                ]
            }
            roadmap_data = json.dumps(fallback_data)
        
        return {"success": True, "data": roadmap_data}
        
    except Exception as e:
        print(f"Error generating career roadmap: {str(e)}")
        return {"success": False, "error": str(e)} 