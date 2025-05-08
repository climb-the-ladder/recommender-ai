import os
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
import json
from typing import Dict, List

# Load environment variables
load_dotenv()

class AlternativeCareersAnalyzer:
    def __init__(self):
        self.careers_df = pd.read_csv('recommender-data/raw/similar_careers_dataset.csv')
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
    def get_similar_careers(self, career):
        try:
            similar_careers = self.careers_df[self.careers_df['Career_Field'] == career]['Similar_Careers'].iloc[0]
            return [c.strip() for c in similar_careers.split(',')]
        except (IndexError, KeyError):
            return []

    def get_career_requirements(self, career):
        """
        Get typical academic requirements for a career
        """
        career_requirements = {
            'Software Engineer': ['Mathematics', 'Physics', 'Computer Science'],
            'Data Scientist': ['Mathematics', 'Statistics', 'Computer Science'],
            'Doctor': ['Biology', 'Chemistry', 'Physics'],
            'Lawyer': ['English', 'History', 'Social Studies'],
            'Architect': ['Mathematics', 'Physics', 'Art'],
            'Real Estate Developer': ['Mathematics', 'Economics', 'Geography'],
            'Property Manager': ['Mathematics', 'Economics', 'Business'],
            'Urban Planner': ['Geography', 'Mathematics', 'Social Studies'],
            'Real Estate Agent': ['Economics', 'Mathematics', 'Communication'],
            'Construction Manager': ['Mathematics', 'Physics', 'Management'],
        }
        
        return career_requirements.get(career, ['Mathematics', 'English', 'Critical Thinking'])

    def calculate_subject_match(self, academic_scores: Dict, required_subjects: List[str]) -> float:
        """
        Calculate how well the academic scores match required subjects
        """
        subject_mapping = {
            'Mathematics': 'subject_mathematics',
            'Physics': 'subject_physics',
            'Chemistry': 'subject_chemistry',
            'Biology': 'subject_biology',
            'English': 'subject_english',
            'Geography': 'subject_geography',
            'History': 'subject_history'
        }
        
        total_score = 0
        relevant_subjects = 0
        
        for subject in required_subjects:
            if subject in subject_mapping:
                score_key = subject_mapping[subject]
                if score_key in academic_scores and academic_scores[score_key] is not None:
                    total_score += academic_scores[score_key]
                    relevant_subjects += 1
        
        return (total_score / relevant_subjects) if relevant_subjects > 0 else 70

    def analyze_career_match(self, career: str, academic_scores: Dict, predicted_career: str) -> Dict:
        """
        Analyze how well a career matches with the student's profile using GPT and academic alignment
        """
        # Get required subjects for this career
        required_subjects = self.get_career_requirements(career)
        
        # Calculate initial match score based on academic alignment
        subject_match_score = self.calculate_subject_match(academic_scores, required_subjects)
        
        formatted_academics = {
            "GPA": academic_scores.get("gpa", "Not provided"),
            "Subjects": {
                "Mathematics": academic_scores.get("subject_mathematics"),
                "Physics": academic_scores.get("subject_physics"),
                "Chemistry": academic_scores.get("subject_chemistry"),
                "Biology": academic_scores.get("subject_biology"),
                "English": academic_scores.get("subject_english"),
                "Geography": academic_scores.get("subject_geography"),
                "History": academic_scores.get("subject_history")
            }
        }

        prompt = f"""
        Analyze the suitability of '{career}' for a student with the following profile:

        Academic Profile:
        - GPA: {formatted_academics['GPA']}
        - Key Subject Scores: {json.dumps(formatted_academics['Subjects'], indent=2)}
        
        Career Context:
        - Current Career Interest: {predicted_career}
        - Required Subjects: {', '.join(required_subjects)}
        - Initial Subject Match Score: {subject_match_score:.1f}

        Task:
        1. Analyze how the student's academic strengths align with {career}
        2. Consider the relationship between {career} and {predicted_career}
        3. Evaluate subject performance in required areas: {', '.join(required_subjects)}

        Provide analysis in the specified JSON format.
        """

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system", 
                        "content": """You are a career counseling expert who provides detailed academic-based career analysis.
                        Focus on specific subjects and their relevance to careers.
                        Provide concrete explanations linking academic performance to career requirements.
                        Be specific about which subjects and skills matter for each career.
                        
                        Return ONLY valid JSON in the following format:
                        {
                            "matching_score": <score 0-100>,
                            "explanation": "<2-3 sentences>",
                            "key_skills": ["skill1", "skill2", "skill3"]
                        }"""
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=400,
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            
            # Ensure the matching score takes into account both GPT analysis and subject match
            final_score = (result["matching_score"] + subject_match_score) / 2
            
            return {
                "matching_score": round(final_score),
                "explanation": result["explanation"],
                "key_skills": result["key_skills"]
            }
            
        except Exception as e:
            print(f"Error in GPT analysis for {career}: {str(e)}")
            # Provide a more specific fallback based on subject match
            return {
                "matching_score": round(subject_match_score),
                "explanation": f"Based on your academic profile, you show strong potential in {', '.join(required_subjects[:2])} which are key requirements for a {career}.",
                "key_skills": [
                    f"Proficiency in {required_subjects[0]}",
                    f"Strong foundation in {required_subjects[1]}",
                    "Analytical and problem-solving abilities"
                ]
            }

    def get_alternative_careers(self, predicted_career: str, academic_scores: Dict) -> List[Dict]:
        """
        Get alternative careers with detailed analysis based on academic performance
        """
        similar_careers = self.get_similar_careers(predicted_career)
        
        analyzed_careers = []
        for career in similar_careers[:5]:  # Limit to top 5 alternatives
            analysis = self.analyze_career_match(career, academic_scores, predicted_career)
            analyzed_careers.append({
                "career": career,
                "matching_score": analysis["matching_score"],
                "explanation": analysis["explanation"],
                "key_skills": analysis["key_skills"]
            })
            
        # Sort by matching score
        analyzed_careers.sort(key=lambda x: x["matching_score"], reverse=True)
        return analyzed_careers 