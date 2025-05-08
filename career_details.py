import os
from openai import OpenAI
from dotenv import load_dotenv
import json


# Load environment variables and API key
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def get_career_details(career_name):
    """
    Get detailed information about a career using OpenAI's API
    
    Args:
        career_name: The name of the career to get details for
        
    Returns:
        A dictionary containing various details about the career
    """
    try:
        # Create a prompt for OpenAI to generate structured information about the career        
        user_prompt = f"""You are a career information specialist that provides accurate, concise details about careers in JSON format.

        Provide detailed information about a career as a {career_name}. 
        Return the response in JSON format with the following fields:
        - description: A 2-3 sentence overview of the career
        - salary_range: The typical salary range for this career (e.g. $X-$Y per year)
        - difficulty: Rating from 1-10 how challenging this career is to pursue
        - education: Required educational background
        - skills: List of 5 key skills needed
        - job_outlook: Future prospects for this career field
        - day_to_day: Brief description of typical day-to-day activities
        - advancement: Career advancement opportunities
        - work_life_balance: An object with 'rating' (number 1-10) and 'explanation' (brief text explanation)
        - pros: List of 3 advantages of this career
        - cons: List of 3 challenges or disadvantages
        
        Ensure the work_life_balance field is structured as an object with 'rating' and 'explanation' properties.
        """
        
        try:
            # Use gpt-3.5-turbo model which is more reliable
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7
            )
            career_data = response.choices[0].message.content
            
        except Exception as api_error:
            print(f"Error with primary model: {str(api_error)}")
            # Try an even simpler fallback without JSON format requirements
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7
            )
            career_data = response.choices[0].message.content
        
        # Validate the response has the correct work_life_balance structure
        try:
            parsed_data = json.loads(career_data)
            if not isinstance(parsed_data.get('work_life_balance'), dict):
                # Fix the work_life_balance field if it's not an object
                if isinstance(parsed_data.get('work_life_balance'), (int, str)):
                    value = parsed_data.get('work_life_balance')
                    rating = int(value) if isinstance(value, int) else 5
                    parsed_data['work_life_balance'] = {
                        "rating": rating,
                        "explanation": "Work-life balance details unavailable"
                    }
            career_data = json.dumps(parsed_data)
        except (json.JSONDecodeError, TypeError, ValueError):
            # If there's an error parsing, we'll just return the original data
            pass
        
        return {"success": True, "data": career_data}
        
    except Exception as e:
        print(f"Error getting career details: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "data": json.dumps({
                "description": f"Information about {career_name} is currently unavailable.",
                "salary_range": "Data unavailable",
                "difficulty": "Data unavailable",
                "education": "Data unavailable",
                "skills": ["Data unavailable"],
                "job_outlook": "Data unavailable",
                "day_to_day": "Data unavailable",
                "advancement": "Data unavailable",
                "work_life_balance": {
                    "rating": 5,
                    "explanation": "Data unavailable"
                },
                "pros": ["Data unavailable"],
                "cons": ["Data unavailable"]
            })
        } 