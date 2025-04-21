# recommender-ai/gpt_chatbot.py

import os
from openai import OpenAI
from dotenv import load_dotenv
from chatbot import CareerChatbot

# Load environment variables and API key
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Initialize the career chatbot
career_chatbot = CareerChatbot()

# Store chat history
chat_histories = {}

def handle_chat(message, career=None, gpa=None, subject_grades=None, session_id="default"):
    """
    Handle chat messages using OpenAI's API with fallback to rule-based responses
    
    Args:
        message: The user's message
        career: The user's selected career
        gpa: The user's overall GPA
        subject_grades: Dictionary of subject grades
        session_id: Unique identifier for the chat session
    """
    # Initialize chat history for this session if it doesn't exist
    if session_id not in chat_histories:
        chat_histories[session_id] = []
    
    if subject_grades is None:
        subject_grades = {}
    
    # Get university and career information
    university_info = ""
    similar_careers_info = ""
    grades_info = ""
    
    # Format subject grades information
    if subject_grades:
        grades_info = "\nSubject Grades:\n"
        for subject, grade in subject_grades.items():
            grades_info += f"- {subject}: {grade}/100\n"
    
    if career and gpa:
        try:
            # Convert GPA to float if it's a string
            if isinstance(gpa, str):
                gpa = float(gpa)
                
            # Get university recommendations and similar careers
            unis, similar_careers = career_chatbot.recommend(gpa, career)
            
            # Format university information
            if unis:
                university_info = f"\n\nBased on a GPA of {gpa}/100 and interest in {career}, these universities are recommended:\n"
                for i, uni in enumerate(unis[:5], 1):
                    university_info += f"{i}. {uni['University_Name']} ({uni['Rank_Tier']})\n"
                
                if len(unis) > 5:
                    university_info += f"\nThere are {len(unis) - 5} more universities that match these criteria."
            else:
                university_info = f"\n\nNo specific universities found for {career} with a GPA of {gpa}/100."
            
            # Format similar careers information
            if similar_careers:
                similar_careers_info = f"\n\nSimilar careers to {career} include: {', '.join(similar_careers[:5])}"
        except Exception as e:
            print(f"Error getting recommendations: {str(e)}")
    
    try:
        # Try to use OpenAI API
        response_text = get_openai_response(message, career, gpa, subject_grades, university_info, similar_careers_info, grades_info, session_id)
        
        # Update chat history
        chat_histories[session_id].append(message)
        chat_histories[session_id].append(response_text)
        
        return response_text
        
    except Exception as e:
        print(f"OpenAI API error: {str(e)}")
        # Fall back to rule-based responses
        return get_fallback_response(message, career, gpa, subject_grades, university_info, similar_careers_info, grades_info)

def get_openai_response(message, career=None, gpa=None, subject_grades=None, university_info="", similar_careers_info="", grades_info="", session_id="default"):
    """Get response from OpenAI API"""
    
    # Create system message with context
    system_message = f"""You are a specialized career advisor focused exclusively on providing information about {career if career else 'various careers'}, education requirements, university recommendations, and related career paths.

    USER INFORMATION:
    Career Interest: {career if career else 'Not specified'}
    Overall GPA: {gpa if gpa else 'Not specified'}/100
    {grades_info}
    
    UNIVERSITY RECOMMENDATIONS:{university_info}
    
    SIMILAR CAREERS:{similar_careers_info}
    
    IMPORTANT INSTRUCTIONS:
    1. ONLY answer questions related to careers, education, universities, academic performance, and professional development.
    2. If the user asks about topics outside this scope (sports, politics, entertainment, general knowledge, etc.), politely redirect them to career-related topics.
    3. For off-topic questions, respond with: "I specialize in providing career advice for {career}, university recommendations, and information on related careers. If you have any questions related to those topics or if there's anything else I can assist you with, feel free to let me know!"
    4. Be conversational and friendly, but stay strictly within your defined scope.
    5. If the user asks about universities and you have university recommendations, share them.
    6. If they ask about universities but you don't have their GPA, ask for it.
    7. If they ask about their grades or academic performance, refer to their subject grades if available.
    
    Your primary purpose is to help users with career guidance and educational planning, not to be a general-purpose assistant.
    """
    
    # Build the conversation history for the API
    messages = [{"role": "system", "content": system_message}]
    
    # Add chat history (limited to last 5 exchanges to save tokens)
    history = chat_histories.get(session_id, [])
    for i in range(0, min(len(history), 10), 2):
        if i/2 >= 5:  # Only include the last 5 exchanges
            break
        messages.append({"role": "user", "content": history[i]})
        if i+1 < len(history):
            messages.append({"role": "assistant", "content": history[i+1]})
    
    # Add the current message
    messages.append({"role": "user", "content": message})
    
    # Call the OpenAI API using the new format
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0.7,
        max_tokens=500
    )
    
    return response.choices[0].message.content

def get_fallback_response(message, career=None, gpa=None, subject_grades=None, university_info="", similar_careers_info="", grades_info=""):
    """Provide rule-based responses when API is unavailable"""
    message = message.lower()
    
    # Check if the message is off-topic
    off_topic_keywords = ['sports', 'game', 'movie', 'music', 'politics', 'news', 'weather', 'celebrity', 'tv', 'show', 'football', 'basketball', 'baseball', 'soccer', 'tennis']
    if any(keyword in message for keyword in off_topic_keywords):
        return f"I specialize in providing career advice for {career}, university recommendations, and information on related careers. If you have any questions related to those topics or if there's anything else I can assist you with, feel free to let me know!"
    
    # Check if the message is about grades
    if any(word in message for word in ['grade', 'grades', 'score', 'marks', 'gpa']):
        if grades_info:
            return f"Here are your academic grades:{grades_info}\nYour overall GPA is {gpa}/100."
        elif gpa:
            return f"Your overall GPA is {gpa}/100. I don't have information about your individual subject grades."
        else:
            return "I don't have information about your grades. What's your GPA on a scale of 0-100?"
    
    # Check if the message is about universities
    elif any(word in message for word in ['university', 'college', 'school', 'education', 'universities']):
        if university_info:
            return f"Here are university recommendations for a career in {career} with a GPA of {gpa}:{university_info}"
        elif gpa and career:
            try:
                unis, _ = career_chatbot.recommend(float(gpa), career)
                if unis:
                    response = f"Based on your GPA of {gpa} and interest in {career}, here are some university recommendations:\n\n"
                    for i, uni in enumerate(unis[:5], 1):
                        response += f"{i}. {uni['University_Name']} ({uni['Rank_Tier']})\n"
                    
                    if len(unis) > 5:
                        response += f"\nThere are {len(unis) - 5} more universities that match your criteria."
                    
                    return response
                else:
                    return f"I couldn't find universities matching your criteria for {career} with a GPA of {gpa}. Consider improving your GPA or exploring related fields."
            except Exception as e:
                return f"I encountered an error while searching for universities: {str(e)}"
        else:
            return "To recommend universities, I need to know your GPA. What's your GPA on a scale of 0-100?"
    
    # Check if the message is about similar careers
    elif any(word in message for word in ['similar', 'alternative', 'other career', 'other careers']):
        if similar_careers_info:
            return f"Here are some career alternatives you might consider:{similar_careers_info}"
        elif career:
            try:
                _, similar_careers = career_chatbot.recommend(70, career)  # Use a default GPA just to get similar careers
                if similar_careers:
                    return f"Similar careers to {career} include: {', '.join(similar_careers[:5])}"
                else:
                    return f"I don't have information about careers similar to {career}."
            except Exception as e:
                return f"I encountered an error while searching for similar careers: {str(e)}"
        else:
            return "To suggest similar careers, I need to know what career you're interested in."
    
    # General response
    else:
        return f"I'm here to help with your questions about pursuing a career as a {career}. You can ask about required education, skills, job outlook, or anything else you'd like to know!"
