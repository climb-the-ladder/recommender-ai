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

def handle_chat(message, career=None, gpa=None, session_id="default"):
    """
    Handle chat messages using OpenAI's API
    
    Args:
        message: The user's message
        career: The user's selected career
        gpa: The user's GPA
        session_id: Unique identifier for the chat session
    """
    # Initialize chat history for this session if it doesn't exist
    if session_id not in chat_histories:
        chat_histories[session_id] = []
    
    # Get university recommendations if we have GPA and career
    university_context = ""
    if gpa and career:
        try:
            unis, similar_careers = career_chatbot.recommend(float(gpa), career)
            if unis:
                university_context = f"The user has a GPA of {gpa}/100 and is interested in a career as a {career}. "
                university_context += f"Based on this, these universities are recommended: "
                university_context += ", ".join([f"{uni['University_Name']} ({uni['Rank_Tier']})" for uni in unis[:5]])
                
                if similar_careers:
                    university_context += f". Similar careers to {career} include: {', '.join(similar_careers[:3])}"
        except Exception as e:
            print(f"Error getting university recommendations: {str(e)}")
    
    # Create system message with context
    system_message = f"""You are a helpful career advisor specializing in {career if career else 'various careers'}.
    Be conversational and friendly. {university_context}
    
    If the user asks about universities and you have university recommendations, share them.
    If they ask about universities but you don't have their GPA, ask for it.
    
    Focus on career advice, education requirements, and university recommendations.
    """
    
    # Build the conversation history for the API
    messages = [{"role": "system", "content": system_message}]
    
    # Add chat history (limited to last 5 exchanges to save tokens)
    for i, (past_msg, past_resp) in enumerate(zip(chat_histories[session_id][::2], chat_histories[session_id][1::2])):
        if i >= 5:  # Only include the last 5 exchanges
            break
        messages.append({"role": "user", "content": past_msg})
        messages.append({"role": "assistant", "content": past_resp})
    
    # Add the current message
    messages.append({"role": "user", "content": message})
    
    try:
        # Call the OpenAI API using the new format
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0.7,
            max_tokens=500
        )
        
        response_text = response.choices[0].message.content
        
        # Update chat history
        chat_histories[session_id].append(message)
        chat_histories[session_id].append(response_text)
        
        return response_text
        
    except Exception as e:
        print(f"OpenAI API error: {str(e)}")
        return f"I'm having trouble connecting to my knowledge base. Please try again in a moment. Error: {str(e)}"

def rule_based_response(message, career=None, gpa=None):
    """
    Fallback rule-based response system
    """
    message = message.lower()
    
    # Check if the message is about universities
    if any(word in message for word in ['university', 'college', 'school', 'education', 'universities']):
        if gpa and career:
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
    
    # Check if the message is about career information
    elif any(word in message for word in ['salary', 'pay', 'earn', 'income']):
        salary_info = {
            'Software Engineer': 'Software Engineers typically earn between $70,000 and $150,000 depending on experience and location.',
            'Data Scientist': 'Data Scientists typically earn between $80,000 and $160,000 depending on experience and location.',
            'Doctor': 'Doctors typically earn between $150,000 and $300,000+ depending on specialty and experience.',
            'Lawyer': 'Lawyers typically earn between $60,000 and $180,000 depending on specialty and location.',
            'Scientist': 'Scientists typically earn between $60,000 and $130,000 depending on field and experience.',
            'Artist': 'Artists\' incomes vary widely, from $20,000 to $100,000+ depending on medium, recognition, and business skills.'
        }
        
        if career in salary_info:
            return salary_info[career]
        else:
            return f"I don't have specific salary information for {career}, but you can research current market rates on job sites like Indeed or Glassdoor."
    
    # Check if the message is about skills
    elif any(word in message for word in ['skill', 'learn', 'know', 'ability']):
        skills_info = {
            'Software Engineer': 'Important skills for Software Engineers include programming languages (like Python, Java, JavaScript), problem-solving, algorithms, data structures, and teamwork.',
            'Data Scientist': 'Data Scientists need skills in statistics, machine learning, Python or R programming, data visualization, and domain knowledge.',
            'Doctor': 'Doctors need strong knowledge of medicine, biology, chemistry, critical thinking, communication, and empathy.',
            'Lawyer': 'Lawyers need skills in critical thinking, research, writing, public speaking, negotiation, and knowledge of laws and regulations.',
            'Scientist': 'Scientists need analytical thinking, research methods, statistics, technical writing, and specialized knowledge in their field.',
            'Artist': 'Artists need creativity, technical skills in their medium, visual communication, marketing, networking, and business management.'
        }
        
        if career in skills_info:
            return skills_info[career]
        else:
            return f"For a career in {career}, you'll likely need a mix of technical skills specific to the field and soft skills like communication, problem-solving, and teamwork."
    
    # General response
    else:
        return f"I'm here to help with your questions about pursuing a career as a {career}. You can ask about required education, skills, job outlook, or anything else you'd like to know!"
