import google.generativeai as genai
import os

class DentalChatbot:
    def __init__(self, api_key=None):
        if api_key:
            genai.configure(api_key=api_key)
            self.api_key = api_key
        else:
            api_key = os.getenv('GEMINI_API_KEY')
            if api_key:
                genai.configure(api_key=api_key)
                self.api_key = api_key
            else:
                self.api_key = None
        
        if self.api_key:
            # Using Gemini 2.5 Pro model
            self.model = genai.GenerativeModel('gemini-2.5-flash')
        else:
            self.model = None
        
        self.system_prompt = """You are a helpful dental health assistant. Your role is to provide accurate, 
        patient-friendly information about:
        
        1. Dental conditions and their causes
        2. Oral hygiene practices and recommendations
        3. Dental treatments and procedures
        4. Foods that promote or harm dental health
        5. Medications related to dental care
        6. Prevention of dental problems
        7. When to see a dentist
        
        Important guidelines:
        - Provide clear, understandable explanations
        - Always recommend consulting a dentist for proper diagnosis and treatment
        - Never provide medical diagnoses
        - Focus on education and prevention
        - Be empathetic and supportive
        - Provide evidence-based information
        
        If asked about non-dental topics, politely redirect to dental health topics.
        """
    
    def chat(self, user_message, conversation_history=None):
        if not self.api_key or not self.model:
            return {
                'response': "Chatbot is not configured. Please set up your Gemini API key to use the chatbot feature.",
                'error': True
            }
        
        try:
            full_prompt = f"{self.system_prompt}\n\nUser: {user_message}\n\nAssistant:"
            
            if conversation_history:
                history_text = "\n".join([
                    f"{'User' if msg['role'] == 'user' else 'Assistant'}: {msg['content']}"
                    for msg in conversation_history[-5:]
                ])
                full_prompt = f"{self.system_prompt}\n\nConversation History:\n{history_text}\n\nUser: {user_message}\n\nAssistant:"
            
            response = self.model.generate_content(full_prompt)
            
            return {
                'response': response.text,
                'error': False
            }
        
        except Exception as e:
            error_message = str(e)
            
            if "API_KEY" in error_message.upper() or "API key" in error_message:
                return {
                    'response': "API key error. Please configure your Gemini API key in the settings.",
                    'error': True
                }
            else:
                return {
                    'response': f"Sorry, I encountered an error: {error_message}. Please try again.",
                    'error': True
                }
    
    def get_dental_tips(self):
        tips = [
            "Brush your teeth at least twice a day with fluoride toothpaste",
            "Floss daily to remove plaque between teeth",
            "Limit sugary and acidic foods and drinks",
            "Visit your dentist regularly for check-ups and cleanings",
            "Replace your toothbrush every 3-4 months",
            "Drink plenty of water throughout the day",
            "Avoid tobacco products",
            "Use mouthwash to help reduce plaque and bacteria",
            "Eat a balanced diet rich in calcium and vitamins",
            "Protect your teeth during sports with a mouthguard"
        ]
        return tips
    
    def is_configured(self):
        return self.api_key is not None and self.model is not None
