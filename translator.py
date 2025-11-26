from deep_translator import GoogleTranslator
import time

class MultilingualTranslator:
    def __init__(self):
        self.supported_languages = {
            'tamil': 'ta',
            'telugu': 'te',
            'hindi': 'hi',
            'malayalam': 'ml',
            'kannada': 'kn',
            'english': 'en'
        }
        
        self.dental_terms = {
            'cavity': {
                'ta': 'பல் குழி',
                'te': 'దంత గుహ',
                'hi': 'दांत में सड़न',
                'ml': 'പല്ല് ദ്വാരം',
                'kn': 'ಹಲ್ಲಿನ ಕುಳಿ'
            },
            'restoration': {
                'ta': 'பல் பழுது',
                'te': 'దంత మరమ్మత్తు',
                'hi': 'दांत की बहाली',
                'ml': 'ദന്ത പുനരുദ്ധാരം',
                'kn': 'ದಂತ ಪುನರುದ್ದಾರ'
            },
            'impacted tooth': {
                'ta': 'பதிந்த பல்',
                'te': 'ప్రభావిత దంతము',
                'hi': 'फंसा हुआ दांत',
                'ml': 'തളർന്ന പല്ല്',
                'kn': 'ಅಡಗಿದ ಹಲ್ಲು'
            },
            'wisdom tooth': {
                'ta': 'அறிவுப்பல்',
                'te': 'విజ్డమ్ టూత్',
                'hi': 'अक्ल दाढ़',
                'ml': 'ജ്ഞാനദന്തം',
                'kn': 'ಜ್ಞಾನ ಹಲ್ಲು'
            },
            'misalignment': {
                'ta': 'சீரற்ற அமைப்பு',
                'te': 'అసమాన అమరిక',
                'hi': 'गलत संरेखण',
                'ml': 'അസന്തുലിത സ്ഥാനം',
                'kn': 'ಅಸಮತೋಲನ'
            },
            'bone loss': {
                'ta': 'எலும்பு இழப்பு',
                'te': 'ఎముక నష్టం',
                'hi': 'हड्डी की क्षति',
                'ml': 'അസ്ഥി നഷ്ടം',
                'kn': 'ಎಲುಬಿನ ನಷ್ಟ'
            },
            'X-ray': {
                'ta': 'எக்ஸ்-ரே',
                'te': 'ఎక్స్-రే',
                'hi': 'एक्स-रे',
                'ml': 'എക്സ്-റേ',
                'kn': 'ಎಕ್ಸ್-ರೇ'
            }
        }
    
    def translate_text(self, text, target_language):
        if target_language.lower() == 'english' or target_language.lower() == 'en':
            return text
        
        lang_code = self.supported_languages.get(target_language.lower())
        if not lang_code:
            return f"Unsupported language: {target_language}"
        
        try:
            time.sleep(0.1)
            
            translator = GoogleTranslator(source='en', target=lang_code)
            translated_text = translator.translate(text)
            
            for term, translations in self.dental_terms.items():
                if term in text.lower() and lang_code in translations:
                    term_translator = GoogleTranslator(source='en', target=lang_code)
                    term_translation = term_translator.translate(term)
                    translated_text = translated_text.replace(
                        term_translation,
                        translations[lang_code]
                    )
            
            return translated_text
        except Exception as e:
            return f"Translation error: {str(e)}"
    
    def translate_to_all_languages(self, text):
        translations = {'english': text}
        
        for lang_name, lang_code in self.supported_languages.items():
            if lang_name != 'english':
                try:
                    translated = self.translate_text(text, lang_name)
                    translations[lang_name.title()] = translated
                    time.sleep(0.2)
                except Exception as e:
                    translations[lang_name.title()] = f"Error: {str(e)}"
        
        return translations
    
    def get_supported_languages(self):
        return list(self.supported_languages.keys())
