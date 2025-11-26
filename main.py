from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import importlib
import uuid
from werkzeug.utils import secure_filename
from dental_analyzer import DentalXrayAnalyzer
from caption_generator import DentalCaptionGenerator
from translator import MultilingualTranslator
from chatbot import DentalChatbot
from tts import MultilingualTTS
from vqa import VQAService

# Adapter selection order: VLM adapter (if enabled) → ML analyzer → rule-based
use_transformer = os.getenv('USE_TRANSFORMER_ADAPTER') == '1'
use_vlm = os.getenv('USE_VLM_ADAPTER') == '1'
analyzer = None
if use_transformer:
    try:
        from dent_adapt_transformer import DentAdaptTransformer
        analyzer = DentAdaptTransformer()
        print("Using Transformer-based adapter (CLIP)")
    except Exception:
        analyzer = None
if analyzer is None and use_vlm:
    try:
        from dent_adapt import DentAdapt
        cb_spec = os.getenv('VLM_ADAPTER_CALLBACK')
        cb_fn = None
        if cb_spec and ':' in cb_spec:
            mod_name, fn_name = cb_spec.split(':', 1)
            mod = importlib.import_module(mod_name)
            cb_fn = getattr(mod, fn_name, None)
        analyzer = DentAdapt(inference_fn=cb_fn)
        print("Using DentAdapt adapter (callback required)")
    except Exception:
        analyzer = None
if analyzer is None:
    try:
        from dental_analyzer_ml import DentalXrayAnalyzerML
        analyzer = DentalXrayAnalyzerML()
        print("Using ML-based analyzer (if model is trained)")
    except Exception:
        analyzer = DentalXrayAnalyzer()
        print("Using rule-based analyzer")

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
caption_gen = DentalCaptionGenerator()
translator = MultilingualTranslator()
chatbot = DentalChatbot()
# initialize TTS and VQA services (graceful if dependencies missing)
try:
    tts = MultilingualTTS(output_dir=os.path.join(app.config['UPLOAD_FOLDER'], 'tts'))
except Exception:
    tts = None

# Build adapter callback if specified for VLM adapter
cb_spec = os.getenv('VLM_ADAPTER_CALLBACK')
cb_fn = None
if cb_spec and ':' in cb_spec:
    try:
        mod_name, fn_name = cb_spec.split(':', 1)
        mod = importlib.import_module(mod_name)
        cb_fn = getattr(mod, fn_name, None)
    except Exception:
        cb_fn = None

# defer VQA model loading to first use (lazy loading) to speed up server startup
vqa_service = None
_vqa_initialized = False

def get_vqa_service():
    global vqa_service, _vqa_initialized
    if not _vqa_initialized:
        try:
            vqa_service = VQAService(adapter_callback=cb_fn)
        except Exception:
            vqa_service = None
        _vqa_initialized = True
    return vqa_service

conversation_history = []

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


@app.route('/tts/<path:filename>', methods=['GET'])
def serve_tts(filename):
    # Serve generated TTS audio files from uploads/tts
    tts_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'tts')
    return send_from_directory(tts_dir, filename)


@app.route('/speak', methods=['POST'])
def speak():
    data = request.json or {}
    text = data.get('text')
    lang = data.get('lang', 'en')
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    if tts is None:
        return jsonify({'error': 'TTS backend not configured. Install gTTS or pyttsx3.'}), 500
    try:
        audio_path = tts.speak(text, lang=lang)
        # Return a URL path to download/stream the file
        filename = os.path.basename(audio_path)
        return jsonify({'success': True, 'audio_path': f"/tts/{filename}", 'local_path': audio_path})
    except Exception as e:
        return jsonify({'error': f'TTS failed: {str(e)}'}), 500


@app.route('/vqa', methods=['POST'])
def vqa():
    # Accepts multipart form: 'image' file and 'question' text
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    file = request.files['image']
    question = request.form.get('question') or request.args.get('question') or (request.json or {}).get('question')
    if not file or file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    if not question:
        return jsonify({'error': 'No question provided'}), 400
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400

    # save file
    filename = secure_filename(file.filename)
    save_name = f"vqa_{uuid.uuid4().hex}_{filename}"
    save_path = os.path.join(app.config['UPLOAD_FOLDER'], save_name)
    file.save(save_path)

    # lazy-load VQA service on first use
    vqa_service = get_vqa_service()
    if vqa_service is None:
        return jsonify({'error': 'VQA service not available. Install transformers and torch or configure adapter callback.'}), 500

    try:
        resp = vqa_service.answer(save_path, question)
        return jsonify({'success': True, 'answer': resp})
    except Exception as e:
        return jsonify({'error': f'VQA failed: {str(e)}'}), 500

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'xray_image' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['xray_image']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Please upload an image file.'}), 400
    
    try:
        detected_conditions = analyzer.analyze_xray(file)
        
        english_caption = caption_gen.generate_caption(detected_conditions)
        
        detailed_report = caption_gen.generate_detailed_report(detected_conditions)
        
        translations = translator.translate_to_all_languages(english_caption)
        
        result = {
            'success': True,
            'english_caption': english_caption,
            'translations': translations,
            'detailed_report': detailed_report,
            'detected_conditions': detected_conditions
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_message = data.get('message', '')
    
    if not user_message:
        return jsonify({'error': 'No message provided'}), 400
    
    try:
        conversation_history.append({'role': 'user', 'content': user_message})
        
        response = chatbot.chat(user_message, conversation_history)
        
        if not response['error']:
            conversation_history.append({'role': 'assistant', 'content': response['response']})
        
        if len(conversation_history) > 20:
            conversation_history.pop(0)
            conversation_history.pop(0)
        
        return jsonify({
            'success': True,
            'response': response['response'],
            'chatbot_configured': chatbot.is_configured()
        })
    
    except Exception as e:
        return jsonify({'error': f'Chat failed: {str(e)}'}), 500

@app.route('/clear_chat', methods=['POST'])
def clear_chat():
    global conversation_history
    conversation_history = []
    return jsonify({'success': True, 'message': 'Chat history cleared'})

@app.route('/dental_tips', methods=['GET'])
def dental_tips():
    tips = chatbot.get_dental_tips()
    return jsonify({'tips': tips})

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'chatbot_configured': chatbot.is_configured()
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
