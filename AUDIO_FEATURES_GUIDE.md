# 🎵 Audio Features Guide - Dental X-Ray Analyzer

## Overview
The Dental X-Ray Analyzer includes comprehensive **Text-to-Speech (TTS)** capabilities for both X-ray analysis results and chatbot responses, supporting **7 languages** with automatic audio generation and playback.

---

## 🔧 Setup Requirements

### 1. Install TTS Backend (Choose One)

**Option A: Google Text-to-Speech (gTTS) - Recommended**
```bash
pip install gTTS>=2.3.0
```
- ✅ Better voice quality
- ✅ Supports 7+ languages natively
- ✅ Online service (requires internet)
- ✅ No additional dependencies

**Option B: pyttsx3 - Offline Alternative**
```bash
pip install pyttsx3>=2.90
```
- ✅ Works completely offline
- ✅ No internet required
- ✅ Fast local processing
- ⚠️ Limited language support (system-dependent)

**Recommended: Install Both**
```bash
pip install gTTS>=2.3.0 pyttsx3>=2.90
```
The system will automatically prefer gTTS, and fall back to pyttsx3 if needed.

### 2. Verify Installation
```bash
python -c "import gtts; print('gTTS installed')"
python -c "import pyttsx3; print('pyttsx3 installed')"
```

---

## 🎙️ Audio Features in Action

### Feature 1: X-Ray Analysis Audio
**When it activates:** After analyzing an X-ray image
**What it does:** Generates audio for the English caption and any selected language translation
**How to use:**
1. Upload a dental X-ray image
2. Click **"Analyze X-Ray"**
3. In the results, select a language tab (English, Tamil, Telugu, etc.)
4. Click the **"Speak"** button
5. Audio player appears automatically
6. Click play ▶️ or use browser controls

**Supported Languages:**
- English (en)
- Tamil (ta)
- Telugu (te)
- Hindi (hi)
- Malayalam (ml)
- Bengali (bn)
- Urdu (ur)

### Feature 2: Chatbot Audio Response
**When it activates:** After chatbot responds to your health question
**What it does:** Generates audio for the last chatbot message
**How to use:**
1. Type a dental health question in the chat box
2. Click **"Send"** or press Enter
3. Wait for the chatbot response
4. Click the **"Speak"** button (yellow/orange button)
5. Audio player appears below the chat
6. Click play ▶️ to hear the response

**Supported Languages:**
- Primarily English (auto-detected from response)
- Can be extended to other languages by modifying the backend

### Feature 3: VQA (Visual Question Answering) Audio
**When it activates:** After asking a question about an X-ray image
**What it does:** Generates audio for the VQA answer
**How to use:**
1. Upload an X-ray image and analyze it
2. Click **"VQA"** button in the chat section
3. Type your question (e.g., "Is there a cavity on the upper left?")
4. Click **"Ask VQA"**
5. Click the **"Speak"** button next to the answer
6. Audio player appears
7. Click play ▶️ to hear the answer

---

## 🔌 Backend API Endpoints

### `/speak` - Generate Speech
**Method:** POST  
**Content-Type:** application/json

**Request Body:**
```json
{
  "text": "The X-ray reveals significant bone loss in the lower jaw.",
  "lang": "en"
}
```

**Response (Success):**
```json
{
  "success": true,
  "audio_path": "/tts/tts_a1b2c3d4e5f6.mp3",
  "local_path": "uploads/tts/tts_a1b2c3d4e5f6.mp3"
}
```

**Response (Error - No Backend):**
```json
{
  "error": "TTS backend not configured. Install gTTS or pyttsx3."
}
```

**Available Language Codes:**
| Language | Code |
|----------|------|
| English | en |
| Tamil | ta |
| Telugu | te |
| Hindi | hi |
| Malayalam | ml |
| Bengali | bn |
| Urdu | ur |
| Spanish | es |
| French | fr |
| German | de |
| And 100+ more (gTTS support) |

### `/tts/<filename>` - Stream Audio
**Method:** GET  
**Purpose:** Download or stream generated audio file

**Example:**
```
GET /tts/tts_a1b2c3d4e5f6.mp3
```

---

## 📊 Audio Processing Flow

```
User Input (Text)
       ↓
[Language Selection]
       ↓
[TTS Engine Selection]
  ├─ gTTS (if available)
  └─ pyttsx3 (fallback)
       ↓
[Audio Generation]
  ├─ MP3 format (gTTS)
  └─ MP3 or WAV (pyttsx3)
       ↓
[Storage]
  └─ uploads/tts/<uuid>.mp3
       ↓
[Browser Audio Player]
  └─ HTML5 <audio> element
```

---

## ⚙️ Configuration & Customization

### Environment Variables
Set these in your `.env` file or system environment:

```bash
# None required - auto-detection
# If both gTTS and pyttsx3 are installed, gTTS takes priority
```

### Programmatic Usage (Python)

**Generate audio in your code:**
```python
from tts import MultilingualTTS

tts = MultilingualTTS(output_dir='uploads/tts')

# English
audio_path = tts.speak("The X-ray shows a cavity", lang='en')
print(f"Audio saved to: {audio_path}")

# Tamil
audio_path = tts.speak("உங்கள் பல் வலி இருக்கிறதா", lang='ta')

# Hindi
audio_path = tts.speak("दंत स्वास्थ्य बहुत महत्वपूर्ण है", lang='hi')
```

### Troubleshooting

**Problem: "TTS backend not configured"**
- Solution: Install gTTS or pyttsx3
  ```bash
  pip install gTTS pyttsx3
  ```

**Problem: Audio plays but voice quality is poor**
- Solution: Try gTTS (better quality)
  ```bash
  pip install gTTS
  ```

**Problem: "No internet" and gTTS not working**
- Solution: Install pyttsx3 for offline use
  ```bash
  pip install pyttsx3
  ```

**Problem: Audio file size too large**
- Solution: gTTS compresses better; use gTTS instead of pyttsx3

**Problem: Language not supported**
- Solution: Check language code in gTTS documentation
  - gTTS supports 100+ languages
  - pyttsx3 limited to system languages

---

## 📈 Performance Metrics

| Component | Speed | Notes |
|-----------|-------|-------|
| TTS Generation (gTTS) | 500-1200ms | Online, better quality |
| TTS Generation (pyttsx3) | 200-800ms | Offline, faster but voice quality varies |
| Audio File Size | 50-150 KB | Per sentence, gTTS MP3 |
| Storage Location | `uploads/tts/` | Auto-cleaned on restart |

---

## 🌍 Multilingual Audio Examples

### English
```
Input: "The X-ray reveals significant bone loss"
Audio: Generated with natural English TTS
```

### Tamil (தமிழ்)
```
Input: "எக்ஸ்-கதிர் பெரிய எலும்பு இழப்பைக் காட்டுகிறது"
Audio: Generated with Tamil voice
```

### Hindi (हिंदी)
```
Input: "एक्स-रे महत्वपूर्ण हड्डी के नुकसान को दिखाता है"
Audio: Generated with Hindi voice
```

---

## 🔐 Privacy & Security

- **Local Caching:** Audio files stored in `uploads/tts/` directory
- **Temporary Storage:** Files remain until server restart or manual cleanup
- **gTTS Privacy:** Sent to Google for processing (standard terms apply)
- **pyttsx3 Privacy:** Completely offline, no data sent

---

## 🚀 Advanced Usage

### Custom Audio Implementation
To add audio to other parts of the app:

```javascript
// In your JavaScript
const resp = await fetch('/speak', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ 
        text: 'Your text here', 
        lang: 'en' 
    })
});
const data = await resp.json();

if (data.success) {
    const audio = new Audio(data.audio_path);
    audio.play();
}
```

### Batch Audio Generation
```python
# Generate audio for multiple conditions
from tts import MultilingualTTS

tts = MultilingualTTS()
conditions = [
    ("Cavity detected", "en"),
    ("கெட்ட பல் கண்டறியப்பட்டது", "ta"),
    ("दांत की क्षति", "hi")
]

for text, lang in conditions:
    path = tts.speak(text, lang=lang)
    print(f"Generated: {path}")
```

---

## 📚 Resources

- **gTTS Documentation:** https://gtts.readthedocs.io/
- **pyttsx3 Documentation:** https://pyttsx3.readthedocs.io/
- **Supported Languages:** Check gTTS or pyttsx3 docs for full list

---

**Last Updated:** November 2025  
**Status:** ✅ Fully Functional
