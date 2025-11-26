import os
import uuid
from pathlib import Path
from typing import Optional

try:
    from gtts import gTTS
    _HAS_GTTS = True
except Exception:
    _HAS_GTTS = False

try:
    import pyttsx3
    _HAS_PYTTSX3 = True
except Exception:
    _HAS_PYTTSX3 = False


class MultilingualTTS:
    """Multilingual text-to-speech helper.

    - Prefers `gTTS` (better voice quality, many languages) when available.
    - Falls back to `pyttsx3` (offline) if gTTS not available.
    - Saves audio files into the provided `output_dir` and returns filename.
    """

    def __init__(self, output_dir: str = 'uploads/tts'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if not _HAS_GTTS and not _HAS_PYTTSX3:
            raise RuntimeError('No TTS backend available. Install `gTTS` or `pyttsx3`.')

        self.use_gtts = _HAS_GTTS
        if not self.use_gtts and _HAS_PYTTSX3:
            # initialize engine lazily
            self.engine = None
        else:
            self.engine = None

    def _filename(self):
        return f"tts_{uuid.uuid4().hex}.mp3"

    def speak(self, text: str, lang: str = 'en') -> str:
        """Generate speech for `text` in language `lang`.

        Returns the relative path to the audio file (mp3).
        """
        filename = self._filename()
        out_path = self.output_dir / filename

        if self.use_gtts:
            try:
                t = gTTS(text=text, lang=lang)
                t.save(str(out_path))
                return str(out_path)
            except Exception as e:
                # fallback to pyttsx3 if available
                if _HAS_PYTTSX3:
                    self.use_gtts = False
                else:
                    raise

        # pyttsx3 fallback (may only support system voices/languages)
        if _HAS_PYTTSX3:
            if self.engine is None:
                self.engine = pyttsx3.init()
            # pyttsx3 cannot save to mp3 directly on some platforms; try saving to wav
            wav_path = str(out_path.with_suffix('.wav'))
            self.engine.save_to_file(text, wav_path)
            self.engine.runAndWait()
            # try to convert wav -> mp3 if ffmpeg available, otherwise return wav path
            mp3_path = str(out_path)
            try:
                import subprocess
                subprocess.run(['ffmpeg', '-y', '-i', wav_path, mp3_path], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                # remove wav
                try:
                    os.remove(wav_path)
                except Exception:
                    pass
                return mp3_path
            except Exception:
                return wav_path

        raise RuntimeError('No available TTS backend')
