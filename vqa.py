import os
import logging
from typing import Optional
from PIL import Image

# configure logger for this module
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s %(levelname)s [vqa] %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.DEBUG)

class VQAService:
    """Visual Question Answering service wrapper.

    Attempts to load a local BLIP VQA model (transformers + torch).
    If unavailable, provides a graceful fallback that uses the existing
    `DentAdapt` adapter callback (if configured) or a simple heuristic reply.
    """

    def __init__(self, adapter_callback: Optional[callable] = None):
        self.adapter_callback = adapter_callback
        self.model = None
        self.processor = None
        self.device = 'cpu'
        # allow model name override to select a smaller model if desired
        model_name = os.getenv('VQA_MODEL_NAME', 'Salesforce/blip-vqa-base')
        try:
            import torch
            from transformers import BlipForQuestionAnswering, BlipProcessor
            self.torch = torch
            # attempt to load a lighter model by default
            logger.debug(f'Attempting to load VQA model: {model_name}')
            self.processor = BlipProcessor.from_pretrained(model_name)
            self.model = BlipForQuestionAnswering.from_pretrained(model_name)
            if torch.cuda.is_available():
                self.device = 'cuda'
            self.model.to(self.device)
        except Exception as e:
            # model not available — will use fallback
            self.model = None
            self.processor = None
            self.torch = None
            # keep error around for debugging and log it
            self._load_error = str(e)
            logger.debug(f'Could not load VQA model {model_name}: {self._load_error}')

    def is_configured(self) -> bool:
        return self.model is not None and self.processor is not None

    def answer(self, image_file, question: str) -> dict:
        """Return a dict with keys: `answer`, `confidence`, `source`."""
        # If local model is available, run it
        if self.is_configured():
            try:
                img = Image.open(image_file).convert('RGB')
                inputs = self.processor(images=img, text=question, return_tensors='pt')
                # move tensors to device
                if self.torch is not None:
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}

                with (self.torch.no_grad() if self.torch is not None else DummyContext()):
                    # Use generate to produce token ids, then decode with tokenizer
                    generated_ids = self.model.generate(**inputs, max_length=32)
                    # processor may not expose decode directly; use tokenizer
                    tokenizer = getattr(self.processor, 'tokenizer', None)
                    if tokenizer is not None:
                        answer = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
                    else:
                        # fallback to processor.decode if available
                        decode_fn = getattr(self.processor, 'decode', None)
                        if callable(decode_fn):
                            answer = decode_fn(generated_ids[0], skip_special_tokens=True)
                        else:
                            answer = generated_ids[0].tolist()

                logger.debug(f'BLIP VQA produced answer: {answer}')
                return {'answer': answer, 'confidence': None, 'source': 'blip-local'}
            except Exception as e:
                # fall through to adapter or heuristic
                # keep the exception message for debugging
                last_exc = e
                # attach last exception info for debugging
                self._last_exc = str(e)
                logger.exception('Error during BLIP VQA inference')


class DummyContext:
    def __enter__(self):
        return None
    def __exit__(self, exc_type, exc, tb):
        return False

        # try adapter callback (e.g., remote VLM) if provided
        if callable(self.adapter_callback):
            try:
                resp = self.adapter_callback(image_file, question)
                return {'answer': resp, 'confidence': None, 'source': 'adapter-callback'}
            except Exception:
                # adapter failed — continue to heuristic fallback
                pass

        # fallback heuristic: run the image analyzer to extract findings and answer based on them
        try:
            from dental_analyzer import DentalXrayAnalyzer
            analyzer = DentalXrayAnalyzer()
            findings = analyzer.analyze_xray(image_file)
            # simplistic mapping: if question mentions 'cavity', return whether cavities detected
            q = question.lower()
            if 'cavity' in q or 'cavities' in q:
                has = any(f['condition'] == 'cavity' for f in findings)
                return {'answer': 'Yes, cavities detected' if has else 'No obvious cavities detected', 'confidence': None, 'source': 'heuristic'}
            # generic reply listing top findings
            top = findings[0]
            ans = f"Detected {top['condition']} in {top.get('location','unknown')} (severity: {top.get('severity','unknown')})"
            return {'answer': ans, 'confidence': top.get('confidence'), 'source': 'heuristic'}
        except Exception as e:
            # return a helpful message including exception for easier debugging
            return {'answer': 'Unable to answer the question with current configuration.', 'confidence': None, 'source': 'none', 'error': str(e)}
