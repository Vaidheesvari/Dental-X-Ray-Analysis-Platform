import os
import json
import numpy as np
from PIL import Image
from dental_analyzer import DentalXrayAnalyzer

class DentAdapt(DentalXrayAnalyzer):
    def __init__(self, inference_fn=None):
        super().__init__()
        self.inference_fn = inference_fn

    def _build_prompt(self):
        labels = [
            "normal",
            "cavity",
            "impacted_tooth",
            "misalignment",
            "bone_loss",
            "bone_structure_anomaly",
            "cyst",
        ]
        locations = [
            "upper left region",
            "upper right region",
            "lower left region",
            "lower right region",
            "overall",
        ]
        severities = ["none", "mild", "moderate", "severe"]
        spec = {
            "schema": {
                "type": "object",
                "properties": {
                    "findings": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "condition": {"type": "string", "enum": labels},
                                "location": {"type": "string", "enum": locations},
                                "severity": {"type": "string", "enum": severities},
                                "confidence": {"type": "number"},
                            },
                            "required": [
                                "condition",
                                "location",
                                "severity",
                                "confidence",
                            ],
                        },
                    }
                },
                "required": ["findings"],
            }
        }
        prompt = (
            "You are a dental radiology assistant. "
            "Analyze the provided dental X-ray and return JSON strictly matching this schema: "
            + json.dumps(spec)
            + "."
            " Prefer 'overall' for normal findings. Use confidence in [0,1]."
        )
        return prompt

    def _parse_response(self, text):
        try:
            data = json.loads(text)
            findings = data.get("findings", [])
            result = []
            for f in findings:
                result.append(
                    {
                        "condition": f.get("condition", "normal"),
                        "location": f.get("location", "overall"),
                        "severity": f.get("severity", "moderate"),
                        "confidence": float(f.get("confidence", 0.5)),
                    }
                )
            return result
        except Exception:
            return None

    def _ask_model(self, img, prompt):
        if callable(self.inference_fn):
            return self.inference_fn(img, prompt)
        return None

    def analyze_xray(self, image_file):
        img = Image.open(image_file).convert("RGB")
        img_array = np.array(img)
        preprocessed = self._preprocess_image(img_array)
        features = self._extract_cnn_like_features(preprocessed)

        prompt = self._build_prompt()
        response_text = self._ask_model(img, prompt)
        if isinstance(response_text, str):
            parsed = self._parse_response(response_text)
            if isinstance(parsed, list) and len(parsed) > 0:
                return parsed

        detected_conditions = super()._classify_conditions(features, preprocessed)
        return detected_conditions