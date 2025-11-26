import numpy as np
from PIL import Image
from dental_analyzer import DentalXrayAnalyzer

class DentAdaptTransformer(DentalXrayAnalyzer):
    def __init__(self):
        super().__init__()
        self.model = None
        self.processor = None
        try:
            import torch
            from transformers import CLIPProcessor, CLIPModel
            self.torch = torch
            self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            self.device = "cpu"
            self.model.to(self.device)
        except Exception:
            self.model = None
            self.processor = None
            self.torch = None

    def _labels(self):
        return [
            "normal panoramic dental x-ray",
            "dental x-ray shows cavity",
            "dental x-ray shows bone loss",
            "dental x-ray shows impacted tooth",
            "dental x-ray shows misalignment",
            "dental x-ray shows bone structure anomaly",
            "dental x-ray shows restoration or metallic filling",
            "dental x-ray shows cyst"
        ]

    def _map_condition(self, idx):
        mapping = {
            0: "normal",
            1: "cavity",
            2: "bone_loss",
            3: "impacted_tooth",
            4: "misalignment",
            5: "bone_structure_anomaly",
            6: "restoration",
            7: "cyst"
        }
        return mapping.get(idx, "normal")

    def _calibrate(self, p):
        return float(max(0.1, min(0.98, p ** 0.9)))

    def analyze_xray(self, image_file):
        img = Image.open(image_file).convert("RGB")
        img_array = np.array(img)
        preprocessed = self._preprocess_image(img_array)
        features = self._extract_cnn_like_features(preprocessed)
        if self.model is None or self.processor is None:
            detected = super()._classify_conditions(features, preprocessed)
            return detected
        texts = self._labels()
        inputs = self.processor(text=texts, images=img, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with self.torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits_per_image[0]
            probs = self.torch.softmax(logits, dim=0).cpu().numpy()
        ranked = sorted([(i, float(probs[i])) for i in range(len(texts))], key=lambda x: x[1], reverse=True)
        findings = []
        for idx, p in ranked:
            cond = self._map_condition(idx)
            if cond == "normal" and any(pp > 0.35 for ii, pp in ranked if self._map_condition(ii) != "normal"):
                continue
            if cond != "normal" and p < 0.35:
                continue
            conf = self._calibrate(p)
            if cond == "bone_loss":
                loc = self._localize_bone_loss(features['quadrant_stats'])
            else:
                metric = 'bright_ratio' if cond == 'restoration' else 'dark_ratio'
                loc = self._localize_condition(features['quadrant_stats'], metric)
            if cond == "normal":
                loc = "overall"
            sev = 'none' if cond == 'normal' else ('severe' if conf > 0.85 else 'moderate' if conf > 0.6 else 'mild')
            findings.append({
                'condition': cond,
                'location': loc,
                'severity': sev,
                'confidence': conf,
                'ml_model': True
            })
        if len(findings) == 0:
            idx, p = ranked[0]
            cond = self._map_condition(idx)
            conf = self._calibrate(p)
            loc = "overall" if cond == "normal" else self._localize_condition(features['quadrant_stats'], 'dark_ratio')
            sev = 'none' if cond == 'normal' else ('severe' if conf > 0.85 else 'moderate' if conf > 0.6 else 'mild')
            findings.append({
                'condition': cond,
                'location': loc,
                'severity': sev,
                'confidence': conf,
                'ml_model': True
            })
        return findings