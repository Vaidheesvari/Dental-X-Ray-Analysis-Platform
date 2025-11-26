"""
Machine Learning-based Dental X-Ray Analyzer
Uses trained Random Forest model for classification
"""

import numpy as np
from PIL import Image
import cv2
import os
import joblib
from dental_analyzer import DentalXrayAnalyzer

class DentalXrayAnalyzerML(DentalXrayAnalyzer):
    def __init__(self, model_path='models/dental_model.pkl', encoder_path='models/label_encoder.pkl'):
        super().__init__()
        self.model_path = model_path
        self.encoder_path = encoder_path
        self.model = None
        self.label_encoder = None
        self.use_ml_model = False
        
        # Try to load the trained model
        if os.path.exists(model_path) and os.path.exists(encoder_path):
            try:
                self.model = joblib.load(model_path)
                self.label_encoder = joblib.load(encoder_path)
                self.use_ml_model = True
                print(f"✓ ML Model loaded from {model_path}")
            except Exception as e:
                print(f"Warning: Could not load ML model: {str(e)}")
                print("Falling back to rule-based classification")
                self.use_ml_model = False
        else:
            print(f"Warning: Model files not found at {model_path}")
            print("Falling back to rule-based classification")
            self.use_ml_model = False
    
    def _flatten_features(self, features):
        """Convert feature dictionary to a flat numpy array (same as training)"""
        feature_list = []
        
        # Low-dimensional features
        feature_list.append(features['edge_density'])
        feature_list.append(features['gradient_mean'])
        feature_list.append(features['gradient_std'])
        feature_list.append(features['laplacian_var'])
        feature_list.append(features['contour_count'])
        feature_list.append(features['avg_contour_area'])
        feature_list.append(features['max_contour_area'])
        feature_list.append(features['dark_region_density'])
        feature_list.append(features['dark_region_mean_intensity'])
        feature_list.append(features['texture_contrast'])
        feature_list.append(features['texture_homogeneity'])
        feature_list.append(features['texture_energy'])
        feature_list.append(features['texture_correlation'])
        feature_list.append(features['morph_difference'])
        feature_list.append(features['symmetry_score'])
        
        # Quadrant statistics (4 quadrants × 3 stats = 12 features)
        for quadrant in ['upper_left', 'upper_right', 'lower_left', 'lower_right']:
            quadrant_stats = features['quadrant_stats'][quadrant]
            feature_list.append(quadrant_stats['mean'])
            feature_list.append(quadrant_stats['std'])
            feature_list.append(quadrant_stats['dark_ratio'])
        
        return np.array(feature_list).reshape(1, -1)
    
    def _predict_with_ml_model(self, features):
        if not self.use_ml_model:
            return None
        
        try:
            # Flatten features
            feature_vector = self._flatten_features(features)
            
            # Predict
            predicted_class_encoded = self.model.predict(feature_vector)[0]
            predicted_proba = self.model.predict_proba(feature_vector)[0]
            
            # Decode class
            predicted_class = self.label_encoder.inverse_transform([predicted_class_encoded])[0]
            
            # Get confidence (probability of predicted class)
            confidence = predicted_proba[predicted_class_encoded]
            
            # Get all class probabilities
            class_probas = {
                self.label_encoder.classes_[i]: proba 
                for i, proba in enumerate(predicted_proba)
            }
            
            return {
                'condition': predicted_class,
                'confidence': float(confidence),
                'all_probabilities': class_probas
            }
        except Exception as e:
            print(f"Error in ML prediction: {str(e)}")
            return None

    def _calibrate_confidence(self, p):
        c = max(0.15, min(0.98, (p ** 0.85)))
        return float(c)

    def _severity_for_condition(self, condition, confidence, features):
        if condition == 'normal':
            return 'none'
        if condition == 'cavity':
            d = features.get('dark_region_density', 0.0)
            return 'severe' if d > 0.25 or confidence > 0.85 else 'moderate' if d > 0.18 or confidence > 0.65 else 'mild'
        if condition == 'bone_loss':
            low = sum(1 for q in features['quadrant_stats'].values() if q['mean'] < 90)
            return 'severe' if low >= 3 or confidence > 0.85 else 'moderate' if low >= 2 or confidence > 0.65 else 'mild'
        if condition == 'misalignment':
            s = features.get('symmetry_score', 1.0)
            return 'severe' if s < 0.6 else 'moderate' if s < 0.75 else 'mild'
        if condition == 'impacted_tooth':
            g = features.get('gradient_mean', 0.0)
            return 'severe' if g > 45 or confidence > 0.85 else 'moderate' if g > 35 or confidence > 0.65 else 'mild'
        if condition in ['bone_structure_anomaly', 'cyst']:
            return 'moderate' if confidence > 0.6 else 'mild'
        if condition == 'restoration':
            d = features.get('bright_region_density', 0.0)
            return 'severe' if d > 0.02 or confidence > 0.9 else 'moderate' if d > 0.008 or confidence > 0.7 else 'mild'
        return 'moderate'

    def _location_for_condition(self, condition, features):
        if condition == 'bone_loss':
            return self._localize_bone_loss(features['quadrant_stats'])
        metric_map = {
            'cavity': 'dark_ratio',
            'cyst': 'dark_ratio',
            'restoration': 'bright_ratio',
            'impacted_tooth': 'std',
            'misalignment': 'std',
            'bone_structure_anomaly': 'std'
        }
        metric = metric_map.get(condition, 'dark_ratio')
        return self._localize_condition(features['quadrant_stats'], metric)
    
    def analyze_xray(self, image_file):
        """Analyze X-ray image using ML model if available, else rule-based"""
        img = Image.open(image_file).convert('RGB')
        img_array = np.array(img)
        
        preprocessed = self._preprocess_image(img_array)
        features = self._extract_cnn_like_features(preprocessed)
        
        # Try ML model first
        if self.use_ml_model:
            ml_result = self._predict_with_ml_model(features)
            if ml_result:
                probas = ml_result['all_probabilities']
                items = sorted(probas.items(), key=lambda x: x[1], reverse=True)
                detected_conditions = []
                for label, p in items:
                    if label == 'normal' and any(v > 0.4 for k, v in probas.items() if k != 'normal'):
                        continue
                    if label != 'normal' and p < 0.4:
                        continue
                    conf = self._calibrate_confidence(p)
                    loc = 'overall' if label == 'normal' else self._location_for_condition(label, features)
                    sev = self._severity_for_condition(label, conf, features)
                    detected_conditions.append({
                        'condition': label,
                        'location': loc,
                        'severity': sev,
                        'confidence': conf,
                        'ml_model': True
                    })
                if len(detected_conditions) == 0:
                    top_label, top_p = items[0]
                    conf = self._calibrate_confidence(top_p)
                    loc = 'overall' if top_label == 'normal' else self._location_for_condition(top_label, features)
                    sev = self._severity_for_condition(top_label, conf, features)
                    detected_conditions.append({
                        'condition': top_label,
                        'location': loc,
                        'severity': sev,
                        'confidence': conf,
                        'ml_model': True
                    })
                return detected_conditions
        
        # Fall back to rule-based classification
        detected_conditions = super()._classify_conditions(features, preprocessed)
        for cond in detected_conditions:
            cond['ml_model'] = False
        
        return detected_conditions

