

import numpy as np
import os
import pickle
import json
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import joblib
from tqdm import tqdm
import cv2
from PIL import Image
from dental_analyzer import DentalXrayAnalyzer

class DentalModelTrainer:
    def __init__(self, images_dir='images', model_save_path='models'):
        self.images_dir = images_dir
        self.model_save_path = model_save_path
        self.analyzer = DentalXrayAnalyzer()
        self.label_encoder = LabelEncoder()
        
        # Create models directory if it doesn't exist
        os.makedirs(model_save_path, exist_ok=True)
        
        # Condition classes
        self.condition_classes = [
            'normal', 'cavity', 'impacted_tooth', 'misalignment', 
            'bone_loss', 'bone_structure_anomaly', 'cyst', 'restoration'
        ]
        
    def extract_features_from_image(self, image_path, return_preprocessed=False):
        """Extract features from a single image"""
        try:
            # Use the analyzer's preprocessing and feature extraction
            img = Image.open(image_path).convert('RGB')
            img_array = np.array(img)
            
            preprocessed = self.analyzer._preprocess_image(img_array)
            features = self.analyzer._extract_cnn_like_features(preprocessed)
            
            # Convert features to a flat vector for ML model
            feature_vector = self._flatten_features(features)
                
            if return_preprocessed:
                return feature_vector, features, preprocessed
            return feature_vector, features
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
            if return_preprocessed:
                return None, None, None
            return None, None
    
    def _flatten_features(self, features):
        """Convert feature dictionary to a flat numpy array"""
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
        
        return np.array(feature_list)
    
    def generate_labels_from_rules(self, features, preprocessed=None):
        """Generate labels using existing rule-based classifier"""
        # Create a dummy preprocessed image if not provided
        if preprocessed is None:
            # Create a minimal dummy array for the classifier
            preprocessed = np.zeros((512, 512), dtype=np.uint8)
        
        detected = self.analyzer._classify_conditions(features, preprocessed)
        
        # Get the primary condition (first detected condition)
        if detected and len(detected) > 0:
            primary_condition = detected[0]['condition']
            return primary_condition
        return 'normal'
    
    def _augment_image_arrays(self, img_array):
        aug_list = []
        aug_list.append(img_array)
        aug_list.append(cv2.flip(img_array, 1))
        h, w = img_array.shape[:2]
        for angle in [-5, 5]:
            M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
            rotated = cv2.warpAffine(img_array, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            aug_list.append(rotated)
        for alpha in [0.9, 1.1]:
            adjusted = cv2.convertScaleAbs(img_array, alpha=alpha, beta=5)
            aug_list.append(adjusted)
        return aug_list

    def load_and_preprocess_dataset(self):
        """Load all images and extract features with augmentation"""
        print("Loading dataset...")
        image_files = list(Path(self.images_dir).glob("*.jpg"))
        print(f"Found {len(image_files)} images")
        
        X = []  # Features
        y = []  # Labels
        image_paths = []  # Keep track of image paths
        
        print("Extracting features from images with augmentation...")
        for image_path in tqdm(image_files, desc="Processing images"):
            try:
                img = Image.open(image_path).convert('RGB')
                base_array = np.array(img)
                aug_arrays = self._augment_image_arrays(base_array)
                for arr in aug_arrays:
                    preprocessed = self.analyzer._preprocess_image(arr)
                    features_dict = self.analyzer._extract_cnn_like_features(preprocessed)
                    feature_vector = self._flatten_features(features_dict)
                    label = self.generate_labels_from_rules(features_dict, preprocessed)
                    X.append(feature_vector)
                    y.append(label)
                    image_paths.append(str(image_path))
            except Exception as e:
                print(f"Error processing {image_path}: {str(e)}")
        
        X = np.array(X)
        y = np.array(y)
        
        print(f"\nDataset loaded:")
        print(f"  Total samples: {len(X)}")
        print(f"  Feature dimensions: {X.shape[1]}")
        print(f"  Classes: {np.unique(y)}")
        print(f"  Class distribution:")
        for cls in np.unique(y):
            count = np.sum(y == cls)
            print(f"    {cls}: {count} ({count/len(y)*100:.1f}%)")
        
        return X, y, image_paths
    
    def train_model(self, X, y, test_size=0.2, random_state=42):
        """Train Random Forest classifier"""
        print("\n" + "="*50)
        print("Training Model")
        print("="*50)
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Split dataset
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=test_size, random_state=random_state, stratify=y_encoded
        )
        
        print(f"Training set: {X_train.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")
        
        # Train Random Forest
        print("\nTraining Random Forest classifier...")
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=-1,
            class_weight='balanced'  # Handle class imbalance
        )
        
        model.fit(X_train, y_train)
        
        # Evaluate
        print("\nEvaluating model...")
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        train_accuracy = accuracy_score(y_train, y_train_pred)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        
        print(f"\nTraining Accuracy: {train_accuracy:.4f}")
        print(f"Test Accuracy: {test_accuracy:.4f}")
        
        # Classification report
        print("\n" + "="*50)
        print("Classification Report (Test Set)")
        print("="*50)
        class_names = self.label_encoder.classes_
        print(classification_report(y_test, y_test_pred, target_names=class_names))
        
        # Confusion matrix
        print("\nConfusion Matrix (Test Set)")
        print("="*50)
        cm = confusion_matrix(y_test, y_test_pred)
        print(cm)
        print(f"\nClass labels: {class_names}")
        
        # Feature importance
        print("\n" + "="*50)
        print("Top 10 Most Important Features")
        print("="*50)
        feature_names = self._get_feature_names()
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:10]
        for i in indices:
            print(f"  {feature_names[i]}: {importances[i]:.4f}")
        
        return model, X_test, y_test, y_test_pred
    
    def _get_feature_names(self):
        """Get feature names for interpretability"""
        feature_names = [
            'edge_density', 'gradient_mean', 'gradient_std', 'laplacian_var',
            'contour_count', 'avg_contour_area', 'max_contour_area',
            'dark_region_density', 'dark_region_mean_intensity',
            'texture_contrast', 'texture_homogeneity', 'texture_energy',
            'texture_correlation', 'morph_difference', 'symmetry_score'
        ]
        
        # Quadrant features
        for quadrant in ['upper_left', 'upper_right', 'lower_left', 'lower_right']:
            feature_names.append(f'{quadrant}_mean')
            feature_names.append(f'{quadrant}_std')
            feature_names.append(f'{quadrant}_dark_ratio')
        
        return feature_names
    
    def save_model(self, model, filename='dental_model.pkl'):
        """Save trained model and label encoder"""
        model_path = os.path.join(self.model_save_path, filename)
        encoder_path = os.path.join(self.model_save_path, 'label_encoder.pkl')
        
        # Save model
        joblib.dump(model, model_path)
        print(f"\nModel saved to: {model_path}")
        
        # Save label encoder
        joblib.dump(self.label_encoder, encoder_path)
        print(f"Label encoder saved to: {encoder_path}")
        
        # Save model metadata
        metadata = {
            'feature_count': model.n_features_in_,
            'n_classes': len(self.label_encoder.classes_),
            'classes': self.label_encoder.classes_.tolist(),
            'n_estimators': model.n_estimators,
            'max_depth': model.max_depth
        }
        
        metadata_path = os.path.join(self.model_save_path, 'model_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"Model metadata saved to: {metadata_path}")
        
        return model_path, encoder_path, metadata_path
    
    def train(self):
        """Main training pipeline"""
        print("="*50)
        print("Dental X-Ray Analyzer - Model Training")
        print("="*50)
        
        # Load dataset
        X, y, image_paths = self.load_and_preprocess_dataset()
        
        if len(X) == 0:
            print("Error: No features extracted. Please check your images directory.")
            return
        
        # Train model
        model, X_test, y_test, y_test_pred = self.train_model(X, y)
        
        # Save model
        self.save_model(model)
        
        print("\n" + "="*50)
        print("Training Complete!")
        print("="*50)
        print(f"\nModel files saved in: {self.model_save_path}/")
        print("  - dental_model.pkl (trained model)")
        print("  - label_encoder.pkl (label encoder)")
        print("  - model_metadata.json (model metadata)")

if __name__ == '__main__':
    trainer = DentalModelTrainer(images_dir='images', model_save_path='models')
    trainer.train()

