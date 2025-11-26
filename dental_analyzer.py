import numpy as np
from PIL import Image
import cv2
from skimage import feature, filters, morphology, exposure
from skimage.feature import graycomatrix, graycoprops
from scipy import ndimage

class DentalXrayAnalyzer:
    def __init__(self):
        self.conditions = []
        self.image_size = (512, 512)
    
    def analyze_xray(self, image_file):
        img = Image.open(image_file).convert('RGB')
        img_array = np.array(img)
        
        preprocessed = self._preprocess_image(img_array)
        
        features = self._extract_cnn_like_features(preprocessed)
        
        detected_conditions = self._classify_conditions(features, preprocessed)
        
        return detected_conditions
    
    def _preprocess_image(self, img_array):
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        resized = cv2.resize(gray, self.image_size)
        
        denoised = cv2.fastNlMeansDenoising(resized, None, 10, 7, 21)
        
        equalized = cv2.equalizeHist(denoised)
        
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(equalized)
        
        return enhanced
    
    def _extract_cnn_like_features(self, preprocessed):
        features = {}
        
        edges_canny = cv2.Canny(preprocessed, 50, 150)
        features['edge_density'] = np.sum(edges_canny > 0) / edges_canny.size
        features['edge_map'] = edges_canny
        
        sobelx = cv2.Sobel(preprocessed, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(preprocessed, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
        features['gradient_magnitude'] = gradient_magnitude
        features['gradient_mean'] = np.mean(gradient_magnitude)
        features['gradient_std'] = np.std(gradient_magnitude)
        
        laplacian = cv2.Laplacian(preprocessed, cv2.CV_64F)
        features['laplacian_var'] = np.var(laplacian)
        
        contours, hierarchy = cv2.findContours(edges_canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        features['contour_count'] = len(contours)
        features['contours'] = contours
        
        if len(contours) > 0:
            areas = [cv2.contourArea(c) for c in contours]
            features['avg_contour_area'] = np.mean(areas) if areas else 0
            features['max_contour_area'] = np.max(areas) if areas else 0
        else:
            features['avg_contour_area'] = 0
            features['max_contour_area'] = 0
        
        distances = np.array([255 - preprocessed[i, j] for i in range(preprocessed.shape[0]) 
                             for j in range(preprocessed.shape[1]) if preprocessed[i, j] < 100])
        if len(distances) > 0:
            features['dark_region_density'] = len(distances) / preprocessed.size
            features['dark_region_mean_intensity'] = 255 - np.mean(distances)
        else:
            features['dark_region_density'] = 0
            features['dark_region_mean_intensity'] = 255

        bright_vals = np.array([preprocessed[i, j] for i in range(preprocessed.shape[0])
                                 for j in range(preprocessed.shape[1]) if preprocessed[i, j] > 200])
        if len(bright_vals) > 0:
            features['bright_region_density'] = len(bright_vals) / preprocessed.size
            features['bright_region_mean_intensity'] = np.mean(bright_vals)
        else:
            features['bright_region_density'] = 0
            features['bright_region_mean_intensity'] = 0
        
        try:
            quantized = (preprocessed // 32).astype(np.uint8)
            glcm = graycomatrix(quantized, distances=[1, 2], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], 
                               levels=8, symmetric=True, normed=True)
            features['texture_contrast'] = np.mean(graycoprops(glcm, 'contrast'))
            features['texture_homogeneity'] = np.mean(graycoprops(glcm, 'homogeneity'))
            features['texture_energy'] = np.mean(graycoprops(glcm, 'energy'))
            features['texture_correlation'] = np.mean(graycoprops(glcm, 'correlation'))
        except:
            features['texture_contrast'] = 0
            features['texture_homogeneity'] = 1
            features['texture_energy'] = 1
            features['texture_correlation'] = 0
        
        kernel = np.ones((5,5), np.uint8)
        _, binary = cv2.threshold(preprocessed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        erosion = cv2.erode(binary, kernel, iterations=1)
        dilation = cv2.dilate(binary, kernel, iterations=1)
        opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        closing = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        features['morph_difference'] = np.sum(np.abs(opening.astype(int) - closing.astype(int))) / binary.size
        
        h, w = preprocessed.shape
        quadrants = {
            'upper_left': preprocessed[:h//2, :w//2],
            'upper_right': preprocessed[:h//2, w//2:],
            'lower_left': preprocessed[h//2:, :w//2],
            'lower_right': preprocessed[h//2:, w//2:]
        }
        
        features['quadrant_stats'] = {}
        for name, quad in quadrants.items():
            features['quadrant_stats'][name] = {
                'mean': np.mean(quad),
                'std': np.std(quad),
                'dark_ratio': np.sum(quad < 80) / quad.size,
                'bright_ratio': np.sum(quad > 200) / quad.size
            }
        
        features['symmetry_score'] = self._calculate_symmetry(preprocessed)
        
        return features
    
    def _calculate_symmetry(self, image):
        h, w = image.shape
        left_half = image[:, :w//2]
        right_half = np.fliplr(image[:, w//2:])
        
        min_width = min(left_half.shape[1], right_half.shape[1])
        left_half = left_half[:, :min_width]
        right_half = right_half[:, :min_width]
        
        diff = np.abs(left_half.astype(float) - right_half.astype(float))
        symmetry = 1.0 - (np.mean(diff) / 255.0)
        
        return symmetry
    
    def _classify_conditions(self, features, preprocessed):
        detected = []
        
        dark_threshold = 0.12
        edge_density_threshold = 0.15
        
        if features['dark_region_density'] > dark_threshold and features['edge_density'] > edge_density_threshold:
            severity = 'severe' if features['dark_region_density'] > 0.25 else \
                      'moderate' if features['dark_region_density'] > 0.18 else 'mild'
            
            location = self._localize_condition(features['quadrant_stats'], 'dark_ratio')
            
            confidence = min(0.92, 0.65 + features['dark_region_density'] * 1.5 + features['edge_density'] * 0.8)
            
            detected.append({
                'condition': 'cavity',
                'location': location,
                'severity': severity,
                'confidence': confidence
            })
        
        if features['symmetry_score'] < 0.75:
            quad_stats = features['quadrant_stats']
            upper_variance = abs(quad_stats['upper_left']['mean'] - quad_stats['upper_right']['mean'])
            lower_variance = abs(quad_stats['lower_left']['mean'] - quad_stats['lower_right']['mean'])
            
            if upper_variance > 25 or lower_variance > 25:
                detected.append({
                    'condition': 'misalignment',
                    'location': 'upper and lower teeth',
                    'severity': 'moderate' if max(upper_variance, lower_variance) > 40 else 'mild',
                    'confidence': min(0.88, 0.60 + (1.0 - features['symmetry_score']) * 0.8)
                })
        
        if features['gradient_mean'] > 35 and features['laplacian_var'] > 800:
            if features['contour_count'] > 50 and features['max_contour_area'] > 500:
                location = self._localize_condition(features['quadrant_stats'], 'std')
                
                detected.append({
                    'condition': 'impacted_tooth',
                    'location': location,
                    'severity': 'moderate',
                    'confidence': min(0.85, 0.62 + features['gradient_mean'] / 100)
                })
        
        quad_stats = features['quadrant_stats']
        low_density_count = sum(1 for q in quad_stats.values() if q['mean'] < 90)
        
        if low_density_count >= 2 or features['texture_homogeneity'] < 0.4:
            location = self._localize_bone_loss(quad_stats)
            severity = 'moderate' if low_density_count >= 3 else 'mild'
            
            detected.append({
                'condition': 'bone_loss',
                'location': location,
                'severity': severity,
                'confidence': min(0.82, 0.58 + low_density_count * 0.15)
            })
        
        if features['texture_contrast'] > 200 and features['morph_difference'] > 0.15:
            detected.append({
                'condition': 'bone_structure_anomaly',
                'location': self._localize_condition(features['quadrant_stats'], 'std'),
                'severity': 'moderate',
                'confidence': 0.76
            })
        
        irregular_contours = [c for c in features['contours'] 
                            if cv2.contourArea(c) > 300 and cv2.contourArea(c) < 2000]
        if len(irregular_contours) > 5 and features['texture_energy'] < 0.15:
            detected.append({
                'condition': 'cyst',
                'location': self._localize_condition(features['quadrant_stats'], 'dark_ratio'),
                'severity': 'mild',
                'confidence': 0.68
            })

        if features.get('bright_region_density', 0) > 0.004:
            _, bright_bin = cv2.threshold(preprocessed, 200, 255, cv2.THRESH_BINARY)
            b_contours, _ = cv2.findContours(bright_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            b_areas = [cv2.contourArea(c) for c in b_contours if cv2.contourArea(c) > 5]
            b_count = len(b_areas)
            severity = 'mild' if b_count < 3 else 'moderate' if b_count < 6 else 'severe'
            location = self._localize_condition(features['quadrant_stats'], 'bright_ratio')
            confidence = min(0.9, 0.5 + features['bright_region_density'] * 12 + (b_count * 0.04))
            detected.append({
                'condition': 'restoration',
                'location': location,
                'severity': severity,
                'confidence': confidence
            })
        
        if len(detected) == 0:
            detected.append({
                'condition': 'normal',
                'location': 'overall',
                'severity': 'none',
                'confidence': 0.85
            })
        
        return detected
    
    def _localize_condition(self, quadrant_stats, metric):
        scores = {name: stats[metric] for name, stats in quadrant_stats.items()}
        max_quadrant = max(scores, key=scores.get)
        
        location_map = {
            'upper_left': 'upper left region',
            'upper_right': 'upper right region',
            'lower_left': 'lower left region',
            'lower_right': 'lower right region'
        }
        
        return location_map.get(max_quadrant, 'dental region')
    
    def _localize_bone_loss(self, quadrant_stats):
        upper_loss = (quadrant_stats['upper_left']['mean'] + quadrant_stats['upper_right']['mean']) / 2 < 90
        lower_loss = (quadrant_stats['lower_left']['mean'] + quadrant_stats['lower_right']['mean']) / 2 < 90
        
        if upper_loss and lower_loss:
            return 'upper and lower jaw'
        elif upper_loss:
            return 'upper jaw'
        elif lower_loss:
            return 'lower jaw'
        else:
            return 'jaw region'
