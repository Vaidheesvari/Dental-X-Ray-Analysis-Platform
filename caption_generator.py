class DentalCaptionGenerator:
    def __init__(self):
        self.templates = {
            'cavity': {
                'mild': "The X-ray shows a small cavity in the {location}.",
                'moderate': "The X-ray reveals a cavity in the {location}.",
                'severe': "The X-ray indicates a significant cavity in the {location}."
            },
            'impacted_tooth': {
                'mild': "The X-ray shows signs of an impacted tooth in the {location}.",
                'moderate': "The X-ray indicates an impacted wisdom tooth in the {location}.",
                'severe': "The X-ray reveals a severely impacted tooth in the {location}."
            },
            'misalignment': {
                'mild': "The X-ray shows minor misalignment of the {location}.",
                'moderate': "The X-ray reveals misaligned {location}.",
                'severe': "The X-ray indicates significant misalignment of the {location}."
            },
            'bone_loss': {
                'mild': "The X-ray shows early signs of bone loss in the {location}.",
                'moderate': "The X-ray indicates bone loss around the {location}.",
                'severe': "The X-ray reveals significant bone loss in the {location}."
            },
            'normal': {
                'none': "The X-ray appears normal with no significant abnormalities detected. The teeth and bone structure appear healthy."
            },
            'bone_structure_anomaly': {
                'mild': "The X-ray shows minor bone structure irregularities in the {location}.",
                'moderate': "The X-ray indicates bone structure anomalies in the {location}.",
                'severe': "The X-ray reveals significant bone structure changes in the {location}."
            },
            'cyst': {
                'mild': "The X-ray shows a small cystic formation in the {location}.",
                'moderate': "The X-ray indicates a cyst in the {location}.",
                'severe': "The X-ray reveals a large cyst in the {location}."
            }
            ,
            'restoration': {
                'mild': "The X-ray shows dental restorations/fillings in the {location}.",
                'moderate': "The X-ray indicates multiple dental restorations or crowns in the {location}.",
                'severe': "The X-ray reveals extensive restorations or metallic crowns in the {location}."
            }
        }
    
    def generate_caption(self, detected_conditions):
        if not detected_conditions:
            return "Unable to analyze the X-ray image. Please ensure the image is clear and properly oriented."
        
        captions = []
        
        for condition in detected_conditions:
            condition_type = condition['condition']
            severity = condition.get('severity', 'moderate')
            location = condition.get('location', 'dental region')
            
            if condition_type in self.templates:
                if severity in self.templates[condition_type]:
                    template = self.templates[condition_type][severity]
                else:
                    template = self.templates[condition_type]['moderate']
                
                caption = template.format(location=location)
                captions.append(caption)
        
        if len(captions) > 1:
            final_caption = " Additionally, ".join(captions)
        else:
            final_caption = captions[0] if captions else "No significant findings detected."
        
        return final_caption
    
    def generate_detailed_report(self, detected_conditions):
        report = {
            'summary': self.generate_caption(detected_conditions),
            'findings': [],
            'recommendations': self._generate_recommendations(detected_conditions)
        }
        
        for condition in detected_conditions:
            finding = {
                'condition': condition['condition'].replace('_', ' ').title(),
                'location': condition.get('location', 'Unknown'),
                'severity': condition.get('severity', 'moderate').title(),
                'confidence': f"{condition.get('confidence', 0.5) * 100:.1f}%"
            }
            report['findings'].append(finding)
        
        return report
    
    def _generate_recommendations(self, detected_conditions):
        recommendations = []
        
        for condition in detected_conditions:
            condition_type = condition['condition']
            
            if condition_type == 'cavity':
                recommendations.append("Schedule a dental appointment for cavity treatment (filling or restoration).")
            elif condition_type == 'impacted_tooth':
                recommendations.append("Consult with a dental surgeon about the impacted tooth. Extraction may be necessary.")
            elif condition_type == 'misalignment':
                recommendations.append("Consider orthodontic consultation for teeth alignment correction.")
            elif condition_type == 'bone_loss':
                recommendations.append("Consult with a periodontist for bone loss evaluation and treatment options.")
            elif condition_type == 'normal':
                recommendations.append("Maintain good oral hygiene. Continue regular dental check-ups every 6 months.")
            elif condition_type == 'restoration':
                recommendations.append("Review existing dental restorations with your dentist to ensure integrity and fit.")
        
        if not recommendations:
            recommendations.append("Consult with your dentist for professional evaluation and treatment recommendations.")
        
        return list(set(recommendations))
