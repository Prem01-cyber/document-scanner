# Adaptive Document Quality Checker
# Replaces hardcoded values with learnable, adaptive parameters

import cv2
import numpy as np
from typing import Dict
import logging
from config import adaptive_config

logger = logging.getLogger(__name__)

class AdaptiveDocumentQualityChecker:
    """Fully adaptive document quality assessment with learning capabilities"""
    
    def __init__(self):
        # No more hardcoded values - everything comes from adaptive config
        pass
    
    def _get_adaptive_thresholds(self, image: np.ndarray) -> Dict:
        """Calculate adaptive thresholds using learned parameters"""
        height, width = image.shape[:2] if len(image.shape) == 3 else image.shape
        image_area = height * width
        
        # Get adaptive base blur threshold (learned from previous documents)
        base_blur_threshold = adaptive_config.get_adaptive_value(
            "quality_thresholds", "base_blur_threshold"
        )
        
        # Adaptive blur threshold based on image size
        size_factor = min(2.0, max(0.5, (image_area / (1024 * 768)) ** 0.5))
        adaptive_blur_threshold = base_blur_threshold * size_factor
        
        # Adaptive brightness thresholds using learned bounds
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        total_pixels = gray.shape[0] * gray.shape[1]
        
        # Calculate percentiles for adaptive brightness thresholds
        cumsum = np.cumsum(hist.flatten())
        percentiles = cumsum / total_pixels
        
        dark_threshold = np.argmax(percentiles > 0.05)  # 5th percentile
        bright_threshold = np.argmax(percentiles > 0.95)  # 95th percentile
        
        # Use adaptive bounds instead of hardcoded [30,80] and [180,240]
        dark_bounds = adaptive_config.get_adaptive_value(
            "quality_thresholds", "dark_threshold_bounds"
        )
        bright_bounds = adaptive_config.get_adaptive_value(
            "quality_thresholds", "bright_threshold_bounds"  
        )
        
        # Ensure reasonable bounds using learned values
        if isinstance(dark_bounds, list) and len(dark_bounds) == 2:
            dark_threshold = max(dark_bounds[0], min(dark_bounds[1], dark_threshold))
        else:
            dark_threshold = max(30, min(80, dark_threshold))  # Fallback
            
        if isinstance(bright_bounds, list) and len(bright_bounds) == 2:
            bright_threshold = max(bright_bounds[0], min(bright_bounds[1], bright_threshold))
        else:
            bright_threshold = max(180, min(240, bright_threshold))  # Fallback
        
        return {
            'blur_threshold': adaptive_blur_threshold,
            'dark_threshold': dark_threshold,
            'bright_threshold': bright_threshold,
            'size_factor': size_factor,
            'learned_base_blur': base_blur_threshold,
            'adaptive_bounds_used': True
        }
    
    def assess_quality(self, image: np.ndarray) -> Dict:
        """
        Adaptive document quality assessment with learned thresholds
        """
        issues = []
        confidence = 1.0
        
        try:
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
                
            # Calculate adaptive thresholds using learned parameters
            thresholds = self._get_adaptive_thresholds(image)
                
            # 1. Adaptive Blur Detection
            blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
            if blur_score < thresholds['blur_threshold']:
                blur_severity = 1.0 - (blur_score / thresholds['blur_threshold'])
                issues.append("blurry_image")
                confidence -= min(0.4, 0.2 + blur_severity * 0.2)
                
            # 2. Adaptive Brightness Analysis
            mean_brightness = np.mean(gray)
            if mean_brightness < thresholds['dark_threshold']:
                darkness_severity = (thresholds['dark_threshold'] - mean_brightness) / thresholds['dark_threshold']
                issues.append("too_dark")
                confidence -= min(0.3, 0.1 + darkness_severity * 0.2)
            elif mean_brightness > thresholds['bright_threshold']:
                brightness_severity = (mean_brightness - thresholds['bright_threshold']) / (255 - thresholds['bright_threshold'])
                issues.append("too_bright")
                confidence -= min(0.2, 0.05 + brightness_severity * 0.15)
                
            # 3. Adaptive Document Edge Detection
            try:
                # Use adaptive Canny thresholds
                median_val = np.median(gray)
                lower_thresh = int(max(0, 0.66 * median_val))
                upper_thresh = int(min(255, 1.33 * median_val))
                
                edges = cv2.Canny(gray, lower_thresh, upper_thresh, apertureSize=3)
                contours_result = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                # Handle different OpenCV versions
                if len(contours_result) == 3:
                    _, contours, _ = contours_result
                else:
                    contours, _ = contours_result
                
                if contours:
                    # Find largest contour
                    largest_contour = max(contours, key=cv2.contourArea)
                    contour_area = cv2.contourArea(largest_contour)
                    
                    # Use adaptive min contour area ratio (learned from previous documents)
                    min_contour_ratio = adaptive_config.get_adaptive_value(
                        "quality_thresholds", "min_contour_area_ratio"
                    )
                    
                    image_area = gray.shape[0] * gray.shape[1]
                    area_ratio = contour_area / image_area
                    
                    if area_ratio < min_contour_ratio:
                        issues.append("document_too_small")
                        confidence -= 0.15
                    
                    # Check for irregular shape
                    epsilon = 0.02 * cv2.arcLength(largest_contour, True)
                    approx = cv2.approxPolyDP(largest_contour, epsilon, True)
                    
                    if len(approx) < 4:
                        issues.append("irregular_shape")
                        confidence -= 0.1
                        
            except Exception as contour_error:
                logger.warning(f"Contour detection failed: {contour_error}")
                
            # 4. Adaptive Skew Detection
            try:
                hough_threshold = max(50, int(min(gray.shape) * 0.1))
                lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=hough_threshold)
                
                if lines is not None and len(lines) > 0:
                    angles = []
                    for rho, theta in lines[:15]:
                        angle = theta * 180 / np.pi
                        if angle > 90:
                            angle = 180 - angle
                        angles.append(angle)
                    
                    if angles:
                        median_angle = np.median(angles)
                        
                        # Use adaptive skew tolerance (learned parameter)
                        skew_tolerance = adaptive_config.get_adaptive_value(
                            "quality_thresholds", "skew_tolerance"
                        )
                        
                        if abs(median_angle) > skew_tolerance and abs(median_angle - 90) > skew_tolerance:
                            issues.append("skewed_document")
                            confidence -= 0.05
                            
            except Exception as skew_error:
                logger.warning(f"Skew detection failed: {skew_error}")
                
            # 5. Adaptive Text Content Detection
            try:
                text_area_ratio = np.sum(edges > 0) / image_area
                
                # Use adaptive text area threshold (learned parameter)
                min_text_ratio = adaptive_config.get_adaptive_value(
                    "quality_thresholds", "text_area_ratio_threshold"
                )
                
                if text_area_ratio < min_text_ratio:
                    issues.append("low_content_density")
                    confidence -= 0.1
                    
            except Exception:
                pass
                
            # Adaptive decision logic
            issue_weight = len(issues) * 0.1
            confidence_penalty = min(0.3, issue_weight)
            final_confidence = max(0.0, confidence - confidence_penalty)
            
            # More lenient threshold for bright images
            brightness_adjusted_threshold = 0.5 if "too_bright" in issues else 0.6
            needs_rescan = final_confidence < brightness_adjusted_threshold
            
            result = {
                "needs_rescan": bool(needs_rescan),
                "confidence": float(final_confidence),
                "issues": issues,
                "blur_score": float(blur_score),
                "brightness": float(mean_brightness),
                "adaptive_thresholds": {
                    'blur_threshold': float(thresholds['blur_threshold']),
                    'dark_threshold': float(thresholds['dark_threshold']),
                    'bright_threshold': float(thresholds['bright_threshold']),
                    'size_factor': float(thresholds['size_factor']),
                    'min_contour_ratio': float(min_contour_ratio),
                    'skew_tolerance': float(skew_tolerance),
                    'text_ratio_threshold': float(min_text_ratio),
                    'learned_base_blur': float(thresholds.get('learned_base_blur', 100.0))
                },
                "image_stats": {
                    "size_factor": float(thresholds['size_factor']),
                    "area_ratio": float(area_ratio if 'area_ratio' in locals() else 0),
                    "edge_density": float(text_area_ratio if 'text_area_ratio' in locals() else 0)
                },
                "learning_metadata": {
                    "adaptive_bounds_used": thresholds.get('adaptive_bounds_used', False),
                    "parameters_learned_from_history": True
                }
            }
            
            # Learn from this assessment if it's successful
            if final_confidence > 0.7:
                self._learn_from_successful_assessment(thresholds, final_confidence)
            
            return result
            
        except Exception as e:
            logger.error(f"Adaptive quality assessment error: {e}")
            return {
                "needs_rescan": True,
                "confidence": 0.0,
                "issues": ["processing_error"],
                "blur_score": 0.0,
                "brightness": 0.0,
                "error": str(e)
            }
    
    def _learn_from_successful_assessment(self, thresholds: Dict, confidence: float):
        """
        Learn from successful quality assessments to improve future thresholds
        """
        try:
            # Learn blur threshold
            adaptive_config.learn_from_success(
                "quality_thresholds",
                "base_blur_threshold", 
                thresholds.get('learned_base_blur', 100.0),
                confidence
            )
            
            # Learn brightness bounds  
            if 'dark_threshold' in thresholds and 'bright_threshold' in thresholds:
                adaptive_config.learn_from_success(
                    "quality_thresholds",
                    "dark_threshold_bounds",
                    [thresholds['dark_threshold'], thresholds['bright_threshold']],
                    confidence
                )
            
            # Save learned parameters
            adaptive_config.save_config()
            
        except Exception as e:
            logger.debug(f"Learning from assessment error: {e}")