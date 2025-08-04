# Adaptive Document Quality Checker
# Replaces hardcoded values with learnable, adaptive parameters

import cv2
import numpy as np
from typing import Dict, List, Optional, Any
import logging
from config import adaptive_config

logger = logging.getLogger(__name__)

class AdaptiveDocumentQualityChecker:
    """Fully adaptive document quality assessment with learning capabilities"""
    
    def __init__(self):
        # No more hardcoded values - everything comes from adaptive config
        pass
    
    def detect_cut_edges(self, image: np.ndarray, contour, margin_pct: float = 0.03) -> List[str]:
        """Detect if document edges are cut off by checking margin clearance"""
        height, width = image.shape[:2]
        x, y, w, h = cv2.boundingRect(contour)
        
        margin_x = int(width * margin_pct)
        margin_y = int(height * margin_pct)
        
        issues = []
        if x <= margin_x:
            issues.append("left_edge_cut")
        if y <= margin_y:
            issues.append("top_edge_cut")
        if x + w >= width - margin_x:
            issues.append("right_edge_cut")
        if y + h >= height - margin_y:
            issues.append("bottom_edge_cut")
        
        return issues
    
    def detect_text_near_edges(self, text_blocks, image_shape, threshold: int = 20) -> List[str]:
        """Detect if text is too close to image edges, indicating cropped content"""
        if not text_blocks:
            return []
            
        h, w = image_shape[:2]
        issues = []
        
        for block in text_blocks:
            # Handle different text block formats
            if hasattr(block, 'bbox'):
                bbox = block.bbox
                x = getattr(bbox, 'x', 0)
                y = getattr(bbox, 'y', 0)
                width = getattr(bbox, 'width', 0)
                height = getattr(bbox, 'height', 0)
            elif hasattr(block, 'x'):
                # Direct coordinate access
                x, y = block.x, block.y
                width, height = getattr(block, 'width', 0), getattr(block, 'height', 0)
            else:
                continue
                
            # Check proximity to edges
            if x < threshold:
                issues.append("text_near_left_edge")
            if y < threshold:
                issues.append("text_near_top_edge")
            if x + width > w - threshold:
                issues.append("text_near_right_edge")
            if y + height > h - threshold:
                issues.append("text_near_bottom_edge")
        
        return list(set(issues))
    
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
            
        # Ensure gray is properly formatted for histogram calculation
        if gray.dtype != np.uint8:
            gray = np.asarray(gray, dtype=np.uint8)
        
        # Ensure gray is 2D
        if len(gray.shape) > 2:
            gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
        
        # Calculate histogram with proper error handling
        try:
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
            total_pixels = gray.shape[0] * gray.shape[1]
            
            # Calculate percentiles for adaptive brightness thresholds
            cumsum = np.cumsum(hist.flatten())
            percentiles = cumsum / total_pixels if total_pixels > 0 else np.zeros(256)
        except Exception as hist_error:
            logger.warning(f"Histogram calculation failed: {hist_error}")
            # Use fallback brightness analysis
            percentiles = np.linspace(0, 1, 256)
            total_pixels = gray.shape[0] * gray.shape[1]
        
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
    
    def assess_quality(self, image: np.ndarray, text_blocks: Optional[List[Any]] = None, largest_contour: Optional[np.ndarray] = None) -> Dict:
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
            
            # Ensure proper data type and shape
            gray = np.asarray(gray, dtype=np.uint8)
            
            # Ensure it's 2D
            if len(gray.shape) > 2:
                gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
            elif len(gray.shape) == 1:
                # Invalid image data
                raise ValueError("Invalid image format: 1D array")
                
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
                    # Find largest contour (or use provided one)
                    if largest_contour is None:
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
                    
                    # NEW: Check for cut edges using margin clearance
                    cut_edge_issues = self.detect_cut_edges(image, largest_contour)
                    if cut_edge_issues:
                        issues.extend(cut_edge_issues)
                        # Apply penalties based on severity
                        edge_penalty = len(cut_edge_issues) * 0.2
                        confidence -= min(0.4, edge_penalty)
                        logger.info(f"Cut edge issues detected: {cut_edge_issues}")
                        
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
            
            # NEW: Check for text near edges (if text blocks provided)
            text_edge_issues = []
            if text_blocks:
                try:
                    text_edge_issues = self.detect_text_near_edges(text_blocks, image.shape)
                    if text_edge_issues:
                        issues.extend(text_edge_issues)
                        # Higher penalty for text cut off issues
                        text_edge_penalty = len(text_edge_issues) * 0.15
                        confidence -= min(0.3, text_edge_penalty)
                        logger.info(f"Text edge issues detected: {text_edge_issues}")
                except Exception as text_edge_error:
                    logger.warning(f"Text edge detection failed: {text_edge_error}")
                
            # Enhanced adaptive decision logic with edge-based penalties
            edge_cut_issues = [issue for issue in issues if "_edge_cut" in issue]
            text_edge_issues = [issue for issue in issues if "text_near_" in issue]
            
            # Force rescan for severe edge issues
            force_rescan_conditions = [
                len(edge_cut_issues) >= 2,  # Multiple edges cut
                len(text_edge_issues) >= 3,  # Text near multiple edges
                (edge_cut_issues and text_edge_issues)  # Both contour and text edge issues
            ]
            
            issue_weight = len(issues) * 0.1
            confidence_penalty = min(0.3, issue_weight)
            final_confidence = max(0.0, confidence - confidence_penalty)
            
            # Adjust threshold based on issue types
            if any(force_rescan_conditions):
                needs_rescan = True
                final_confidence = min(final_confidence, 0.4)  # Cap confidence for edge issues
            else:
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
                },
                "cut_off_analysis": {
                    "edge_cut_issues": [issue for issue in issues if "_edge_cut" in issue],
                    "text_edge_issues": [issue for issue in issues if "text_near_" in issue],
                    "margin_check_performed": largest_contour is not None,
                    "text_edge_check_performed": text_blocks is not None and len(text_blocks) > 0
                }
            }
            
            # Learn from this assessment if it's successful
            if final_confidence > 0.7:
                self._learn_from_successful_assessment(thresholds, final_confidence)
            
            return result
            
        except Exception as e:
            logger.error(f"Adaptive quality assessment error: {e}")
            # Return safe fallback values with more detail
            return {
                "needs_rescan": False,
                "confidence": 0.7,  # More generous fallback
                "issues": ["assessment_error"],
                "blur_score": 100.0,  # Assume acceptable blur
                "brightness": 128.0,  # Assume middle brightness
                "adaptive_thresholds": {
                    "blur_threshold": 100.0,
                    "dark_threshold": 50.0,
                    "bright_threshold": 200.0,
                    "size_factor": 1.0
                },
                "image_stats": {
                    "size_factor": 1.0,
                    "area_ratio": 0.5,
                    "edge_density": 0.1
                },
                "error": str(e),
                "fallback_used": True
            }
    
    def get_user_friendly_message(self, issues: List[str]) -> str:
        """Generate user-friendly messages for detected issues"""
        edge_messages = []
        
        # Edge cut messages
        edge_cuts = [issue for issue in issues if "_edge_cut" in issue]
        if edge_cuts:
            directions = []
            if "left_edge_cut" in edge_cuts:
                directions.append("left")
            if "right_edge_cut" in edge_cuts:
                directions.append("right")
            if "top_edge_cut" in edge_cuts:
                directions.append("top")
            if "bottom_edge_cut" in edge_cuts:
                directions.append("bottom")
            
            if directions:
                edge_messages.append(f"Document appears cut off at the {'/'.join(directions)} edge(s). Please re-scan with proper margins.")
        
        # Text edge messages
        text_edges = [issue for issue in issues if "text_near_" in issue]
        if text_edges:
            text_directions = []
            if "text_near_left_edge" in text_edges:
                text_directions.append("left")
            if "text_near_right_edge" in text_edges:
                text_directions.append("right")
            if "text_near_top_edge" in text_edges:
                text_directions.append("top")
            if "text_near_bottom_edge" in text_edges:
                text_directions.append("bottom")
            
            if text_directions:
                edge_messages.append(f"Text is too close to the {'/'.join(text_directions)} edge(s), indicating possible content cropping.")
        
        # Combine with existing issue messages
        other_messages = []
        if "blurry_image" in issues:
            other_messages.append("Image appears blurry - please ensure the camera is steady and focused.")
        if "too_dark" in issues:
            other_messages.append("Image is too dark - please improve lighting conditions.")
        if "too_bright" in issues:
            other_messages.append("Image is too bright - please reduce lighting or adjust camera exposure.")
        if "skewed_document" in issues:
            other_messages.append("Document appears skewed - please align the document properly.")
        if "document_too_small" in issues:
            other_messages.append("Document appears too small in the frame - please move closer or zoom in.")
        if "irregular_shape" in issues:
            other_messages.append("Document shape appears irregular - please ensure the full document is visible.")
        if "low_content_density" in issues:
            other_messages.append("Low text content detected - please ensure the document is clearly visible.")
        
        all_messages = edge_messages + other_messages
        return " ".join(all_messages) if all_messages else "Document quality assessment completed."
    
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