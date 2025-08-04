# Adaptive Document Quality Checker
# Replaces hardcoded values with learnable, adaptive parameters

import cv2
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import logging
import os
from src.config import adaptive_config

# ML model imports (with fallback)
try:
    import joblib
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    joblib = None

logger = logging.getLogger(__name__)

class AdaptiveDocumentQualityChecker:
    """Fully adaptive document quality assessment with learning capabilities"""
    
    def __init__(self, ml_model_path: str = "quality_rescan_model.pkl"):
        # No more hardcoded values - everything comes from adaptive config
        self.ml_model_path = ml_model_path
        self.ml_model_data = None
        self._load_ml_model()
    
    def _load_ml_model(self) -> bool:
        """Load ML model if available"""
        if not ML_AVAILABLE:
            logger.warning("ML dependencies not available. Only rule-based scoring will be used.")
            return False
            
        if not os.path.exists(self.ml_model_path):
            logger.info(f"ML model not found at {self.ml_model_path}. Only rule-based scoring will be used.")
            return False
            
        try:
            self.ml_model_data = joblib.load(self.ml_model_path)
            logger.info(f"Loaded ML model: {self.ml_model_data.get('model_type', 'unknown')}")
            logger.info(f"Model performance: ROC AUC = {self.ml_model_data.get('performance', {}).get('roc_auc', 'unknown')}")
            return True
        except Exception as e:
            logger.warning(f"Failed to load ML model: {e}")
            self.ml_model_data = None
            return False
    
    def predict_rescan_with_ml(self, metrics: Dict) -> Tuple[Optional[int], Optional[float], str]:
        """Predict rescan need using ML model
        
        Returns:
            (prediction, probability, status)
            - prediction: 0 or 1, None if model unavailable
            - probability: probability of needing rescan, None if model unavailable  
            - status: "success", "model_unavailable", "prediction_error"
        """
        if not self.ml_model_data:
            return None, None, "model_unavailable"
            
        try:
            model = self.ml_model_data['model']
            scaler = self.ml_model_data.get('scaler')
            feature_columns = self.ml_model_data['feature_columns']
            
            # Extract features in correct order
            features = []
            for col in feature_columns:
                if col == "brightness_issue":
                    features.append(int(metrics.get(col, False)))
                else:
                    features.append(metrics.get(col, 0.0))
            
            # Convert to numpy array
            X = np.array(features).reshape(1, -1)
            
            # Apply scaling if needed (for logistic regression)
            if scaler is not None:
                X = scaler.transform(X)
            
            # Predict
            probability = model.predict_proba(X)[0][1]  # Probability of class 1 (needs rescan)
            prediction = int(probability > 0.5)
            
            return prediction, float(probability), "success"
            
        except Exception as e:
            logger.error(f"ML prediction error: {e}")
            return None, None, "prediction_error"
    
    def compute_quality_risk_score(self, metrics: Dict, weights: Dict = None, document_type: str = "general") -> Tuple[float, List[str], str]:
        """
        Compute a final quality risk score from various quality metrics.
        Args:
            metrics: Dictionary of extracted metrics from quality checker
            weights: Optional weights per metric. Defaults provided below.
            document_type: Document type for adaptive weight selection

        Returns:
            (risk_score: float, issues: List[str], decision: str)
        """
        # Get adaptive weights if available, otherwise use defaults
        if weights is None:
            weights = adaptive_config.get_adaptive_value(
                "quality_risk_weights", document_type
            )
            
            # Fallback to sensible defaults if not learned yet
            if not isinstance(weights, dict):
                weights = {
                    "blur_score": 0.35,
                    "edge_cut_score": 0.25,
                    "text_density_score": 0.2,
                    "brightness_score": 0.1,
                    "skew_score": 0.1
                }
        
        # Normalize scores (0 = good, 1 = bad)
        score_components = {
            "blur_score": min(1.0, 1.0 - metrics.get("blur_confidence", 1.0)),  # Low blur confidence → high risk
            "edge_cut_score": float(metrics.get("edge_cut_flags", 0)) / 4.0,    # Max 4 edges
            "text_density_score": float(metrics.get("text_density_violations", 0)) / 4.0,
            "brightness_score": 1.0 if metrics.get("brightness_issue", False) else 0.0,
            "skew_score": 1.0 if abs(metrics.get("skew_angle", 0)) > 10 else 0.0
        }
        
        # Compute weighted total score
        total_score = sum(score_components[k] * weights.get(k, 0.0) for k in score_components)
        
        # Document-type specific thresholds (can be made adaptive too)
        reject_threshold = 0.65
        warn_threshold = 0.4
        
        # Adjust thresholds for document types that need higher quality
        if document_type in ["certificate", "legal", "financial"]:
            reject_threshold = 0.55  # More strict
            warn_threshold = 0.35
        elif document_type in ["receipt", "note"]:
            reject_threshold = 0.75  # More lenient
            warn_threshold = 0.5
        
        # Risk decision
        if total_score >= reject_threshold:
            decision = "reject"
        elif total_score >= warn_threshold:
            decision = "warn"
        else:
            decision = "accept"
        
        # Explanation generation
        reasons = []
        if score_components["blur_score"] > 0.6:
            reasons.append("Blurry scan")
        if score_components["edge_cut_score"] > 0.5:
            reasons.append("Document may be cut")
        if score_components["text_density_score"] > 0.5:
            reasons.append("Text too close to image edges")
        if score_components["brightness_score"] > 0.5:
            reasons.append("Over or underexposed")
        if score_components["skew_score"] > 0.5:
            reasons.append("Image is skewed")
        
        # Log the scoring breakdown for debugging
        logger.debug(f"Risk scoring for {document_type}: total={total_score:.3f}, components={score_components}, decision={decision}")
        
        return total_score, reasons, decision
    
    def detect_cut_edges(self, image: np.ndarray, contour, document_type: str = "general") -> List[str]:
        """Detect if document edges are cut off using adaptive margin thresholds"""
        height, width = image.shape[:2]
        x, y, w, h = cv2.boundingRect(contour)
        
        # Get adaptive margin percentage based on document type
        margin_pct = adaptive_config.get_adaptive_value(
            "quality_thresholds", "cut_edge_margin_pct", document_type
        )
        
        # Fallback margins for different document types
        if not isinstance(margin_pct, (int, float)):
            margin_defaults = {
                "form": 0.015,
                "letter": 0.05,
                "receipt": 0.02,
                "certificate": 0.04,
                "general": 0.03
            }
            margin_pct = margin_defaults.get(document_type, 0.03)
        
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
    
    def detect_text_density_near_edges(self, text_blocks, image_shape, edge_threshold: int = 30, density_ratio: float = 0.12) -> Dict[str, Any]:
        """Detect high text density near edges indicating possible cut-off"""
        if not text_blocks:
            return {"issues": [], "density_ratios": {}, "total_blocks": 0}
            
        h, w = image_shape[:2]
        total_blocks = len(text_blocks)
        edge_counts = {"left": 0, "right": 0, "top": 0, "bottom": 0}
        
        for block in text_blocks:
            # Handle different text block formats
            if hasattr(block, 'bbox'):
                bbox = block.bbox
                x = getattr(bbox, 'x', 0)
                y = getattr(bbox, 'y', 0)
                width = getattr(bbox, 'width', 0)
                height = getattr(bbox, 'height', 0)
            elif hasattr(block, 'x'):
                x, y = block.x, block.y
                width, height = getattr(block, 'width', 0), getattr(block, 'height', 0)
            else:
                continue
                
            # Count blocks near each edge
            if x < edge_threshold:
                edge_counts["left"] += 1
            if y < edge_threshold:
                edge_counts["top"] += 1
            if x + width > w - edge_threshold:
                edge_counts["right"] += 1
            if y + height > h - edge_threshold:
                edge_counts["bottom"] += 1
        
        # Calculate density ratios
        density_ratios = {edge: count / total_blocks for edge, count in edge_counts.items()}
        
        # Identify violations
        issues = []
        for edge, ratio in density_ratios.items():
            if ratio > density_ratio:
                issues.append(f"edge_density_violation_{edge}")
        
        return {
            "issues": issues,
            "density_ratios": density_ratios,
            "total_blocks": total_blocks,
            "edge_counts": edge_counts
        }
    
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
    
    def extract_quality_metrics(self, image: np.ndarray, text_blocks: Optional[List[Any]] = None, 
                               blur_score: float = 0.0, brightness: float = 128.0, skew_angle: float = 0.0,
                               edge_cut_issues: List[str] = None, text_density_analysis: Dict = None, 
                               area_ratio: float = 0.85) -> Dict:
        """Extract quality metrics in the format expected by risk scoring"""
        edge_cut_issues = edge_cut_issues or []
        text_density_analysis = text_density_analysis or {}
        
        # Normalize blur score to confidence (higher Laplacian variance = better quality)
        blur_confidence = min(1.0, blur_score / 200.0)  # Assume 200+ is good quality
        
        # Count edge cut flags
        edge_cut_flags = len(edge_cut_issues)
        
        # Count text density violations
        density_violations = len(text_density_analysis.get("issues", []))
        
        # Determine brightness issue
        brightness_issue = brightness < 50 or brightness > 200
        
        return {
            "blur_confidence": blur_confidence,
            "edge_cut_flags": edge_cut_flags,
            "text_density_violations": density_violations,
            "brightness_issue": brightness_issue,
            "skew_angle": abs(skew_angle),
            "document_area_ratio": area_ratio
        }
    
    def assess_quality(self, image: np.ndarray, text_blocks: Optional[List[Any]] = None, largest_contour: Optional[np.ndarray] = None, document_type: str = "general") -> Dict:
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
                    
                    # NEW: Check for cut edges using adaptive margin clearance
                    cut_edge_issues = self.detect_cut_edges(image, largest_contour, document_type)
                    if cut_edge_issues:
                        issues.extend(cut_edge_issues)
                        # Apply penalties based on severity
                        edge_penalty = len(cut_edge_issues) * 0.2
                        confidence -= min(0.4, edge_penalty)
                        logger.info(f"Cut edge issues detected: {cut_edge_issues}")
                        
            except Exception as contour_error:
                logger.warning(f"Contour detection failed: {contour_error}")
                
            # 4. Adaptive Skew Detection
            median_angle = 0.0  # Default
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
                        self._last_skew_angle = median_angle  # Store for risk scoring
                        
                        # Use adaptive skew tolerance (learned parameter)
                        skew_tolerance = adaptive_config.get_adaptive_value(
                            "quality_thresholds", "skew_tolerance"
                        )
                        
                        if abs(median_angle) > skew_tolerance and abs(median_angle - 90) > skew_tolerance:
                            issues.append("skewed_document")
                            confidence -= 0.05
                            
            except Exception as skew_error:
                logger.warning(f"Skew detection failed: {skew_error}")
                self._last_skew_angle = 0.0
                
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
            
            # NEW: Comprehensive text edge analysis
            text_edge_issues = []
            text_density_analysis = {}
            if text_blocks:
                try:
                    # Check for text near edges
                    text_edge_issues = self.detect_text_near_edges(text_blocks, image.shape)
                    if text_edge_issues:
                        issues.extend(text_edge_issues)
                        # Higher penalty for text cut off issues
                        text_edge_penalty = len(text_edge_issues) * 0.15
                        confidence -= min(0.3, text_edge_penalty)
                        logger.info(f"Text edge issues detected: {text_edge_issues}")
                    
                    # Check for text density violations near edges
                    text_density_analysis = self.detect_text_density_near_edges(text_blocks, image.shape)
                    density_issues = text_density_analysis.get("issues", [])
                    if density_issues:
                        issues.extend(density_issues)
                        # Penalty for density violations
                        density_penalty = len(density_issues) * 0.1
                        confidence -= min(0.2, density_penalty)
                        logger.info(f"Text density violations detected: {density_issues}")
                        
                except Exception as text_edge_error:
                    logger.warning(f"Text edge detection failed: {text_edge_error}")
                
            # Enhanced adaptive decision logic with comprehensive edge analysis
            edge_cut_issues = [issue for issue in issues if "_edge_cut" in issue]
            text_edge_issues = [issue for issue in issues if "text_near_" in issue]
            density_violations = [issue for issue in issues if "edge_density_violation" in issue]
            
            # Categorize rescan reasons and urgency
            rescan_reasons = []
            rescan_urgency = "low"
            
            # Critical edge issues (high urgency)
            critical_conditions = [
                len(edge_cut_issues) >= 2,  # Multiple edges cut
                len(text_edge_issues) >= 3,  # Text near multiple edges
                (edge_cut_issues and text_edge_issues),  # Both contour and text edge issues
                len(density_violations) >= 2  # Multiple density violations
            ]
            
            if any(critical_conditions):
                rescan_urgency = "high"
                if len(edge_cut_issues) >= 2:
                    rescan_reasons.append("multiple_edges_cut")
                if len(text_edge_issues) >= 3:
                    rescan_reasons.append("text_near_multiple_edges")
                if edge_cut_issues and text_edge_issues:
                    rescan_reasons.append("combined_edge_issues")
                if len(density_violations) >= 2:
                    rescan_reasons.append("high_text_density_near_edges")
            
            # Medium urgency conditions
            medium_conditions = [
                len(edge_cut_issues) == 1,
                len(text_edge_issues) in [1, 2],
                len(density_violations) == 1,
                ("low_content_density" in issues and (edge_cut_issues or text_edge_issues))
            ]
            
            if any(medium_conditions) and rescan_urgency == "low":
                rescan_urgency = "medium"
                if len(edge_cut_issues) == 1:
                    rescan_reasons.append("single_edge_cut")
                if len(text_edge_issues) in [1, 2]:
                    rescan_reasons.append("text_near_edge")
                if len(density_violations) == 1:
                    rescan_reasons.append("text_density_violation")
            
            # Standard quality issues
            if "blurry_image" in issues:
                rescan_reasons.append("blurry_image")
            if "too_dark" in issues or "too_bright" in issues:
                rescan_reasons.append("poor_lighting")
            if "skewed_document" in issues:
                rescan_reasons.append("document_skewed")
            
            issue_weight = len(issues) * 0.1
            confidence_penalty = min(0.3, issue_weight)
            final_confidence = max(0.0, confidence - confidence_penalty)
            
            # Extract quality metrics for risk scoring
            quality_metrics = self.extract_quality_metrics(
                image=image,
                text_blocks=text_blocks,
                blur_score=blur_score,
                brightness=mean_brightness,
                skew_angle=getattr(self, '_last_skew_angle', 0.0),  # Store from skew detection
                edge_cut_issues=edge_cut_issues,
                text_density_analysis=text_density_analysis,
                area_ratio=area_ratio if 'area_ratio' in locals() else 0.85
            )
            
            # Compute risk-based quality score
            risk_score, risk_reasons, risk_decision = self.compute_quality_risk_score(
                quality_metrics, document_type=document_type
            )
            
            # Get ML prediction if available
            ml_prediction, ml_probability, ml_status = self.predict_rescan_with_ml(quality_metrics)
            
            # Determine final decision (prioritize rule-based for now, but log both)
            needs_rescan = risk_decision == "reject"
            final_confidence = round(1.0 - risk_score, 2)
            
            # Update rescan reasons with risk-based analysis
            if risk_reasons:
                rescan_reasons.extend(risk_reasons)
            
            # Update urgency based on risk score
            if risk_score >= 0.65:
                rescan_urgency = "high"
            elif risk_score >= 0.4:
                rescan_urgency = "medium"
            else:
                rescan_urgency = "low"
            
            # Check agreement between rule-based and ML (if available)
            rule_vs_ml_agreement = None
            if ml_prediction is not None:
                rule_vs_ml_agreement = (ml_prediction == int(needs_rescan))
            
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
                    "edge_cut_issues": edge_cut_issues,
                    "text_edge_issues": text_edge_issues,
                    "density_violations": density_violations,
                    "text_density_ratios": text_density_analysis.get("density_ratios", {}),
                    "edge_counts": text_density_analysis.get("edge_counts", {}),
                    "margin_check_performed": largest_contour is not None,
                    "text_edge_check_performed": text_blocks is not None and len(text_blocks) > 0,
                    "document_type": document_type
                },
                "rescan_decision": {
                    "needs_rescan": bool(needs_rescan),
                    "rescan_reasons": rescan_reasons,
                    "rescan_urgency": rescan_urgency,
                    "rescan_decision": risk_decision,
                    "user_message": self.get_user_friendly_message(issues)
                },
                "quality_risk_assessment": {
                    "quality_risk_score": round(risk_score, 2),
                    "risk_decision": risk_decision,
                    "risk_reasons": risk_reasons,
                    "quality_metrics": quality_metrics
                },
                "ml_assessment": {
                    "ml_available": self.ml_model_data is not None,
                    "ml_prediction": ml_prediction,
                    "ml_rescan_probability": round(ml_probability, 3) if ml_probability is not None else None,
                    "ml_status": ml_status,
                    "rule_vs_ml_agreement": rule_vs_ml_agreement
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
                edge_messages.append(f"Text too close to {'/'.join(text_directions)} edge(s). Possibly cut off.")
        
        # Density violation messages
        density_violations = [issue for issue in issues if "edge_density_violation" in issue]
        if density_violations:
            affected_edges = []
            for issue in density_violations:
                if "_left" in issue:
                    affected_edges.append("left")
                elif "_right" in issue:
                    affected_edges.append("right")
                elif "_top" in issue:
                    affected_edges.append("top")
                elif "_bottom" in issue:
                    affected_edges.append("bottom")
            
            if affected_edges:
                edge_messages.append(f"Too much text near {'/'.join(affected_edges)} border(s) — possible scan error.")
        
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
    
    def learn_from_risk_scoring_feedback(self, document_type: str, actual_quality: str, predicted_decision: str, risk_score: float):
        """Learn from feedback on risk scoring accuracy"""
        try:
            current_weights = adaptive_config.get_adaptive_value(
                "quality_risk_weights", document_type
            )
            
            if not isinstance(current_weights, dict):
                # Initialize with defaults if not present
                current_weights = {
                    "blur_score": 0.35,
                    "edge_cut_score": 0.25,
                    "text_density_score": 0.2,
                    "brightness_score": 0.1,
                    "skew_score": 0.1
                }
            
            # Simple learning: adjust weights based on feedback
            # This is a basic implementation - could be made more sophisticated
            adjustment_factor = 0.05
            
            if actual_quality == "poor" and predicted_decision == "accept":
                # False negative - we were too lenient, increase sensitivity
                for key in current_weights:
                    if key in ["blur_score", "edge_cut_score"]:  # Most important for quality
                        current_weights[key] = min(0.5, current_weights[key] + adjustment_factor)
                logger.info(f"Increased quality sensitivity for {document_type} due to false negative")
                
            elif actual_quality == "good" and predicted_decision == "reject":
                # False positive - we were too strict, decrease sensitivity
                for key in current_weights:
                    current_weights[key] = max(0.05, current_weights[key] - adjustment_factor)
                logger.info(f"Decreased quality sensitivity for {document_type} due to false positive")
            
            # Normalize weights to sum to 1.0
            total_weight = sum(current_weights.values())
            if total_weight > 0:
                current_weights = {k: v / total_weight for k, v in current_weights.items()}
            
            # Store learned weights
            adaptive_config.learn_from_success(
                "quality_risk_weights",
                document_type,
                current_weights,
                0.7  # Moderate confidence for weight learning
            )
            
            adaptive_config.save_config()
            logger.info(f"Updated risk weights for {document_type}: {current_weights}")
            
        except Exception as e:
            logger.warning(f"Risk weight learning error for {document_type}: {e}")

    def learn_from_margin_feedback(self, document_type: str, was_false_positive: bool, current_margin: float):
        """Learn from user feedback about margin detection accuracy"""
        try:
            current_value = adaptive_config.get_adaptive_value(
                "quality_thresholds", "cut_edge_margin_pct", document_type
            )
            
            if not isinstance(current_value, (int, float)):
                current_value = current_margin
            
            if was_false_positive:
                # Reduce sensitivity (increase margin) if it was a false positive
                new_margin = min(0.08, current_value * 1.2)  # Cap at 8%
                logger.info(f"Reducing margin sensitivity for {document_type}: {current_value:.3f} -> {new_margin:.3f}")
            else:
                # Increase sensitivity (decrease margin) if it was a true positive that we should catch better
                new_margin = max(0.005, current_value * 0.9)  # Floor at 0.5%  
                logger.info(f"Increasing margin sensitivity for {document_type}: {current_value:.3f} -> {new_margin:.3f}")
            
            # Update the learned margin
            adaptive_config.learn_from_success(
                "quality_thresholds",
                "cut_edge_margin_pct",
                {document_type: new_margin},
                0.8  # Moderate confidence for margin learning
            )
            
            adaptive_config.save_config()
            
        except Exception as e:
            logger.warning(f"Margin learning error for {document_type}: {e}")
    
    def update_text_edge_threshold_from_usage(self, document_type: str, successful_scans: int, edge_violations: int):
        """Update text edge detection threshold based on usage patterns"""
        try:
            if successful_scans > 10:  # Only learn after sufficient data
                violation_rate = edge_violations / successful_scans
                
                # Current threshold (default 20 pixels)
                current_threshold = 20
                
                if violation_rate < 0.05:  # Very few violations - can be more sensitive
                    new_threshold = max(15, current_threshold * 0.9)
                elif violation_rate > 0.15:  # Too many violations - be less sensitive  
                    new_threshold = min(35, current_threshold * 1.1)
                else:
                    new_threshold = current_threshold  # Keep current
                
                if new_threshold != current_threshold:
                    logger.info(f"Adjusting text edge threshold for {document_type}: {current_threshold} -> {new_threshold}")
                    
                    # Store learning data (this would need to be expanded in adaptive_config if needed)
                    # For now, we'll just log the learning
                    
        except Exception as e:
            logger.warning(f"Text edge threshold learning error: {e}")

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
    
    def demo_risk_scoring(self, document_type: str = "general") -> Dict:
        """Demo function to show risk scoring with sample metrics"""
        # Sample metrics for different quality scenarios
        scenarios = {
            "excellent": {
                "blur_confidence": 0.95,
                "edge_cut_flags": 0,
                "text_density_violations": 0,
                "brightness_issue": False,
                "skew_angle": 2.0
            },
            "poor": {
                "blur_confidence": 0.3,
                "edge_cut_flags": 3,
                "text_density_violations": 2,
                "brightness_issue": True,
                "skew_angle": 18.0
            },
            "marginal": {
                "blur_confidence": 0.65,
                "edge_cut_flags": 1,
                "text_density_violations": 1,
                "brightness_issue": False,
                "skew_angle": 6.0
            }
        }
        
        results = {}
        for scenario_name, metrics in scenarios.items():
            risk_score, reasons, decision = self.compute_quality_risk_score(metrics, document_type=document_type)
            results[scenario_name] = {
                "risk_score": round(risk_score, 3),
                "decision": decision,
                "reasons": reasons,
                "confidence": round(1.0 - risk_score, 3)
            }
        
        return results
    
    def compare_prediction_methods(self, quality_metrics: Dict, document_type: str = "general") -> Dict:
        """Compare rule-based vs ML predictions side by side"""
        # Rule-based prediction
        risk_score, risk_reasons, risk_decision = self.compute_quality_risk_score(
            quality_metrics, document_type=document_type
        )
        
        # ML prediction  
        ml_prediction, ml_probability, ml_status = self.predict_rescan_with_ml(quality_metrics)
        
        return {
            "quality_metrics": quality_metrics,
            "rule_based": {
                "risk_score": round(risk_score, 3),
                "decision": risk_decision,
                "reasons": risk_reasons,
                "confidence": round(1.0 - risk_score, 3),
                "needs_rescan": risk_decision == "reject"
            },
            "ml_based": {
                "prediction": ml_prediction,
                "probability": round(ml_probability, 3) if ml_probability else None,
                "needs_rescan": bool(ml_prediction) if ml_prediction is not None else None,
                "confidence": round(max(ml_probability, 1-ml_probability), 3) if ml_probability else None,
                "status": ml_status
            },
            "comparison": {
                "agreement": (ml_prediction == int(risk_decision == "reject")) if ml_prediction is not None else None,
                "rule_more_strict": (risk_decision == "reject" and ml_prediction == 0) if ml_prediction is not None else None,
                "ml_more_strict": (risk_decision != "reject" and ml_prediction == 1) if ml_prediction is not None else None
            }
        }
    
    def log_quality_assessment_for_training(self, quality_metrics: Dict, actual_needs_rescan: Optional[bool] = None, 
                                          user_feedback: Optional[str] = None, document_type: str = "general") -> None:
        """Log quality assessment data for ML training
        
        Args:
            quality_metrics: Extracted quality metrics
            actual_needs_rescan: Actual ground truth (if known)
            user_feedback: User feedback about quality
            document_type: Type of document
        """
        try:
            log_data = {
                **quality_metrics,
                "document_type": document_type,
                "actual_needs_rescan": actual_needs_rescan,
                "user_feedback": user_feedback,
                "timestamp": logger.name  # Simple timestamp placeholder
            }
            
            # In a real implementation, you'd append to a CSV or database
            logger.info(f"Quality assessment logged for training: {log_data}")
            
            # TODO: Implement actual logging to CSV/database
            # This could append to quality_training_data.csv
            
        except Exception as e:
            logger.warning(f"Failed to log quality assessment: {e}")