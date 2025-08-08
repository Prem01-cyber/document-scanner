#!/usr/bin/env python3
"""
Advanced Quality Assessment Features using SciPy, Scikit-Image, and Statistical Analysis

This module provides enhanced quality assessment techniques beyond basic OpenCV operations,
incorporating advanced computer vision and statistical analysis methods.
"""

import numpy as np
import logging
import cv2
from typing import Dict, List, Tuple, Optional

logger = logging.getLogger(__name__)

# Import advanced libraries with graceful fallback and detailed diagnostics
try:
    from scipy.stats import skew, entropy, zscore
    from scipy.spatial.distance import pdist
    from scipy.ndimage import label, gaussian_filter
    SCIPY_AVAILABLE = True
    logger.info("✅ SciPy successfully imported for advanced quality features")
except ImportError as e:
    SCIPY_AVAILABLE = False
    logger.warning(f"❌ SciPy not available: {e}. Using basic quality features only.")

try:
    from skimage.filters import sobel
    from skimage.feature import corner_harris
    from skimage.morphology import binary_closing, disk
    SKIMAGE_AVAILABLE = True
    logger.info("✅ Scikit-image successfully imported for enhanced blur detection")
except ImportError as e:
    SKIMAGE_AVAILABLE = False
    logger.warning(f"❌ Scikit-image not available: {e}. Using basic quality features only.")

try:
    import statsmodels.api as sm
    STATSMODELS_AVAILABLE = True
    logger.info("✅ Statsmodels successfully imported for statistical modeling")
except ImportError as e:
    STATSMODELS_AVAILABLE = False
    logger.warning(f"❌ Statsmodels not available: {e}. Using basic statistical modeling only.")

# Log overall availability status
logger.info(f"🔬 Advanced Quality Features Status: SciPy={SCIPY_AVAILABLE}, Scikit-image={SKIMAGE_AVAILABLE}, Statsmodels={STATSMODELS_AVAILABLE}")


class AdvancedQualityAnalyzer:
    """Advanced quality analysis using statistical and computer vision techniques"""
    
    def __init__(self):
        self.feature_cache = {}
        
    def enhanced_blur_detection(self, gray_image: np.ndarray) -> Dict[str, float]:
        """
        Enhanced blur detection using multiple methods:
        1. Tenengrad gradient energy (Sobel-based)
        2. Laplacian variance (existing)
        3. Brenner's focus measure
        """
        results = {}
        
        # Basic Laplacian variance (existing method)
        laplacian_var = cv2.Laplacian(gray_image, cv2.CV_64F).var()
        results['laplacian_variance'] = float(laplacian_var)
        
        if SKIMAGE_AVAILABLE:
            try:
                # Tenengrad gradient energy (more robust than Laplacian)
                sobel_map = sobel(gray_image)
                tenengrad_score = np.mean(sobel_map ** 2)
                results['tenengrad_energy'] = float(tenengrad_score)
                
                # Normalized tenengrad (0-1 scale)
                results['tenengrad_normalized'] = min(1.0, tenengrad_score / 100.0)
                
            except Exception as e:
                logger.debug(f"Tenengrad calculation failed: {e}")
                results['tenengrad_energy'] = laplacian_var / 1000.0
                results['tenengrad_normalized'] = min(1.0, laplacian_var / 100000.0)
        else:
            # Fallback using OpenCV Sobel
            try:
                sobelx = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
                sobely = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
                sobel_magnitude = np.sqrt(sobelx**2 + sobely**2)
                tenengrad_score = np.mean(sobel_magnitude ** 2)
                results['tenengrad_energy'] = float(tenengrad_score)
                results['tenengrad_normalized'] = min(1.0, tenengrad_score / 10000.0)
            except Exception as e:
                logger.debug(f"Fallback Sobel calculation failed: {e}")
                results['tenengrad_energy'] = laplacian_var / 1000.0
                results['tenengrad_normalized'] = min(1.0, laplacian_var / 100000.0)
        
        # Brenner's focus measure (sum of squared differences)
        try:
            brenner_score = np.sum((gray_image[2:, :] - gray_image[:-2, :]) ** 2)
            results['brenner_focus'] = float(brenner_score)
            results['brenner_normalized'] = min(1.0, brenner_score / 1000000.0)
        except Exception as e:
            logger.debug(f"Brenner focus calculation failed: {e}")
            results['brenner_focus'] = laplacian_var * 1000.0
            results['brenner_normalized'] = results['tenengrad_normalized']
        
        # Composite blur confidence (weighted combination)
        weights = {'tenengrad': 0.5, 'laplacian': 0.3, 'brenner': 0.2}
        composite_score = (
            weights['tenengrad'] * results['tenengrad_normalized'] +
            weights['laplacian'] * min(1.0, laplacian_var / 100.0) +
            weights['brenner'] * results['brenner_normalized']
        )
        results['composite_blur_confidence'] = float(composite_score)
        
        return results
    
    def advanced_brightness_analysis(self, gray_image: np.ndarray) -> Dict[str, float]:
        """
        Advanced brightness and contrast analysis using statistical measures
        """
        results = {}
        
        # Basic statistics
        results['mean_brightness'] = float(np.mean(gray_image))
        results['std_brightness'] = float(np.std(gray_image))
        results['brightness_range'] = float(np.ptp(gray_image))  # Peak-to-peak range
        
        # Histogram analysis
        hist, _ = np.histogram(gray_image, bins=256, range=(0, 256))
        hist_normalized = hist / np.sum(hist)
        
        if SCIPY_AVAILABLE:
            try:
                # Statistical measures of brightness distribution
                results['brightness_skewness'] = float(skew(hist_normalized))
                results['brightness_entropy'] = float(entropy(hist_normalized + 1e-10))  # Add small epsilon to avoid log(0)
                
                # Detect over/under exposure patterns
                results['overexposure_ratio'] = float(np.sum(hist[240:]) / np.sum(hist))  # Pixels > 240
                results['underexposure_ratio'] = float(np.sum(hist[:16]) / np.sum(hist))  # Pixels < 16
                
            except Exception as e:
                logger.debug(f"SciPy brightness analysis failed: {e}")
                # Fallback calculations
                results['brightness_skewness'] = 0.0
                results['brightness_entropy'] = 0.5
                results['overexposure_ratio'] = float(np.sum(gray_image > 240) / gray_image.size)
                results['underexposure_ratio'] = float(np.sum(gray_image < 16) / gray_image.size)
        else:
            # Basic fallback calculations
            results['brightness_skewness'] = 0.0
            results['brightness_entropy'] = 0.5
            results['overexposure_ratio'] = float(np.sum(gray_image > 240) / gray_image.size)
            results['underexposure_ratio'] = float(np.sum(gray_image < 16) / gray_image.size)
        
        # Contrast measures
        results['rms_contrast'] = float(np.sqrt(np.mean((gray_image - np.mean(gray_image)) ** 2)))
        
        # Safe Michelson contrast calculation to avoid overflow
        try:
            max_val = float(np.max(gray_image))
            min_val = float(np.min(gray_image))
            numerator = max_val - min_val
            denominator = max_val + min_val
            
            if denominator > 0:
                results['michelson_contrast'] = numerator / denominator
            else:
                results['michelson_contrast'] = 0.0
        except (OverflowError, RuntimeWarning):
            results['michelson_contrast'] = 0.0
        
        # Quality flags
        results['brightness_issue'] = (
            results['overexposure_ratio'] > 0.1 or 
            results['underexposure_ratio'] > 0.1 or
            results['brightness_entropy'] < 0.3
        )
        
        return results
    
    def text_layout_analysis(self, text_blocks: List, image_shape: Tuple[int, int]) -> Dict[str, float]:
        """
        Statistical analysis of text block positioning and spacing
        """
        results = {
            'text_block_count': len(text_blocks),
            'text_spacing_variance': 0.0,
            'text_alignment_score': 0.0,
            'text_density_uniformity': 0.0
        }
        
        if len(text_blocks) < 2:
            return results
        
        try:
            # Extract text block coordinates
            coordinates = []
            x_positions = []
            y_positions = []
            
            for block in text_blocks:
                if hasattr(block, 'bbox'):
                    bbox = block.bbox
                    x = getattr(bbox, 'x', 0)
                    y = getattr(bbox, 'y', 0)
                elif hasattr(block, 'x'):
                    x, y = block.x, block.y
                else:
                    continue
                    
                coordinates.append((x, y))
                x_positions.append(x)
                y_positions.append(y)
            
            if len(coordinates) < 2:
                return results
            
            if SCIPY_AVAILABLE:
                try:
                    # Calculate pairwise distances between text blocks
                    distances = pdist(coordinates)
                    results['text_spacing_variance'] = float(np.std(distances))
                    results['mean_text_spacing'] = float(np.mean(distances))
                    
                except Exception as e:
                    logger.debug(f"Text spacing analysis failed: {e}")
                    # Fallback calculation
                    coords_array = np.array(coordinates)
                    center = np.mean(coords_array, axis=0)
                    distances_from_center = np.linalg.norm(coords_array - center, axis=1)
                    results['text_spacing_variance'] = float(np.std(distances_from_center))
                    results['mean_text_spacing'] = float(np.mean(distances_from_center))
            else:
                # Basic fallback calculation
                coords_array = np.array(coordinates)
                center = np.mean(coords_array, axis=0)
                distances_from_center = np.linalg.norm(coords_array - center, axis=1)
                results['text_spacing_variance'] = float(np.std(distances_from_center))
                results['mean_text_spacing'] = float(np.mean(distances_from_center))
            
            # Text alignment analysis
            x_std = np.std(x_positions)
            y_std = np.std(y_positions)
            results['x_alignment_variance'] = float(x_std)
            results['y_alignment_variance'] = float(y_std)
            
            # Normalized alignment score (0=perfect alignment, 1=random scatter)
            image_width, image_height = image_shape[1], image_shape[0]
            results['text_alignment_score'] = float(min(1.0, (x_std + y_std) / (image_width + image_height)))
            
            # Text density uniformity (check for clustering vs even distribution)
            if len(coordinates) > 4:
                # Divide image into quadrants and check text distribution
                quadrant_counts = [0, 0, 0, 0]  # top-left, top-right, bottom-left, bottom-right
                mid_x, mid_y = image_width // 2, image_height // 2
                
                for x, y in coordinates:
                    if x < mid_x and y < mid_y:
                        quadrant_counts[0] += 1
                    elif x >= mid_x and y < mid_y:
                        quadrant_counts[1] += 1
                    elif x < mid_x and y >= mid_y:
                        quadrant_counts[2] += 1
                    else:
                        quadrant_counts[3] += 1
                
                # Calculate uniformity (lower variance = more uniform distribution)
                expected_per_quadrant = len(coordinates) / 4
                uniformity_variance = np.var(quadrant_counts)
                results['text_density_uniformity'] = float(1.0 / (1.0 + uniformity_variance / (expected_per_quadrant + 1)))
            
        except Exception as e:
            logger.debug(f"Text layout analysis failed: {e}")
        
        return results
    
    def morphological_edge_analysis(self, gray_image: np.ndarray, contour: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Advanced edge and morphological analysis for detecting cut-off content
        """
        results = {}
        
        try:
            # Edge detection with adaptive thresholds
            median_val = np.median(gray_image)
            lower_thresh = int(max(0, 0.66 * median_val))
            upper_thresh = int(min(255, 1.33 * median_val))
            
            edges = cv2.Canny(gray_image, lower_thresh, upper_thresh)
            results['edge_density'] = float(np.sum(edges > 0) / edges.size)
            
            if SCIPY_AVAILABLE:
                try:
                    # Connected component analysis
                    labeled_array, num_features = label(edges > 0)
                    results['connected_components'] = int(num_features)
                    
                    # Component size analysis
                    if num_features > 0:
                        component_sizes = []
                        for i in range(1, num_features + 1):
                            component_size = np.sum(labeled_array == i)
                            component_sizes.append(component_size)
                        
                        results['largest_component_ratio'] = float(max(component_sizes) / edges.size)
                        results['component_size_variance'] = float(np.std(component_sizes))
                    else:
                        results['largest_component_ratio'] = 0.0
                        results['component_size_variance'] = 0.0
                    
                except Exception as e:
                    logger.debug(f"Component analysis failed: {e}")
                    results['connected_components'] = 0
                    results['largest_component_ratio'] = 0.0
                    results['component_size_variance'] = 0.0
            else:
                # Basic OpenCV connected components
                try:
                    num_labels, labels = cv2.connectedComponents(edges)
                    results['connected_components'] = int(num_labels - 1)  # Subtract background
                    results['largest_component_ratio'] = 0.0
                    results['component_size_variance'] = 0.0
                except Exception as e:
                    logger.debug(f"OpenCV connected components failed: {e}")
                    results['connected_components'] = 0
                    results['largest_component_ratio'] = 0.0
                    results['component_size_variance'] = 0.0
            
            # Edge proximity to image borders
            h, w = gray_image.shape
            border_thickness = 10
            
            # Count edge pixels near borders
            top_edge_pixels = np.sum(edges[:border_thickness, :] > 0)
            bottom_edge_pixels = np.sum(edges[-border_thickness:, :] > 0)
            left_edge_pixels = np.sum(edges[:, :border_thickness] > 0)
            right_edge_pixels = np.sum(edges[:, -border_thickness:] > 0)
            
            total_border_pixels = 2 * border_thickness * (h + w)
            results['border_edge_density'] = float((top_edge_pixels + bottom_edge_pixels + 
                                                   left_edge_pixels + right_edge_pixels) / total_border_pixels)
            
            # Individual border analysis
            results['top_border_edge_ratio'] = float(top_edge_pixels / (border_thickness * w))
            results['bottom_border_edge_ratio'] = float(bottom_edge_pixels / (border_thickness * w))
            results['left_border_edge_ratio'] = float(left_edge_pixels / (border_thickness * h))
            results['right_border_edge_ratio'] = float(right_edge_pixels / (border_thickness * h))
            
        except Exception as e:
            logger.error(f"Morphological edge analysis failed: {e}")
            # Fallback values
            results.update({
                'edge_density': 0.0,
                'connected_components': 0,
                'largest_component_ratio': 0.0,
                'component_size_variance': 0.0,
                'border_edge_density': 0.0,
                'top_border_edge_ratio': 0.0,
                'bottom_border_edge_ratio': 0.0,
                'left_border_edge_ratio': 0.0,
                'right_border_edge_ratio': 0.0
            })
        
        return results
    
    def statistical_risk_modeling(self, features: Dict[str, float], document_type: str = "document") -> Dict[str, float]:
        """
        Advanced statistical risk modeling using logistic regression principles
        """
        results = {}
        
        try:
            # Feature selection and normalization
            key_features = {
                'blur_risk': 1.0 - features.get('composite_blur_confidence', 0.5),
                'brightness_risk': float(features.get('brightness_issue', False)),
                'contrast_risk': 1.0 - min(1.0, features.get('rms_contrast', 50) / 100.0),
                'edge_risk': min(1.0, features.get('border_edge_density', 0.1) * 10.0),
                'layout_risk': features.get('text_alignment_score', 0.0),
                'spacing_risk': min(1.0, features.get('text_spacing_variance', 100) / 200.0)
            }
            
            # Document-type specific weight adjustments
            weight_adjustments = {
                'certificate': {'blur_risk': 1.5, 'contrast_risk': 1.3},
                'legal': {'blur_risk': 1.4, 'edge_risk': 1.6},
                'financial': {'blur_risk': 1.3, 'layout_risk': 1.4},
                'receipt': {'blur_risk': 0.8, 'brightness_risk': 0.9},
                'note': {'layout_risk': 0.7, 'spacing_risk': 0.8}
            }
            
            adjustments = weight_adjustments.get(document_type, {})
            
            # Apply document-specific adjustments
            adjusted_features = {}
            for feature, value in key_features.items():
                adjustment = adjustments.get(feature, 1.0)
                adjusted_features[feature] = min(1.0, value * adjustment)
            
            if STATSMODELS_AVAILABLE:
                try:
                    # Simple logistic regression simulation
                    feature_array = np.array(list(adjusted_features.values())).reshape(1, -1)
                    
                    # Simulated coefficients (would be learned from data in practice)
                    coefficients = np.array([0.4, 0.3, 0.2, 0.25, 0.15, 0.1])
                    intercept = -1.2
                    
                    # Logistic function
                    linear_combination = np.dot(feature_array, coefficients) + intercept
                    probability = 1 / (1 + np.exp(-linear_combination[0]))
                    
                    results['statistical_risk_probability'] = float(probability)
                    results['statistical_confidence'] = float(1.0 - probability)
                    
                except Exception as e:
                    logger.debug(f"Statistical modeling failed: {e}")
                    # Fallback to weighted average
                    weights = [0.3, 0.25, 0.2, 0.15, 0.1, 0.05]
                    weighted_risk = np.average(list(adjusted_features.values()), weights=weights)
                    results['statistical_risk_probability'] = float(weighted_risk)
                    results['statistical_confidence'] = float(1.0 - weighted_risk)
            else:
                # Weighted average fallback
                weights = [0.3, 0.25, 0.2, 0.15, 0.1, 0.05]
                weighted_risk = np.average(list(adjusted_features.values()), weights=weights)
                results['statistical_risk_probability'] = float(weighted_risk)
                results['statistical_confidence'] = float(1.0 - weighted_risk)
            
            # Feature importance analysis
            results['feature_contributions'] = adjusted_features
            results['dominant_risk_factor'] = max(adjusted_features.keys(), 
                                                 key=lambda k: adjusted_features[k])
            
        except Exception as e:
            logger.error(f"Statistical risk modeling failed: {e}")
            results = {
                'statistical_risk_probability': 0.5,
                'statistical_confidence': 0.5,
                'feature_contributions': {},
                'dominant_risk_factor': 'unknown'
            }
        
        return results
    
    def comprehensive_quality_analysis(self, gray_image: np.ndarray, text_blocks: List = None, 
                                     contour: Optional[np.ndarray] = None, 
                                     document_type: str = "document") -> Dict[str, any]:
        """
        Comprehensive quality analysis combining all advanced techniques
        """
        if text_blocks is None:
            text_blocks = []
            
        all_results = {}
        
        try:
            # 1. Enhanced blur detection
            blur_results = self.enhanced_blur_detection(gray_image)
            all_results.update(blur_results)
            
            # 2. Advanced brightness analysis  
            brightness_results = self.advanced_brightness_analysis(gray_image)
            all_results.update(brightness_results)
            
            # 3. Text layout analysis
            if text_blocks:
                layout_results = self.text_layout_analysis(text_blocks, gray_image.shape)
                all_results.update(layout_results)
            
            # 4. Morphological edge analysis
            edge_results = self.morphological_edge_analysis(gray_image, contour)
            all_results.update(edge_results)
            
            # 5. Statistical risk modeling
            risk_results = self.statistical_risk_modeling(all_results, document_type)
            all_results.update(risk_results)
            
            # 6. Overall quality score
            blur_score = all_results.get('composite_blur_confidence', 0.5)
            brightness_score = 1.0 - float(all_results.get('brightness_issue', False))
            layout_score = 1.0 - all_results.get('text_alignment_score', 0.0)
            edge_score = 1.0 - min(1.0, all_results.get('border_edge_density', 0.1) * 5.0)
            
            overall_quality = np.mean([blur_score, brightness_score, layout_score, edge_score])
            all_results['overall_quality_score'] = float(overall_quality)
            
            # 7. Quality classification
            if overall_quality >= 0.8:
                all_results['quality_class'] = 'excellent'
            elif overall_quality >= 0.6:
                all_results['quality_class'] = 'good'  
            elif overall_quality >= 0.4:
                all_results['quality_class'] = 'fair'
            else:
                all_results['quality_class'] = 'poor'
                
        except Exception as e:
            logger.error(f"Comprehensive quality analysis failed: {e}")
            all_results = {
                'overall_quality_score': 0.5,
                'quality_class': 'unknown',
                'composite_blur_confidence': 0.5,
                'brightness_issue': False,
                'statistical_risk_probability': 0.5
            }
        
        return all_results


# Global instance for easy access
advanced_analyzer = AdvancedQualityAnalyzer()


def get_advanced_quality_features(gray_image: np.ndarray, text_blocks: List = None, 
                                 contour: Optional[np.ndarray] = None,
                                 document_type: str = "document") -> Dict[str, any]:
    """
    Convenience function to get all advanced quality features
    """
    return advanced_analyzer.comprehensive_quality_analysis(
        gray_image, text_blocks, contour, document_type
    )