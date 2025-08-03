# Document Scanner Application
# Complete workflow: Quality Check -> OCR -> Enhanced Key-Value Extraction

import cv2
import numpy as np
from google.cloud import vision
import spacy
from scipy.spatial.distance import cdist
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import asyncio
import json
import base64
from typing import List, Dict, Tuple, Optional, Set
import re
from dataclasses import dataclass
from PIL import Image
import io
import logging
from collections import defaultdict

def ensure_json_serializable(obj):
    """Convert numpy types to native Python types for JSON serialization"""
    if isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64)):
        return int(obj)
    elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: ensure_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [ensure_json_serializable(item) for item in obj]
    else:
        return obj

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class BoundingBox:
    x: int
    y: int
    width: int
    height: int
    
    @property
    def center_x(self) -> float:
        return self.x + self.width / 2
    
    @property
    def center_y(self) -> float:
        return self.y + self.height / 2
    
    @property
    def right(self) -> int:
        return self.x + self.width
    
    @property
    def bottom(self) -> int:
        return self.y + self.height
    
@dataclass
class TextBlock:
    text: str
    bbox: BoundingBox
    confidence: float

@dataclass
class KeyValuePair:
    key: str
    value: str
    key_bbox: BoundingBox
    value_bbox: BoundingBox
    confidence: float
    extraction_method: str

class DocumentQualityChecker:
    """Adaptive document quality assessment using OpenCV"""
    
    def __init__(self):
        # Adaptive thresholds - will be calculated based on image characteristics
        self.min_contour_area_ratio = 0.1  # Minimum document area as ratio of image area
        self.adaptive_blur_threshold = True  # Calculate threshold based on image size and content
        
    def _calculate_adaptive_thresholds(self, image: np.ndarray) -> Dict:
        """Calculate adaptive thresholds based on image characteristics"""
        height, width = image.shape[:2] if len(image.shape) == 3 else image.shape
        image_area = height * width
        
        # Adaptive blur threshold based on image size and content
        # Larger images can tolerate more blur
        base_blur_threshold = 100.0
        size_factor = min(2.0, max(0.5, (image_area / (1024 * 768)) ** 0.5))
        adaptive_blur_threshold = base_blur_threshold * size_factor
        
        # Adaptive brightness thresholds based on image statistics
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
        
        # Ensure reasonable bounds
        dark_threshold = max(30, min(80, dark_threshold))
        bright_threshold = max(180, min(240, bright_threshold))
        
        return {
            'blur_threshold': adaptive_blur_threshold,
            'dark_threshold': dark_threshold,
            'bright_threshold': bright_threshold,
            'size_factor': size_factor
        }

    def assess_quality(self, image: np.ndarray) -> Dict:
        """
        Adaptive document quality assessment with dynamic thresholds
        Returns: {"needs_rescan": bool, "confidence": float, "issues": list}
        """
        issues = []
        confidence = 1.0
        
        try:
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # Calculate adaptive thresholds
            thresholds = self._calculate_adaptive_thresholds(image)
                
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
                confidence -= min(0.2, 0.05 + brightness_severity * 0.15)  # More tolerant of brightness
                
            # 3. Adaptive Document Edge Detection
            try:
                # Use adaptive Canny thresholds based on image statistics
                median_val = np.median(gray)
                lower_thresh = int(max(0, 0.66 * median_val))
                upper_thresh = int(min(255, 1.33 * median_val))
                
                edges = cv2.Canny(gray, lower_thresh, upper_thresh, apertureSize=3)
                contours_result = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                # Handle different OpenCV versions
                if len(contours_result) == 3:
                    _, contours, _ = contours_result  # OpenCV 3.x
                else:
                    contours, _ = contours_result     # OpenCV 4.x
                
                if contours:
                    # Find largest contour (should be document)
                    largest_contour = max(contours, key=cv2.contourArea)
                    contour_area = cv2.contourArea(largest_contour)
                    
                    # Check if document is properly captured (adaptive threshold)
                    image_area = gray.shape[0] * gray.shape[1]
                    area_ratio = contour_area / image_area
                    
                    if area_ratio < self.min_contour_area_ratio:
                        issues.append("document_too_small")
                        confidence -= 0.15
                    
                    # Check for irregular shape (document might be cropped)
                    epsilon = 0.02 * cv2.arcLength(largest_contour, True)
                    approx = cv2.approxPolyDP(largest_contour, epsilon, True)
                    
                    if len(approx) < 4:  # Not rectangular enough
                        issues.append("irregular_shape")
                        confidence -= 0.1  # Reduced penalty
                        
            except Exception as contour_error:
                logger.warning(f"Contour detection failed: {contour_error}")
                # Don't heavily penalize edge detection issues
                
            # 4. Adaptive Skew Detection
            try:
                # Use adaptive Hough threshold based on image content
                hough_threshold = max(50, int(min(gray.shape) * 0.1))
                lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=hough_threshold)
                
                if lines is not None and len(lines) > 0:
                    angles = []
                    for rho, theta in lines[:15]:  # Check more lines for better accuracy
                        angle = theta * 180 / np.pi
                        # Normalize angles to 0-180 range
                        if angle > 90:
                            angle = 180 - angle
                        angles.append(angle)
                    
                    if angles:
                        # Use median instead of mean for better robustness
                        median_angle = np.median(angles)
                        skew_tolerance = 8  # More tolerant threshold
                        
                        if abs(median_angle) > skew_tolerance and abs(median_angle - 90) > skew_tolerance:
                            issues.append("skewed_document")
                            confidence -= 0.05  # Reduced penalty for skew
                            
            except Exception as skew_error:
                logger.warning(f"Skew detection failed: {skew_error}")
                # Don't penalize for skew detection failure
            
            # 5. Text Content Detection (new adaptive check)
            try:
                # Check if there's sufficient text-like content
                text_area_ratio = np.sum(edges > 0) / image_area
                if text_area_ratio < 0.01:  # Very little edge content
                    issues.append("low_content_density")
                    confidence -= 0.1
                    
            except Exception:
                pass  # Non-critical check
                
            # Adaptive decision logic based on context
            issue_weight = len(issues) * 0.1
            confidence_penalty = min(0.3, issue_weight)
            final_confidence = max(0.0, confidence - confidence_penalty)
            
            # More lenient threshold for bright images (common in forms)
            brightness_adjusted_threshold = 0.5 if "too_bright" in issues else 0.6
            needs_rescan = final_confidence < brightness_adjusted_threshold
            
            return {
                "needs_rescan": bool(needs_rescan),
                "confidence": float(final_confidence),
                "issues": issues,
                "blur_score": float(blur_score),
                "brightness": float(mean_brightness),
                "adaptive_thresholds": {
                    'blur_threshold': float(thresholds['blur_threshold']),
                    'dark_threshold': float(thresholds['dark_threshold']),
                    'bright_threshold': float(thresholds['bright_threshold']),
                    'size_factor': float(thresholds['size_factor'])
                },
                "image_stats": {
                    "size_factor": float(thresholds['size_factor']),
                    "area_ratio": float(area_ratio if 'area_ratio' in locals() else 0),
                    "edge_density": float(text_area_ratio if 'text_area_ratio' in locals() else 0)
                }
            }
            
        except Exception as e:
            logger.error(f"Quality assessment error: {e}")
            return {
                "needs_rescan": True,
                "confidence": 0.0,
                "issues": ["processing_error"],
                "blur_score": 0.0,
                "brightness": 0.0,
                "error": str(e)
            }

class GoogleOCRProcessor:
    """Google Cloud Vision API integration"""
    
    def __init__(self):
        try:
            self.client = vision.ImageAnnotatorClient()
        except Exception as e:
            logger.error(f"Failed to initialize Google Cloud Vision client: {e}")
            logger.error("Please set GOOGLE_APPLICATION_CREDENTIALS environment variable")
            logger.error("Example: export GOOGLE_APPLICATION_CREDENTIALS='./credentials/key.json'")
            raise HTTPException(
                status_code=500, 
                detail="Google Cloud Vision API not configured. Please check credentials."
            )
        
    def extract_text_with_bounds(self, image_bytes: bytes) -> List[TextBlock]:
        """Extract text with bounding boxes using Google OCR"""
        try:
            image = vision.Image(content=image_bytes)
            response = self.client.document_text_detection(image=image)
            
            if response.error.message:
                raise Exception(f"OCR Error: {response.error.message}")
                
            text_blocks = []
            
            # Process each page (usually just one for single image)
            for page in response.full_text_annotation.pages:
                for block in page.blocks:
                    block_text = ""
                    
                    # Combine all text in the block
                    for paragraph in block.paragraphs:
                        for word in paragraph.words:
                            word_text = "".join([symbol.text for symbol in word.symbols])
                            block_text += word_text + " "
                    
                    # Get bounding box
                    vertices = block.bounding_box.vertices
                    x_coords = [v.x for v in vertices]
                    y_coords = [v.y for v in vertices]
                    
                    bbox = BoundingBox(
                        x=min(x_coords),
                        y=min(y_coords),
                        width=max(x_coords) - min(x_coords),
                        height=max(y_coords) - min(y_coords)
                    )
                    
                    if block_text.strip():
                        text_blocks.append(TextBlock(
                            text=block_text.strip(),
                            bbox=bbox,
                            confidence=float(block.confidence) if block.confidence else 0.0
                        ))
                        
            return text_blocks
            
        except Exception as e:
            logger.error(f"OCR processing error: {e}")
            raise HTTPException(status_code=500, detail=f"OCR failed: {e}")

class EnhancedKeyValueExtractor:
    """
    Advanced Key-Value extraction optimized for forms and documents:
    1. Form layout detection and specialized processing
    2. Enhanced spatial relationship analysis
    3. Better key vs value distinction
    4. Prevention of reciprocal pairings
    5. Multi-method confidence scoring
    """
    
    def __init__(self):
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("spaCy model not found. Install with: python -m spacy download en_core_web_sm")
            self.nlp = None
        
        # Dynamic pattern learning - patterns discovered from actual content
        self.learned_patterns = {}
        self.pattern_confidence = {}
        
        # Base value characteristics (not hardcoded patterns)
        self.value_characteristics = {
            'has_numbers': lambda text: bool(re.search(r'\d', text)),
            'has_mixed_case': lambda text: text != text.lower() and text != text.upper(),
            'has_punctuation': lambda text: bool(re.search(r'[.,\-/\\@#$%&*()_+=]', text)),
            'is_capitalized': lambda text: text.istitle(),
            'has_spaces': lambda text: ' ' in text,
            'length_short': lambda text: len(text) <= 10,
            'length_medium': lambda text: 10 < len(text) <= 30,
            'length_long': lambda text: len(text) > 30
        }
        
        # Key characteristics (learned, not hardcoded)
        self.key_characteristics = {
            'ends_with_colon': lambda text: text.endswith(':'),
            'ends_with_identifier': lambda text: any(text.endswith(suffix) for suffix in ['No.', 'Number', 'ID', 'Code', 'Name']),
            'short_length': lambda text: 2 <= len(text) <= 25,
            'title_case': lambda text: text.istitle(),
            'all_caps': lambda text: text.isupper() and len(text) > 2,
            'contains_common_words': self._contains_form_words
        }
        
        # Dynamically discovered form field patterns
        self.discovered_field_types = set()
        
        # Statistical tracking for adaptive learning
        self.text_stats = {
            'avg_key_length': 15,
            'avg_value_length': 25,
            'common_separators': {':'},
            'left_aligned_threshold': 150
        }
    
    def _contains_form_words(self, text: str) -> bool:
        """Check if text contains common form field words (adaptive)"""
        text_lower = text.lower().replace(':', '').strip()
        # Common form words that can be learned dynamically
        common_form_words = [
            'name', 'address', 'phone', 'email', 'date', 'birth', 'nationality',
            'id', 'number', 'code', 'first', 'last', 'full', 'street', 'city',
            'state', 'zip', 'country', 'postal', 'contact', 'reference'
        ]
        
        # Check for exact matches and partial matches
        for word in common_form_words:
            if word in text_lower:
                self.discovered_field_types.add(word)
                return True
        return False
    
    def _learn_from_document(self, text_blocks: List[TextBlock]) -> None:
        """Learn patterns and statistics from the current document"""
        all_text = [block.text for block in text_blocks]
        
        # Update length statistics
        lengths = [len(text) for text in all_text]
        if lengths:
            avg_length = sum(lengths) / len(lengths)
            # Adaptive threshold learning
            self.text_stats['avg_text_length'] = avg_length
            
        # Learn common patterns from actual content
        for text in all_text:
            # Discover value patterns
            if re.search(r'\d', text):  # Contains numbers
                if '@' in text and '.' in text:
                    self._update_pattern('email', text)
                elif any(word in text.lower() for word in ['street', 'ave', 'blvd', 'road']):
                    self._update_pattern('address', text)
                elif len(text.split()) == 1 and text.replace('-', '').replace(' ', '').isdigit():
                    self._update_pattern('id_number', text)
            
            # Discover key patterns
            if text.endswith(':'):
                self._update_pattern('label_with_colon', text)
                
    def _update_pattern(self, pattern_type: str, example: str) -> None:
        """Update learned patterns with new examples"""
        if pattern_type not in self.learned_patterns:
            self.learned_patterns[pattern_type] = []
            self.pattern_confidence[pattern_type] = 0.1
        
        self.learned_patterns[pattern_type].append(example)
        # Increase confidence as we see more examples
        self.pattern_confidence[pattern_type] = min(0.9, 
                                                   self.pattern_confidence[pattern_type] + 0.1)
        
    def detect_form_layout(self, text_blocks: List[TextBlock]) -> Dict[str, any]:
        """
        Adaptive form layout detection using learned patterns
        """
        # First, learn from this document
        self._learn_from_document(text_blocks)
        
        form_indicators = 0
        total_blocks = len(text_blocks)
        label_like_blocks = 0
        value_like_blocks = 0
        
        # Adaptive form detection using characteristics
        for block in text_blocks:
            text = block.text.strip()
            
            # Count key characteristics
            key_score = 0
            for char_name, char_func in self.key_characteristics.items():
                if char_func(text):
                    key_score += 1
            
            # Count value characteristics  
            value_score = 0
            for char_name, char_func in self.value_characteristics.items():
                if char_func(text):
                    value_score += 1
            
            # Determine if this looks like a label or value
            if key_score >= 2:  # Looks like a label
                label_like_blocks += 1
                form_indicators += 1
            elif value_score >= 2:  # Looks like a value
                value_like_blocks += 1
                form_indicators += 0.5
                
            # Bonus for discovering form field words
            if self._contains_form_words(text):
                form_indicators += 1
        
        # Adaptive threshold based on discovered patterns
        base_threshold = 0.3
        if len(self.discovered_field_types) > 2:  # Found multiple field types
            base_threshold = 0.2  # More confident it's a form
            
        is_form = form_indicators / total_blocks > base_threshold if total_blocks > 0 else False
        
        # Analyze spatial distribution for form layout
        y_positions = [block.bbox.y for block in text_blocks]
        x_positions = [block.bbox.x for block in text_blocks]
        
        # Check for regular spacing (forms often have consistent spacing)
        y_diffs = [abs(y_positions[i] - y_positions[i-1]) for i in range(1, len(y_positions))]
        regular_spacing = len(set([d for d in y_diffs if 20 <= d <= 100])) <= 3 if y_diffs else False
        
        # Check for left-aligned structure
        left_aligned_count = sum(1 for x in x_positions if x < 100)
        left_aligned = left_aligned_count / len(x_positions) > 0.5 if x_positions else False
        
        return {
            'is_form': bool(is_form),
            'form_confidence': float(min(1.0, form_indicators / max(1, total_blocks))),
            'regular_spacing': bool(regular_spacing),
            'left_aligned': bool(left_aligned),
            'form_score': float(form_indicators)
        }

    def analyze_text_semantics(self, text_blocks: List[TextBlock]) -> Dict[str, any]:
        """Use NLP to understand document structure and semantics"""
        if not self.nlp:
            return {}
        
        full_text = " ".join([block.text for block in text_blocks])
        doc = self.nlp(full_text)
        
        # Extract named entities and their types
        entities = {}
        for ent in doc.ents:
            entities[ent.text] = ent.label_
        
        # Identify potential keys based on POS tags and dependency parsing
        potential_keys = []
        for token in doc:
            # Look for tokens that might be keys (nouns followed by colons, etc.)
            if token.pos_ in ['NOUN', 'PROPN'] and token.dep_ in ['compound', 'nmod']:
                potential_keys.append(token.text)
        
        return {
            'entities': entities,
            'potential_keys': potential_keys,
            'tokens': [(token.text, token.pos_, token.dep_) for token in doc]
        }
    
    def identify_potential_keys(self, text_blocks: List[TextBlock], form_analysis: Dict = None) -> List[Tuple[TextBlock, float, str]]:
        """
        Enhanced key identification with form-specific logic:
        1. Structural indicators (colons, formatting)
        2. Form field recognition
        3. Semantic analysis
        4. Position and layout analysis
        """
        key_candidates = []
        is_form = form_analysis and form_analysis.get('is_form', False) if form_analysis else False
        
        for block in text_blocks:
            text = block.text.strip()
            text_lower = text.lower()
            confidence = 0.0
            methods = []
            
            # Skip if it's clearly a value (stronger filtering)
            if self._is_definitely_value(text):
                logger.debug(f"Skipping '{text}' - identified as definitely a value")
                continue
            
            # Method 1: Adaptive form field recognition (highest priority for forms)
            if is_form:
                if self._contains_form_words(text):
                    confidence += 0.6
                    methods.append('discovered_form_field')
                    
                    # Bonus for learned patterns
                    for pattern_type, pattern_confidence in self.pattern_confidence.items():
                        if pattern_confidence > 0.3 and pattern_type in self.learned_patterns:
                            for example in self.learned_patterns[pattern_type]:
                                if self._similar_pattern(text, example):
                                    confidence += 0.1
                                    methods.append(f'learned_{pattern_type}')
                                    break
            
            # Method 2: Structural indicators
            if text.endswith(':'):
                confidence += 0.5 if is_form else 0.4
                methods.append('colon_terminator')
            
            if text.endswith(('No.', 'Number', 'ID', 'Code', '#', 'Name')):
                confidence += 0.4
                methods.append('suffix_indicator')
            
            if text.startswith(('#', 'Ref:', 'ID:', 'Name:')):
                confidence += 0.4
                methods.append('prefix_indicator')
            
            # Method 3: Length and word count heuristics (refined)
            words = text.split()
            if 1 <= len(words) <= 3 and 3 <= len(text) <= 40:
                confidence += 0.3
                methods.append('label_length')
            
            # Method 4: Capitalization patterns (refined for forms)
            if text.istitle() and len(words) <= 3:
                confidence += 0.2
                methods.append('title_case')
            elif text.isupper() and len(text) > 2 and len(text) <= 20:
                confidence += 0.1
                methods.append('upper_case')
            
            # Method 5: Position-based analysis (enhanced for forms)
            if is_form and form_analysis.get('left_aligned', False):
                if block.bbox.x < 150:  # Strong left alignment in forms
                    confidence += 0.3
                    methods.append('form_left_position')
            elif block.bbox.x < 200:  # General left positioning
                confidence += 0.1
                methods.append('left_position')
            
            # Method 6: NLP-based semantic analysis
            if self.nlp:
                doc = self.nlp(text)
                for token in doc:
                    if token.pos_ in ['NOUN', 'PROPN'] and not token.ent_type_ in ['PERSON', 'ORG']:
                        confidence += 0.15
                        methods.append('semantic_label')
                        break
            
            # Method 7: Anti-value patterns (prevent values from being keys)
            if not self._looks_like_value(text):
                confidence += 0.1
                methods.append('not_value')
            
            # Method 8: Form-specific bonuses
            if is_form:
                # Bonus for consistent formatting with other labels
                if any(other_block.text.endswith(':') for other_block in text_blocks if other_block != block):
                    if text.endswith(':'):
                        confidence += 0.1
                        methods.append('consistent_formatting')
            
            # Adjust threshold based on context (made more lenient)
            min_threshold = 0.3 if is_form else 0.25
            if confidence >= min_threshold:
                key_candidates.append((block, confidence, '+'.join(methods)))
                logger.debug(f"Key candidate: '{text}' (confidence: {confidence:.2f}, methods: {'+'.join(methods)})")
        
        return key_candidates
    
    def _is_definitely_value(self, text: str) -> bool:
        """Adaptive filter to identify text that is definitely a value, not a key"""
        text = text.strip()
        
        # Use learned characteristics instead of hardcoded patterns
        value_indicators = 0
        
        for char_name, char_func in self.value_characteristics.items():
            if char_func(text):
                value_indicators += 1
        
        # Strong value indicators (made less strict)
        if value_indicators >= 5:  # Multiple value characteristics
            return True
            
        # Specific high-confidence value patterns
        if '@' in text and '.' in text and len(text) > 5:  # Email-like
            return True
            
        # Pure numbers (but not short ones that might be labels)
        if text.replace('.', '').replace(',', '').isdigit() and len(text) > 2:
            return True
            
        # Names (capitalized single/double words, not ending with label suffixes)
        if (text.istitle() and len(text.split()) <= 2 and len(text) > 3 and 
            not text.endswith((':', 'No.', 'ID', 'Code', 'Name', 'Number'))):
            return True
            
        # Long descriptive text (likely content, not labels)
        if len(text) > 40 and not text.endswith(':'):
            return True
            
        return False
    
    def _looks_like_value(self, text: str) -> bool:
        """Check if text looks like a typical value rather than a key using adaptive characteristics"""
        text = text.strip()
        
        # Count value characteristics
        value_score = 0
        for char_name, char_func in self.value_characteristics.items():
            if char_func(text):
                value_score += 1
        
        # If it has multiple value characteristics, likely a value
        if value_score >= 3:
            return True
        
        # Check against learned patterns
        for pattern_type, confidence in self.pattern_confidence.items():
            if pattern_type in self.learned_patterns and confidence > 0.5:
                # Check similarity to learned examples
                for example in self.learned_patterns[pattern_type]:
                    if self._similar_pattern(text, example):
                        return True
        
        # Very long text is usually a value/content
        if len(text) > 50:
            return True
        
        # Check if it's a single word that's capitalized (likely a name/value)
        if len(text.split()) == 1 and text.istitle() and len(text) > 3:
            return True
            
        return False
    
    def _similar_pattern(self, text1: str, text2: str) -> bool:
        """Check if two texts have similar patterns (length, case, content type)"""
        # Simple similarity check based on characteristics
        len_diff = abs(len(text1) - len(text2)) / max(len(text1), len(text2))
        if len_diff > 0.5:  # Very different lengths
            return False
            
        # Check case patterns
        case_match = (text1.isupper() == text2.isupper() and 
                     text1.islower() == text2.islower() and
                     text1.istitle() == text2.istitle())
        
        # Check number presence
        num_match = bool(re.search(r'\d', text1)) == bool(re.search(r'\d', text2))
        
        return case_match and num_match
    
    def find_spatial_values(self, key_block: TextBlock, all_blocks: List[TextBlock]) -> List[Tuple[TextBlock, float, str]]:
        """
        Find potential values using advanced spatial analysis
        """
        value_candidates = []
        key_center_x = key_block.bbox.center_x
        key_center_y = key_block.bbox.center_y
        key_right = key_block.bbox.right
        
        for block in all_blocks:
            if block == key_block:
                continue
            
            confidence = 0.0
            methods = []
            
            block_center_x = block.bbox.center_x
            block_center_y = block.bbox.center_y
            block_left = block.bbox.x
            
            # Spatial relationship analysis
            horizontal_distance = abs(key_center_x - block_center_x)
            vertical_distance = abs(key_center_y - block_center_y)
            
            # Method 1: Same line (horizontal alignment)
            if vertical_distance < key_block.bbox.height * 0.8:
                confidence += 0.4
                methods.append('same_line')
                
                # Bonus for being to the right of key
                if block_left >= key_right - 10:  # Small overlap tolerance
                    confidence += 0.3
                    methods.append('right_of_key')
            
            # Method 2: Directly below key (common in forms)
            elif (vertical_distance < key_block.bbox.height * 2 and 
                  horizontal_distance < key_block.bbox.width * 1.5):
                confidence += 0.3
                methods.append('below_key')
            
            # Method 3: Proximity scoring (closer is better)
            total_distance = np.sqrt(horizontal_distance**2 + vertical_distance**2)
            if total_distance < 300:  # Within reasonable distance
                proximity_score = max(0, (300 - total_distance) / 300) * 0.2
                confidence += proximity_score
                methods.append('proximity')
            
            # Method 4: Value pattern matching
            value_type = self._classify_value_type(block.text)
            if value_type:
                confidence += 0.2
                methods.append(f'pattern_{value_type}')
            
            # Method 5: Size and formatting consistency
            if self._has_value_formatting(block.text):
                confidence += 0.1
                methods.append('value_formatting')
            
            if confidence >= 0.3:  # Minimum threshold
                value_candidates.append((block, confidence, '+'.join(methods)))
        
        return value_candidates
    
    def _classify_value_type(self, text: str) -> Optional[str]:
        """Classify the type of value based on patterns"""
        text = text.strip()
        
        for pattern_type, pattern in self.value_patterns.items():
            if re.search(pattern, text, re.IGNORECASE):
                return pattern_type
        return None
    
    def _has_value_formatting(self, text: str) -> bool:
        """Check if text has typical value formatting"""
        text = text.strip()
        
        # Contains numbers
        if re.search(r'\d', text):
            return True
        
        # Proper case or all caps (like names, addresses)
        if text.istitle() or (text.isupper() and len(text) > 3):
            return True
        
        # Contains special characters typical in values
        if re.search(r'[.,\-/\\@]', text):
            return True
            
        return False
    
    def cluster_key_value_pairs(self, key_candidates: List[Tuple[TextBlock, float, str]], 
                               all_blocks: List[TextBlock], form_analysis: Dict = None) -> List[KeyValuePair]:
        """
        Enhanced key-value pairing with reciprocal prevention and better scoring
        """
        kv_pairs = []
        used_value_blocks = set()
        used_key_blocks = set()
        is_form = form_analysis and form_analysis.get('is_form', False) if form_analysis else False
        
        # Sort key candidates by confidence
        key_candidates.sort(key=lambda x: x[1], reverse=True)
        
        for key_block, key_confidence, key_method in key_candidates:
            # Skip if this block is already used as a value
            if id(key_block) in used_value_blocks:
                continue
                
            # Find potential values for this key
            value_candidates = self.find_spatial_values(key_block, all_blocks)
            
            if not value_candidates:
                continue
            
            # Filter out already used blocks and prevent reciprocal pairings
            available_candidates = []
            for block, conf, method in value_candidates:
                if (id(block) not in used_value_blocks and 
                    id(block) not in used_key_blocks and
                    block != key_block):  # Prevent self-pairing
                    
                    # Additional check: prevent reciprocal pairing
                    # (don't pair A->B if B->A already exists)
                    is_reciprocal = any(
                        existing_pair.value_bbox == key_block.bbox and 
                        existing_pair.key_bbox == block.bbox 
                        for existing_pair in kv_pairs
                    )
                    
                    if not is_reciprocal:
                        # Form-specific value validation
                        if is_form:
                            # Ensure the value makes sense for the key
                            if self._validate_form_pair(key_block.text, block.text):
                                available_candidates.append((block, conf, method))
                        else:
                            available_candidates.append((block, conf, method))
            
            if not available_candidates:
                continue
            
            # Enhanced scoring for form layouts
            if is_form:
                available_candidates = self._score_form_pairs(key_block, available_candidates)
            
            # Sort by confidence and take the best match
            available_candidates.sort(key=lambda x: x[1], reverse=True)
            best_value_block, value_confidence, value_method = available_candidates[0]
            
            # Enhanced confidence calculation
            combined_confidence = self._calculate_enhanced_confidence(
                key_block, best_value_block, key_confidence, value_confidence, is_form
            )
            
            # Adjust threshold based on context (made more lenient)
            min_threshold = 0.4 if is_form else 0.35
            
            if combined_confidence >= min_threshold:
                kv_pair = KeyValuePair(
                    key=key_block.text.strip(),
                    value=best_value_block.text.strip(),
                    key_bbox=key_block.bbox,
                    value_bbox=best_value_block.bbox,
                    confidence=combined_confidence,
                    extraction_method=f"key:{key_method}|value:{value_method}"
                )
                
                kv_pairs.append(kv_pair)
                used_value_blocks.add(id(best_value_block))
                used_key_blocks.add(id(key_block))
        
        return kv_pairs
    
    def _validate_form_pair(self, key_text: str, value_text: str) -> bool:
        """Adaptive validation for key-value pairs using learned characteristics"""
        key_lower = key_text.lower().strip(':').strip()
        value_lower = value_text.lower().strip()
        
        # Use discovered form field types for validation
        discovered_fields = list(self.discovered_field_types)
        
        # If key contains a discovered field type, validate appropriately
        key_field_type = None
        for field_type in discovered_fields:
            if field_type in key_lower:
                key_field_type = field_type
                break
        
        if key_field_type:
            # Value shouldn't contain field type names (prevent field name as value)
            if any(field in value_lower for field in discovered_fields):
                return False
            
            # Specific validations based on discovered field types
            if key_field_type in ['name', 'first', 'last']:
                # Names shouldn't look like addresses or IDs
                if (any(word in value_lower for word in ['street', 'avenue', 'road', 'blvd']) or
                    value_text.replace('-', '').replace(' ', '').isdigit()):
                    return False
            
            elif key_field_type == 'address':
                # Addresses shouldn't be simple single words (likely names)
                if (len(value_text.split()) == 1 and value_text.istitle() and 
                    not any(char.isdigit() for char in value_text)):
                    return False
        
        # General validation using characteristics
        key_chars = sum(1 for func in self.key_characteristics.values() if func(key_text))
        value_chars = sum(1 for func in self.value_characteristics.values() if func(value_text))
        
        # Key should look more like a key than a value, and vice versa
        if key_chars <= value_chars - 2:  # Value looks more like a key
            return False
        
        return True
    
    def _score_form_pairs(self, key_block: TextBlock, value_candidates: List[Tuple[TextBlock, float, str]]) -> List[Tuple[TextBlock, float, str]]:
        """Enhanced scoring for form-specific layouts"""
        enhanced_candidates = []
        
        for value_block, confidence, method in value_candidates:
            enhanced_confidence = confidence
            
            # Form-specific spatial bonuses
            key_center_y = key_block.bbox.center_y
            value_center_y = value_block.bbox.center_y
            vertical_distance = abs(key_center_y - value_center_y)
            
            # Same line bonus (stronger for forms)
            if vertical_distance < key_block.bbox.height * 0.8:
                enhanced_confidence += 0.2
                
                # Right-of-key bonus (typical form layout)
                if value_block.bbox.x > key_block.bbox.right - 10:
                    enhanced_confidence += 0.3
            
            # Adaptive value type matching bonuses
            key_text = key_block.text.lower()
            value_text = value_block.text
            
            # Use discovered field types for intelligent matching
            for field_type in self.discovered_field_types:
                if field_type in key_text:
                    # Check if value characteristics match the field type
                    if field_type in ['name', 'first', 'last']:
                        if (value_text.istitle() and len(value_text.split()) <= 3 and 
                            not any(char.isdigit() for char in value_text)):
                            enhanced_confidence += 0.2
                    elif field_type == 'address':
                        if (len(value_text) > 10 and 
                            any(char.isdigit() for char in value_text) and
                            any(word in value_text.lower() for word in ['street', 'ave', 'blvd', 'road', 'way', 'dr', 'ct'])):
                            enhanced_confidence += 0.2
                    elif field_type in ['phone', 'number']:
                        if any(char.isdigit() for char in value_text):
                            enhanced_confidence += 0.15
                    elif field_type in ['email']:
                        if '@' in value_text and '.' in value_text:
                            enhanced_confidence += 0.25
            
            enhanced_candidates.append((value_block, enhanced_confidence, method))
        
        return enhanced_candidates
    
    def _calculate_enhanced_confidence(self, key_block: TextBlock, value_block: TextBlock, 
                                     key_confidence: float, value_confidence: float, is_form: bool) -> float:
        """Calculate enhanced confidence score considering multiple factors"""
        
        # Base confidence from key and value scores
        base_confidence = (key_confidence + value_confidence) / 2
        
        # Spatial relationship bonus
        spatial_bonus = 0.0
        key_center_y = key_block.bbox.center_y
        value_center_y = value_block.bbox.center_y
        vertical_distance = abs(key_center_y - value_center_y)
        
        # Perfect horizontal alignment
        if vertical_distance < key_block.bbox.height * 0.5:
            spatial_bonus += 0.15
            
            # Additional bonus for proper left-to-right ordering
            if value_block.bbox.x > key_block.bbox.right:
                spatial_bonus += 0.1
        
        # Adaptive text compatibility bonus
        compatibility_bonus = 0.0
        if is_form:
            key_text = key_block.text.lower().strip(':').strip()
            value_text = value_block.text.strip()
            
            # Use discovered field types for semantic compatibility
            for field_type in self.discovered_field_types:
                if field_type in key_text:
                    if field_type in ['name', 'first', 'last']:
                        # Names should be proper case, short, no numbers
                        if (value_text.istitle() and len(value_text.split()) <= 3 and 
                            not any(char.isdigit() for char in value_text)):
                            compatibility_bonus += 0.15
                    elif field_type == 'address':
                        # Addresses should be longer, contain numbers and address words
                        if (len(value_text) > 15 and 
                            any(char.isdigit() for char in value_text) and
                            any(word in value_text.lower() for word in ['street', 'ave', 'blvd', 'road', 'way', 'dr', 'ct'])):
                            compatibility_bonus += 0.15
                    elif field_type == 'nationality':
                        # Countries should be proper case, no numbers
                        if (value_text.istitle() and not any(char.isdigit() for char in value_text) and 
                            len(value_text.split()) <= 3):
                            compatibility_bonus += 0.1
        
        # Length compatibility (keys should be shorter than values typically)
        if len(key_block.text) < len(value_block.text):
            compatibility_bonus += 0.05
        
        final_confidence = min(1.0, base_confidence + spatial_bonus + compatibility_bonus)
        return final_confidence
    
    def extract_key_value_pairs(self, text_blocks: List[TextBlock]) -> List[KeyValuePair]:
        """
        Main extraction method using the enhanced approach with form detection
        """
        if not text_blocks:
            return []
        
        try:
            # Step 1: Detect form layout
            form_analysis = self.detect_form_layout(text_blocks)
            logger.info(f"Form detection: {form_analysis}")
            
            # Step 2: Identify potential keys with form context
            key_candidates = self.identify_potential_keys(text_blocks, form_analysis)
            
            if not key_candidates:
                logger.warning("No key candidates found")
                return []
            
            logger.info(f"Found {len(key_candidates)} key candidates")
            for i, (block, conf, method) in enumerate(key_candidates):
                logger.debug(f"Key candidate {i+1}: '{block.text}' (confidence: {conf:.2f}, method: {method})")
            
            # Step 3: Use enhanced clustering approach
            kv_pairs = self.cluster_key_value_pairs(key_candidates, text_blocks, form_analysis)
            logger.info(f"Clustering produced {len(kv_pairs)} initial pairs")
            
            # Step 4: Post-processing and validation
            validated_pairs = self._validate_and_clean_pairs(kv_pairs)
            
            logger.info(f"Extracted {len(validated_pairs)} key-value pairs using enhanced method")
            
            return validated_pairs
            
        except Exception as e:
            logger.error(f"Key-value extraction error: {e}")
            return []
    
    def _validate_and_clean_pairs(self, kv_pairs: List[KeyValuePair]) -> List[KeyValuePair]:
        """Validate and clean extracted pairs"""
        validated_pairs = []
        
        for pair in kv_pairs:
            # Remove common artifacts
            key = pair.key.strip(':').strip()
            value = pair.value.strip()
            
            # Skip if key or value is too short or looks invalid
            if len(key) < 2 or len(value) < 1:
                continue
            
            # Skip if key and value are identical
            if key.lower() == value.lower():
                continue
            
            # Create cleaned pair
            cleaned_pair = KeyValuePair(
                key=key,
                value=value,
                key_bbox=pair.key_bbox,
                value_bbox=pair.value_bbox,
                confidence=pair.confidence,
                extraction_method=pair.extraction_method
            )
            
            validated_pairs.append(cleaned_pair)
        
        return validated_pairs

class DocumentProcessor:
    """Main document processing orchestrator with enhanced key-value extraction"""
    
    def __init__(self):
        self.quality_checker = DocumentQualityChecker()
        self.ocr_processor = GoogleOCRProcessor()
        self.kv_extractor = EnhancedKeyValueExtractor()  # Updated extractor
        
    def process_image_bytes(self, image_bytes: bytes) -> Dict:
        """Process image from bytes"""
        try:
            # Convert bytes to OpenCV image
            nparr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                raise ValueError("Invalid image format")
                
            return self.process_image(image, image_bytes)
            
        except Exception as e:
            logger.error(f"Image processing error: {e}")
            raise HTTPException(status_code=400, detail=f"Image processing failed: {e}")
    
    def process_image(self, image: np.ndarray, image_bytes: bytes) -> Dict:
        """Complete processing pipeline with enhanced extraction"""
        start_time = asyncio.get_event_loop().time() if asyncio.get_event_loop().is_running() else 0
        
        # Step 1: Quality Assessment
        quality_result = self.quality_checker.assess_quality(image)
        
        if quality_result["needs_rescan"]:
            return {
                "status": "rescan_needed",
                "quality_assessment": quality_result,
                "message": "Document quality is insufficient. Please rescan."
            }
            
        # Step 2: OCR Processing
        try:
            text_blocks = self.ocr_processor.extract_text_with_bounds(image_bytes)
        except Exception as e:
            return {
                "status": "ocr_failed",
                "error": str(e),
                "quality_assessment": quality_result
            }
            
        # Step 3: Enhanced Key-Value Extraction
        kv_pairs = self.kv_extractor.extract_key_value_pairs(text_blocks)
        
        # Prepare enhanced response
        processing_time = (asyncio.get_event_loop().time() - start_time) if start_time else 0
        
        return {
            "status": "success",
            "quality_assessment": quality_result,
            "text_blocks_count": len(text_blocks),
            "key_value_pairs": [
                {
                    "key": kv.key,
                    "value": kv.value,
                    "key_bbox": {
                        "x": kv.key_bbox.x,
                        "y": kv.key_bbox.y,
                        "width": kv.key_bbox.width,
                        "height": kv.key_bbox.height
                    },
                    "value_bbox": {
                        "x": kv.value_bbox.x,
                        "y": kv.value_bbox.y,
                        "width": kv.value_bbox.width,
                        "height": kv.value_bbox.height
                    },
                    "confidence": float(kv.confidence),
                    "extraction_method": kv.extraction_method  # New field showing how it was extracted
                }
                for kv in kv_pairs
            ],
            "processing_time_seconds": processing_time,
            "extraction_statistics": {
                "total_text_blocks": int(len(text_blocks)),
                "identified_key_value_pairs": int(len(kv_pairs)),
                "average_confidence": float(sum(kv.confidence for kv in kv_pairs) / len(kv_pairs)) if kv_pairs else 0.0,
                "extraction_methods_used": list(set(kv.extraction_method for kv in kv_pairs))
            },
            "extracted_text_blocks": [
                {
                    "text": block.text,
                    "bbox": {
                        "x": block.bbox.x,
                        "y": block.bbox.y,
                        "width": block.bbox.width,
                        "height": block.bbox.height
                    },
                    "confidence": float(block.confidence)
                }
                for block in text_blocks
            ]
        }

# FastAPI Application
app = FastAPI(title="Document Scanner API", version="2.0.0", description="Enhanced Document Scanner with Intelligent Key-Value Extraction")
processor = DocumentProcessor()

@app.post("/scan-document")
async def scan_document(file: UploadFile = File(...)):
    """
    Main endpoint for document scanning with enhanced key-value extraction
    Accepts image files and returns intelligently extracted key-value pairs with bounding boxes
    """
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
            
        # Read file content
        image_bytes = await file.read()
        
        if len(image_bytes) == 0:
            raise HTTPException(status_code=400, detail="Empty file")
            
        # Process document
        result = processor.process_image_bytes(image_bytes)
        
        # Ensure all data is JSON serializable
        serializable_result = ensure_json_serializable(result)
        
        logger.debug(f"Returning serializable result with status: {serializable_result.get('status')}")
        
        return JSONResponse(content=serializable_result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {e}")

@app.post("/quality-check")
async def quality_check_only(file: UploadFile = File(...)):
    """
    Endpoint for quick quality assessment only
    """
    try:
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
            
        image_bytes = await file.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image format")
            
        quality_result = processor.quality_checker.assess_quality(image)
        
        response = {
            "status": "quality_checked",
            "quality_assessment": quality_result
        }
        
        # Ensure all data is JSON serializable
        serializable_response = ensure_json_serializable(response)
        
        return JSONResponse(content=serializable_response)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Quality check error: {e}")
        raise HTTPException(status_code=500, detail=f"Quality check failed: {e}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "enhanced-document-scanner", "version": "2.0.0"}

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "service": "Enhanced Document Scanner API",
        "version": "2.0.0",
        "description": "Intelligent document processing with advanced key-value extraction",
        "features": [
            "OpenCV-based quality assessment",
            "Google Cloud Vision OCR",
            "Intelligent key-value extraction (no hardcoded patterns)",
            "Spatial analysis and NLP-based processing",
            "Multi-method confidence scoring"
        ],
        "endpoints": {
            "/scan-document": "POST - Complete document processing with enhanced extraction",
            "/quality-check": "POST - Quality assessment only",
            "/health": "GET - Health check"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
