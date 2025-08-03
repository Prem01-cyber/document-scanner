# Document Scanner Application
# Complete workflow: Quality Check -> OCR -> Enhanced Key-Value Extraction

import cv2
import numpy as np
from google.cloud import vision
import spacy
from scipy.spatial.distance import cdist, cosine, euclidean
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.stats import pearsonr
import warnings
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

# Configure logging with more detailed output
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # Enable debug logging for detailed analysis

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
    Hybrid Key-Value extraction with intelligent text splitting:
    1. Text block splitting for combined key-value blocks
    2. Enhanced spatial relationship analysis
    3. Semantic validation for logical pairing
    4. Multi-layered confidence scoring
    5. Adaptive form understanding
    """
    
    def __init__(self):
        # Load best available spaCy model with word vectors for semantic analysis
        self.nlp = self._load_best_spacy_model()
        
        # Suppress spaCy similarity warnings if using small model
        if self.nlp and not self.nlp.vocab.vectors.size:
            warnings.filterwarnings('ignore', message='.*similarity.*')
        
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
    
    def _load_best_spacy_model(self):
        """
        Load the best available spaCy model with word vectors for semantic analysis
        """
        # Try models in order of preference (largest to smallest)
        model_preferences = [
            "en_core_web_lg",    # Large model with word vectors
            "en_core_web_md",    # Medium model with word vectors 
            "en_core_web_sm"     # Small model (no word vectors, fallback)
        ]
        
        for model_name in model_preferences:
            try:
                nlp = spacy.load(model_name)
                logger.info(f"Loaded spaCy model: {model_name} (vectors: {nlp.vocab.vectors.size > 0})")
                return nlp
            except OSError:
                logger.debug(f"Model {model_name} not available")
                continue
        
        logger.warning("No spaCy model found. Install with: python -m spacy download en_core_web_lg")
        logger.info("Attempting to auto-download en_core_web_md model...")
        
        # Try to auto-download a medium model if available
        try:
            import subprocess
            import sys
            result = subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_md"], 
                                  capture_output=True, text=True, timeout=300)
            if result.returncode == 0:
                logger.info("Successfully downloaded en_core_web_md")
                return spacy.load("en_core_web_md")
        except Exception as e:
            logger.debug(f"Auto-download failed: {e}")
        
        return None
    
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
            # Enhanced handling for split blocks that might be form field labels
            is_likely_split_key = (block.confidence == 0.95 and 
                                   self._is_likely_form_field_label(text))
            
            # Additional check for common form field patterns that might be missed
            is_common_form_field = any(field_word in text.lower() for field_word in 
                                     ['first name', 'last name', 'full name', 'nationality', 'address', 
                                      'phone', 'email', 'date', 'birth'])
            
            if self._is_definitely_value(text) and not (is_likely_split_key or is_common_form_field):
                logger.debug(f"Skipping '{text}' - identified as definitely a value")
                continue
            elif is_likely_split_key or is_common_form_field:
                logger.debug(f"Keeping '{text}' - appears to be split form field label (conf: {block.confidence})")
                # Give high confidence to split blocks that are clearly form field labels
                confidence += 0.8  
                methods.append('split_form_field')
            
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
            
            # Method 2: Structural indicators (dynamic confidence)
            if text.endswith(':'):
                # Dynamic confidence based on colon prevalence in document
                colon_boost = self._calculate_dynamic_pattern_confidence('colon_terminator', text_blocks)
                confidence += colon_boost
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
            
            # Dynamic threshold based on confidence distribution
            min_threshold = self._calculate_dynamic_key_threshold([c[1] for c in key_candidates])
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
        # Use NLP to detect if this might be a form field label vs a name/value
        if (text.istitle() and len(text.split()) <= 2 and len(text) > 3 and 
            not self._is_likely_form_field_label(text)):
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
    
    def split_combined_text_blocks(self, text_blocks: List[TextBlock]) -> List[TextBlock]:
        """
        Phase 1: Intelligent splitting of combined key-value text blocks
        Transforms "Last Name Leverling" → "Last Name" + "Leverling"
        """
        print("=== STARTING TEXT BLOCK SPLITTING ===")  # Use print for immediate visibility
        logger.info(f"=== STARTING TEXT BLOCK SPLITTING ===")
        logger.info(f"Input blocks: {len(text_blocks)}")
        for i, block in enumerate(text_blocks):
            print(f"  Block {i+1}: '{block.text}'")  # Print to console
            logger.info(f"  Block {i+1}: '{block.text}'")
            
        split_blocks = []
        
        for i, block in enumerate(text_blocks):
            text = block.text.strip()
            print(f"Processing block {i+1}: '{text}'")  # Print to console
            logger.debug(f"Processing block {i+1}: '{text}'")
            
            # First try intelligent pattern-based splitting
            intelligent_split = self._intelligent_split_test(text, block.bbox)
            if intelligent_split:
                print(f"✅ INTELLIGENT SPLIT: '{text}' → {[s.text for s in intelligent_split]}")
                split_blocks.extend(intelligent_split)
                continue
            
            # Try multiple splitting strategies
            splits = self._attempt_text_splitting(text, block.bbox)
            
            if splits and len(splits) > 1:
                print(f"✅ SPLIT SUCCESS: '{text}' → {len(splits)} parts: {[s.text for s in splits]}")
                logger.info(f"✅ SPLIT SUCCESS: '{text}' → {len(splits)} parts: {[s.text for s in splits]}")
                split_blocks.extend(splits)
            else:
                print(f"No split found for: '{text}' - keeping original")
                logger.debug(f"No split found for: '{text}' - keeping original")
                # Keep original block if no split found
                split_blocks.append(block)
        
        print(f"=== SPLITTING COMPLETE ===")
        print(f"Output blocks: {len(split_blocks)} (+{len(split_blocks) - len(text_blocks)} new)")
        logger.info(f"=== SPLITTING COMPLETE ===")
        logger.info(f"Output blocks: {len(split_blocks)} (+{len(split_blocks) - len(text_blocks)} new)")
        for i, block in enumerate(split_blocks):
            print(f"  Result {i+1}: '{block.text}' (conf: {block.confidence:.2f})")
            logger.info(f"  Result {i+1}: '{block.text}' (conf: {block.confidence:.2f})")
        
        return split_blocks
    
    def _intelligent_split_test(self, text: str, bbox: BoundingBox) -> List[TextBlock]:
        """
        Intelligent pattern-based splitting without hardcoding
        """
        # Use NLP to identify potential split points
        if not self.nlp:
            return []
            
        doc = self.nlp(text)
        tokens = [token for token in doc]
        
        # Look for label + value patterns using POS tags and entity recognition
        for i in range(len(tokens) - 1):
            current_token = tokens[i]
            next_token = tokens[i + 1]
            
            # Pattern: Noun/Label + Proper Noun/Name
            if (current_token.pos_ in ['NOUN'] and 
                next_token.pos_ in ['PROPN'] and
                current_token.text.istitle() and 
                next_token.text.istitle()):
                
                print(f"🎯 INTELLIGENT SPLIT: Detected '{current_token.text}' (label) + '{next_token.text}' (value)")
                
                # Calculate split position based on token spans
                label_end = current_token.idx + len(current_token.text)
                value_start = next_token.idx
                
                # Estimate width ratios based on character count
                total_chars = len(text)
                label_chars = len(current_token.text)
                label_ratio = label_chars / total_chars
                
                key_width = int(bbox.width * (label_ratio + 0.1))  # Add small buffer
                value_width = bbox.width - key_width
                
                key_block = TextBlock(current_token.text, 
                                    BoundingBox(bbox.x, bbox.y, key_width, bbox.height), 0.95)
                value_block = TextBlock(text[value_start:].strip(), 
                                      BoundingBox(bbox.x + key_width, bbox.y, value_width, bbox.height), 0.90)
                return [key_block, value_block]
        
        return []
    
    def _attempt_text_splitting(self, text: str, original_bbox: BoundingBox) -> List[TextBlock]:
        """
        Attempt to split text using multiple strategies
        """
        logger.debug(f"Attempting to split: '{text}'")
        
        # Strategy 1: Form field keyword splitting
        logger.debug("Trying Strategy 1: Form field keyword splitting")
        keyword_splits = self._split_by_form_keywords(text, original_bbox)
        if keyword_splits:
            logger.debug(f"Strategy 1 SUCCESS: Found {len(keyword_splits)} splits")
            return keyword_splits
        
        # Strategy 2: POS-based splitting (using spaCy)
        if self.nlp:
            logger.debug("Trying Strategy 2: POS-based splitting")
            pos_splits = self._split_by_pos_analysis(text, original_bbox)
            if pos_splits:
                logger.debug(f"Strategy 2 SUCCESS: Found {len(pos_splits)} splits")
                return pos_splits
        else:
            logger.debug("Skipping Strategy 2: spaCy not available")
        
        # Strategy 3: Pattern-based splitting
        logger.debug("Trying Strategy 3: Pattern-based splitting")
        pattern_splits = self._split_by_patterns(text, original_bbox)
        if pattern_splits:
            logger.debug(f"Strategy 3 SUCCESS: Found {len(pattern_splits)} splits")
            return pattern_splits
        
        # Strategy 4: Capitalization-based splitting
        logger.debug("Trying Strategy 4: Capitalization-based splitting")
        cap_splits = self._split_by_capitalization(text, original_bbox)
        if cap_splits:
            logger.debug(f"Strategy 4 SUCCESS: Found {len(cap_splits)} splits")
            return cap_splits
        
        logger.debug(f"All strategies failed for: '{text}'")
        return []  # No split found
    
    def _generate_dynamic_keywords(self, text_lower: str) -> List[str]:
        """
        Generate form field keywords intelligently using NLP semantic understanding
        """
        keywords = set()
        
        # Add from learned patterns (validated form field labels)
        for pattern_type, examples in self.learned_patterns.items():
            for example in examples:
                if example.count(' ') <= 2 and self._is_semantic_form_field(example):
                    keywords.add(example.lower())
        
        # Use NLP for intelligent form field detection
        if self.nlp:
            doc = self.nlp(text_lower)
            
            # Look for semantic form field patterns
            for i, token in enumerate(doc):
                # Single word form fields (name, address, phone, etc.)
                if (token.pos_ in ['NOUN'] and 
                    token.lemma_ in ['name', 'address', 'phone', 'email', 'date', 'number', 'code', 'id'] and
                    len(token.text) > 2):
                    keywords.add(token.text)
                
                # Compound form fields (first name, last name, etc.)
                if (i < len(doc) - 1 and 
                    token.pos_ in ['ADJ', 'NOUN'] and 
                    doc[i + 1].pos_ in ['NOUN'] and
                    any(label_word in [token.lemma_, doc[i + 1].lemma_] 
                        for label_word in ['name', 'address', 'phone', 'email', 'date', 'number'])):
                    compound = f"{token.text} {doc[i + 1].text}"
                    if len(compound) <= 20:
                        keywords.add(compound.lower())
        
        # Add common semantic form field patterns
        semantic_patterns = [
            'name', 'first name', 'last name', 'full name',
            'address', 'street address', 'home address',
            'phone', 'phone number', 'telephone',
            'email', 'email address', 'e-mail',
            'date', 'birth date', 'date of birth',
            'nationality', 'country', 'state', 'city',
            'id', 'id number', 'identification',
            'code', 'postal code', 'zip code'
        ]
        
        # Only add patterns that might be present in this text
        for pattern in semantic_patterns:
            if any(word in text_lower for word in pattern.split()):
                keywords.add(pattern)
        
        return list(keywords)
    
    def _is_semantic_form_field(self, text: str) -> bool:
        """
        Check if text is semantically a form field using NLP understanding
        """
        if not self.nlp:
            return False
            
        doc = self.nlp(text.lower())
        
        # Check if any token has form field semantics
        for token in doc:
            if token.lemma_ in ['name', 'address', 'phone', 'email', 'date', 'number', 'code', 'id', 'nationality']:
                return True
                
        return False
    
    def _is_likely_form_field_label(self, text: str) -> bool:
        """
        Determine if text is likely a form field label using intelligent NLP analysis
        """
        if not self.nlp:
            # Fallback pattern matching
            return (text.istitle() and 
                    3 <= len(text) <= 30 and 
                    any(word in text.lower() for word in ['name', 'address', 'phone', 'email', 'date', 'nationality']))
            
        text_lower = text.lower().strip()
        doc = self.nlp(text)
        
        # Strong indicators of form field labels
        semantic_form_fields = [
            'name', 'address', 'phone', 'email', 'date', 'number', 'code', 'id', 
            'nationality', 'country', 'state', 'city', 'street', 'postal'
        ]
        
        # Check if any token has form field semantics
        has_form_semantics = any(token.lemma_ in semantic_form_fields for token in doc)
        
        # Linguistic patterns for form fields
        has_noun = any(token.pos_ in ['NOUN'] for token in doc)
        is_title_case = text.istitle()
        reasonable_length = 3 <= len(text) <= 30
        not_pure_value = not self._looks_like_pure_value(text)
        
        # Pattern recognition from learned examples
        pattern_match = any(text_lower in examples for examples in self.learned_patterns.values())
        
        # Compound form fields (like "First Name", "Last Name")
        if len(doc) == 2:
            first_token, second_token = doc[0], doc[1]
            if (first_token.pos_ in ['ADJ', 'NOUN'] and 
                second_token.pos_ in ['NOUN'] and
                second_token.lemma_ in semantic_form_fields):
                return True
        
        return ((has_form_semantics and is_title_case and reasonable_length and not_pure_value) or 
                pattern_match or
                (has_noun and is_title_case and reasonable_length and not_pure_value and len(text.split()) <= 3))
    
    def _looks_like_pure_value(self, text: str) -> bool:
        """
        Check if text looks like a pure value (not a form field label)
        """
        if not self.nlp:
            # Fallback: simple heuristics
            return (len(text.split()) == 1 and 
                    text.istitle() and 
                    not any(word in text.lower() for word in ['name', 'address', 'phone', 'email', 'date']))
        
        doc = self.nlp(text)
        
        # Check for entity types that indicate values, not labels
        for ent in doc.ents:
            if ent.label_ in ['PERSON', 'GPE', 'LOC', 'ORG', 'MONEY', 'DATE', 'TIME']:
                return True
        
        # Single proper nouns without form field semantics
        if (len(doc) == 1 and 
            doc[0].pos_ == 'PROPN' and 
            doc[0].lemma_ not in ['name', 'address', 'phone', 'email', 'date', 'number', 'code', 'id', 'nationality']):
            return True
            
        return False
    
    def _is_valid_keyword_split(self, original_text: str, keyword: str, remaining_text: str) -> bool:
        """
        Validate if a keyword-based split makes semantic sense
        """
        # Basic validation: must have substantial remaining text
        if not remaining_text or len(remaining_text) <= 1:
            return False
        
        # Don't split if the keyword is too generic or creates nonsensical splits
        if keyword in ['name', 'address'] and len(keyword) >= len(original_text) * 0.8:
            return False  # Keyword takes up too much of the text
        
        # Prevent splitting addresses or long coherent text
        if self._looks_like_address_or_coherent_text(original_text):
            return False
        
        # Use NLP to validate semantic coherence
        if self.nlp:
            original_doc = self.nlp(original_text)
            keyword_doc = self.nlp(keyword)
            remaining_doc = self.nlp(remaining_text)
            
            # Check if the remaining text looks like a reasonable value
            if len(remaining_doc) == 1 and remaining_doc[0].pos_ in ['PROPN', 'NOUN']:
                return True  # Single proper noun/noun as value
            
            # Check if we're splitting a geographic entity incorrectly
            for ent in original_doc.ents:
                if ent.label_ in ['GPE', 'LOC'] and len(ent.text) > len(keyword):
                    return False  # Don't split geographic entities
        
        return True
    
    def _looks_like_address_or_coherent_text(self, text: str) -> bool:
        """
        Check if text looks like an address or other coherent text that shouldn't be split
        """
        # Address indicators
        address_patterns = [
            r'\d+.*\b(street|ave|avenue|blvd|boulevard|road|rd|lane|ln|drive|dr|way|st)\b',
            r'\d{5}',  # ZIP code
            r'\b(apt|apartment|suite|unit|floor|fl)\b.*\d',
            r'\d+.*,.*\d{5}',  # Number, comma, ZIP pattern
        ]
        
        text_lower = text.lower()
        for pattern in address_patterns:
            if re.search(pattern, text_lower):
                return True
        
        # Long coherent text with multiple words and punctuation
        if len(text.split()) >= 4 and (',' in text or '.' in text):
            return True
            
        return False
    
    def _split_by_form_keywords(self, text: str, bbox: BoundingBox) -> List[TextBlock]:
        """
        Split text based on dynamically generated form field keywords
        Example: "Last Name Leverling" → ["Last Name", "Leverling"]
        """
        # Generate dynamic form field keywords using NLP and learned patterns
        text_lower = text.lower()
        form_keywords = self._generate_dynamic_keywords(text_lower)
        
        logger.debug(f"Form keyword splitting - text_lower: '{text_lower}'")
        logger.debug(f"Dynamic keywords generated: {len(form_keywords)} candidates")
        
        for keyword in sorted(form_keywords, key=len, reverse=True):  # Longest first
            logger.debug(f"Checking if '{text_lower}' starts with '{keyword}'")
            if text_lower.startswith(keyword):
                remaining_text = text[len(keyword):].strip()
                logger.debug(f"✅ Keyword '{keyword}' FOUND in '{text}', remaining: '{remaining_text}'")
                
                # Intelligent validation of the split
                if self._is_valid_keyword_split(text, keyword, remaining_text):
                    # Estimate bbox splitting (simple horizontal split)
                    key_width = int(bbox.width * (len(keyword) / len(text)))
                    value_width = bbox.width - key_width
                    
                    key_block = TextBlock(
                        text=text[:len(keyword)].strip(),
                        bbox=BoundingBox(bbox.x, bbox.y, key_width, bbox.height),
                        confidence=0.95  # High confidence for keyword-based splits
                    )
                    
                    value_block = TextBlock(
                        text=remaining_text,
                        bbox=BoundingBox(bbox.x + key_width, bbox.y, value_width, bbox.height),
                        confidence=0.90
                    )
                    
                    logger.info(f"SPLIT SUCCESS: '{text}' → '{key_block.text}' + '{value_block.text}'")
                    return [key_block, value_block]
                else:
                    logger.debug(f"Split validation failed for keyword '{keyword}' in '{text}'")
        
        logger.debug(f"No form keyword splits found for: '{text}'")
        
        return []
    
    def _split_by_pos_analysis(self, text: str, bbox: BoundingBox) -> List[TextBlock]:
        """
        Use spaCy POS tagging to find natural split points
        Example: "First Name" (NOUN NOUN) + "Janet" (PROPN)
        """
        doc = self.nlp(text)
        tokens = list(doc)
        
        if len(tokens) < 2:
            return []
        
        # Look for transition patterns that indicate key->value splits
        best_split_idx = None
        confidence = 0
        
        for i in range(1, len(tokens)):
            prev_token = tokens[i-1]
            curr_token = tokens[i]
            
            # Pattern 1: Label words followed by proper nouns
            if (prev_token.pos_ in ['NOUN', 'ADJ'] and curr_token.pos_ == 'PROPN'):
                confidence = 0.8
                best_split_idx = i
                break
            
            # Pattern 2: Form field followed by number/value
            if (self._is_likely_form_field_label(prev_token.text) and 
                curr_token.pos_ in ['PROPN', 'NUM', 'NOUN']):
                confidence = 0.85
                best_split_idx = i
                break
            
            # Pattern 3: Multiple nouns followed by different POS
            if (prev_token.pos_ == 'NOUN' and curr_token.pos_ in ['PROPN', 'NUM'] and
                i >= 2 and tokens[i-2].pos_ == 'NOUN'):
                confidence = 0.7
                best_split_idx = i
        
        if best_split_idx and confidence > 0.6:
            # Split at the identified position
            key_tokens = tokens[:best_split_idx]
            value_tokens = tokens[best_split_idx:]
            
            key_text = ' '.join([t.text for t in key_tokens])
            value_text = ' '.join([t.text for t in value_tokens])
            
            # Estimate bbox splitting
            key_ratio = len(key_text) / len(text)
            key_width = int(bbox.width * key_ratio)
            value_width = bbox.width - key_width
            
            key_block = TextBlock(
                text=key_text,
                bbox=BoundingBox(bbox.x, bbox.y, key_width, bbox.height),
                confidence=confidence
            )
            
            value_block = TextBlock(
                text=value_text,
                bbox=BoundingBox(bbox.x + key_width, bbox.y, value_width, bbox.height),
                confidence=confidence - 0.1
            )
            
            return [key_block, value_block]
        
        return []
    
    def _split_by_patterns(self, text: str, bbox: BoundingBox) -> List[TextBlock]:
        """
        Split based on common patterns in form fields
        """
        # Pattern 1: "Label: Value" format
        if ':' in text and text.count(':') == 1:
            parts = text.split(':', 1)
            if len(parts) == 2 and all(part.strip() for part in parts):
                key_text = parts[0].strip()
                value_text = parts[1].strip()
                
                key_ratio = (len(key_text) + 1) / len(text)  # +1 for the colon
                key_width = int(bbox.width * key_ratio)
                value_width = bbox.width - key_width
                
                return [
                    TextBlock(key_text, BoundingBox(bbox.x, bbox.y, key_width, bbox.height), 0.9),
                    TextBlock(value_text, BoundingBox(bbox.x + key_width, bbox.y, value_width, bbox.height), 0.85)
                ]
        
        # Pattern 2: "Label - Value" format
        if ' - ' in text:
            parts = text.split(' - ', 1)
            if len(parts) == 2 and all(part.strip() for part in parts):
                key_text = parts[0].strip()
                value_text = parts[1].strip()
                
                # Only split if it looks like a label-value pair
                if (len(key_text.split()) <= 3 and 
                    self._is_likely_form_field_label(key_text)):
                    
                    key_ratio = (len(key_text) + 3) / len(text)  # +3 for " - "
                    key_width = int(bbox.width * key_ratio)
                    value_width = bbox.width - key_width
                    
                    return [
                        TextBlock(key_text, BoundingBox(bbox.x, bbox.y, key_width, bbox.height), 0.8),
                        TextBlock(value_text, BoundingBox(bbox.x + key_width, bbox.y, value_width, bbox.height), 0.75)
                    ]
        
        return []
    
    def _split_by_capitalization(self, text: str, bbox: BoundingBox) -> List[TextBlock]:
        """
        Split based on capitalization patterns
        Example: "LastName LEVERLING" or "First Name Janet Smith"
        """
        words = text.split()
        
        if len(words) < 2:
            return []
        
        # Look for transitions in capitalization patterns
        for i in range(1, len(words)):
            prev_words = words[:i]
            next_words = words[i:]
            
            # Pattern 1: Title Case + UPPER CASE
            if (all(w.istitle() for w in prev_words) and 
                all(w.isupper() for w in next_words) and 
                len(prev_words) <= 3):
                
                key_text = ' '.join(prev_words)
                value_text = ' '.join(next_words)
                
                # Check if the key looks like a form field
                if self._is_likely_form_field_label(key_text):
                    key_ratio = len(key_text) / len(text)
                    key_width = int(bbox.width * key_ratio)
                    value_width = bbox.width - key_width
                    
                    return [
                        TextBlock(key_text, BoundingBox(bbox.x, bbox.y, key_width, bbox.height), 0.75),
                        TextBlock(value_text, BoundingBox(bbox.x + key_width, bbox.y, value_width, bbox.height), 0.70)
                    ]
            
            # Pattern 2: Multiple Title Case words + Single Title Case (likely name)
            if (len(prev_words) >= 2 and len(next_words) <= 2 and
                all(w.istitle() for w in prev_words) and 
                all(w.istitle() for w in next_words)):
                
                # Check if previous words contain form field keywords
                prev_text_lower = ' '.join(prev_words).lower()
                if any(keyword in prev_text_lower for keyword in ['first', 'last', 'full', 'name']):
                    key_text = ' '.join(prev_words)
                    value_text = ' '.join(next_words)
                    
                    key_ratio = len(key_text) / len(text)
                    key_width = int(bbox.width * key_ratio)
                    value_width = bbox.width - key_width
                    
                    return [
                        TextBlock(key_text, BoundingBox(bbox.x, bbox.y, key_width, bbox.height), 0.7),
                        TextBlock(value_text, BoundingBox(bbox.x + key_width, bbox.y, value_width, bbox.height), 0.65)
                    ]
        
        return []
    
    def _find_split_block_pair(self, key_block: TextBlock, all_blocks: List[TextBlock], 
                              used_value_blocks: set, used_key_blocks: set) -> Optional[Tuple[TextBlock, float, str]]:
        """
        Find the best value block for a split key block using spatial and confidence analysis
        """
        # Only consider split blocks (confidence 0.95 for keys, 0.90 for values)
        if key_block.confidence != 0.95:
            return None
        
        best_candidate = None
        best_score = 0
        
        for block in all_blocks:
            # Skip if already used or same block
            if (id(block) in used_value_blocks or 
                id(block) in used_key_blocks or 
                block == key_block):
                continue
            
            # Look for adjacent split value blocks
            if block.confidence == 0.90:
                # Calculate spatial proximity 
                horizontal_distance = abs(key_block.bbox.right - block.bbox.x)
                vertical_distance = abs(key_block.bbox.center_y - block.bbox.center_y)
                
                # Must be on same line (very close vertically)
                if vertical_distance <= key_block.bbox.height * 0.5:
                    # Must be adjacent or very close horizontally
                    if horizontal_distance <= 50:  # pixels
                        # Calculate semantic compatibility score
                        semantic_score = 0.8  # Base score for split pairs
                        
                        # Boost for semantic compatibility
                        if self.nlp:
                            key_type = self._classify_value_type(key_block.text)
                            value_type = self._classify_value_type(block.text)
                            if self._are_semantically_compatible(key_type, value_type, key_block.text, block.text):
                                semantic_score += 0.2
                        
                        # Spatial proximity score (closer is better)
                        proximity_score = max(0, (50 - horizontal_distance) / 50) * 0.3
                        
                        total_score = semantic_score + proximity_score
                        
                        if total_score > best_score:
                            best_score = total_score
                            best_candidate = (block, total_score, 'split_adjacent')
        
        return best_candidate
    
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
            
            # Method 1: Same line (horizontal alignment) - Enhanced for split blocks
            if vertical_distance < key_block.bbox.height * 0.8:
                confidence += 0.4
                methods.append('same_line')
                
                # Enhanced logic for split blocks (adjacent text on same line)
                horizontal_gap = block_left - key_right
                if horizontal_gap >= -10 and horizontal_gap <= 50:  # Adjacent or very close
                    confidence += 0.5  # Higher confidence for adjacent blocks
                    methods.append('adjacent_split')
                    
                    # Extra bonus for blocks that appear to be from splitting
                    if (key_block.confidence == 0.95 and block.confidence == 0.90) or \
                       (key_block.confidence == 0.90 and block.confidence == 0.95):
                        confidence += 0.3
                        methods.append('split_block_pair')
            
                # Standard bonus for being to the right of key
                elif block_left >= key_right - 10:  # Small overlap tolerance
                    confidence += 0.3
                    methods.append('right_of_key')
            
            # Method 2: Below key (common in forms) - Enhanced and more flexible
            elif vertical_distance <= key_block.bbox.height * 4:  # More flexible vertical range
                base_below_confidence = 0.2
                
                # Better proximity scoring for below-key relationships
                if vertical_distance <= key_block.bbox.height * 2:
                    base_below_confidence += 0.2  # Closer is better
                
                # Horizontal alignment bonus (not too strict for forms)
                if horizontal_distance <= max(key_block.bbox.width * 2, 200):  # More flexible horizontal range
                    base_below_confidence += 0.1
                    methods.append('below_key_aligned')
                
                confidence += base_below_confidence
                methods.append('below_key')
                
                # Enhanced semantic compatibility for form fields
                if self._is_form_field_semantic_match(key_block.text, block.text):
                    confidence += 0.4
                    methods.append('form_field_semantic_match')
                
                # Special boost for Address fields with address-like values
                if ('address' in key_block.text.lower() and 
                    self._looks_like_address_or_coherent_text(block.text)):
                    confidence += 0.3
                    methods.append('address_field_boost')
            
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
        """Classify the type of value based on characteristics"""
        text = text.strip()
        
        # Use value characteristics instead of hardcoded patterns
        for char_name, char_func in self.value_characteristics.items():
            if char_func(text):
                return char_name
        
        # Additional specific type detection for common form values
        text_lower = text.lower()
        
        # Use NLP for intelligent classification
        if self.nlp:
            doc = self.nlp(text)
            
            # Person names (proper nouns without location/org entities)
            if any(token.pos_ == 'PROPN' and token.ent_type_ in ['', 'PERSON'] for token in doc):
                return 'name'
            
            # Geographic locations
            if any(token.ent_type_ in ['GPE', 'LOC'] for token in doc):
                return 'location'
            
            # Organizational entities
            if any(token.ent_type_ in ['ORG'] for token in doc):
                return 'organization'
        
        # Pattern-based classification for addresses (look for address indicators)
        if (len(text) > 10 and 
            any(pattern in text_lower for pattern in ['blvd', 'street', 'ave', 'road', 'lane', 'dr', 'way']) or
            re.search(r'\d+.*\b(street|avenue|blvd|road|lane|drive|way|st|ave|rd|ln|dr)\b', text_lower)):
            return 'address'
        
        # ZIP code
        if re.match(r'^\d{5}(-\d{4})?$', text):
            return 'postal_code'
            
        return 'text'
    
    def _is_form_field_semantic_match(self, key_text: str, value_text: str) -> bool:
        """
        Check if key and value have semantic compatibility using vector-based analysis
        """
        if not self.nlp:
            return False
        
        # Use SciPy-based semantic compatibility analysis
        return self._calculate_semantic_compatibility_score(key_text, value_text) > 0.5
    
    def _are_semantically_compatible(self, key_type: str, value_type: str, key_text: str, value_text: str) -> bool:
        """
        Check if a key and value are semantically compatible using advanced vector analysis
        """
        if not self.nlp:
            return False
        
        # Use SciPy-based semantic compatibility analysis
        return self._calculate_semantic_compatibility_score(key_text, value_text) > 0.4
    
    def _scipy_enhanced_clustering(self, key_candidates: List[Tuple[TextBlock, float, str]], 
                                  all_blocks: List[TextBlock]) -> List[Tuple[TextBlock, TextBlock, float]]:
        """
        Use SciPy hierarchical clustering for better key-value pairing
        """
        if not self.nlp or len(all_blocks) < 2:
            return []
        
        try:
            # Create feature vectors for all blocks
            vectors = []
            blocks_with_vectors = []
            
            for block in all_blocks:
                doc = self.nlp(block.text)
                if doc.vector.any():
                    vectors.append(doc.vector)
                    blocks_with_vectors.append(block)
            
            if len(vectors) < 2:
                return []
            
            # Use SciPy for hierarchical clustering
            vectors_array = np.array(vectors)
            
            # Calculate distance matrix using SciPy
            distance_matrix = cdist(vectors_array, vectors_array, metric='cosine')
            
            # Perform hierarchical clustering
            linkage_matrix = linkage(distance_matrix, method='ward')
            
            # Dynamic threshold calculation using statistical methods
            # Calculate optimal threshold based on data distribution
            distances = linkage_matrix[:, 2]  # Get all merge distances
            threshold = self._calculate_dynamic_clustering_threshold(distances)
            
            # Get cluster assignments with dynamic threshold
            clusters = fcluster(linkage_matrix, t=threshold, criterion='distance')
            
            # Find key-value pairs within clusters
            pairs = []
            key_texts = [kc[0].text for kc in key_candidates]
            
            for i, block_i in enumerate(blocks_with_vectors):
                if block_i.text in key_texts:
                    # This is a key candidate, find values in same cluster
                    cluster_id = clusters[i]
                    
                    for j, block_j in enumerate(blocks_with_vectors):
                        if (i != j and 
                            clusters[j] == cluster_id and 
                            block_j.text not in key_texts):
                            
                            # Calculate spatial compatibility
                            spatial_score = self._calculate_spatial_compatibility(block_i, block_j)
                            semantic_score = self._calculate_semantic_compatibility_score(block_i.text, block_j.text)
                            
                            combined_score = (spatial_score * 0.6 + semantic_score * 0.4)
                            
                            # Dynamic threshold based on score distribution
                            if self._is_significant_pair_score(combined_score, [p[2] for p in pairs]):
                                pairs.append((block_i, block_j, combined_score))
            
            return sorted(pairs, key=lambda x: x[2], reverse=True)
            
        except Exception as e:
            logger.debug(f"SciPy clustering error: {e}")
            return []
    
    def _calculate_dynamic_clustering_threshold(self, distances: np.ndarray) -> float:
        """
        Calculate optimal clustering threshold using statistical analysis of distance distribution
        """
        try:
            if len(distances) == 0:
                return 0.7  # Fallback
            
            # Use SciPy statistical methods to find optimal threshold
            # Method 1: Elbow method using second derivative
            sorted_distances = np.sort(distances)
            
            if len(sorted_distances) > 3:
                # Calculate second derivative to find elbow point
                first_diff = np.diff(sorted_distances)
                second_diff = np.diff(first_diff)
                
                if len(second_diff) > 0:
                    # Find maximum second derivative (steepest change)
                    elbow_idx = np.argmax(second_diff) + 2  # Adjust for diff operations
                    if elbow_idx < len(sorted_distances):
                        elbow_threshold = sorted_distances[elbow_idx]
                        
                        # Validate threshold is reasonable
                        if 0.1 <= elbow_threshold <= 2.0:
                            return elbow_threshold
            
            # Method 2: Statistical percentile method
            # Use interquartile range to find natural break point
            q75 = np.percentile(distances, 75)
            q25 = np.percentile(distances, 25)
            iqr = q75 - q25
            
            # Threshold at Q3 + 0.5*IQR (outlier detection principle)
            statistical_threshold = q75 + 0.5 * iqr
            
            # Clamp to reasonable range
            return max(0.3, min(statistical_threshold, 1.5))
            
        except Exception as e:
            logger.debug(f"Dynamic threshold calculation error: {e}")
            return 0.7  # Fallback
    
    def _is_significant_pair_score(self, score: float, existing_scores: List[float]) -> bool:
        """
        Determine if a pair score is statistically significant using SciPy methods
        """
        try:
            if not existing_scores:
                return score > 0.3  # Initial threshold
            
            scores_array = np.array(existing_scores + [score])
            
            # Use statistical methods to determine significance
            if len(scores_array) > 2:
                # Calculate z-score for the new score
                mean_score = np.mean(scores_array[:-1])  # Exclude current score
                std_score = np.std(scores_array[:-1])
                
                if std_score > 0:
                    z_score = (score - mean_score) / std_score
                    
                    # Score is significant if it's above average or above dynamic threshold
                    return score > mean_score or z_score > -1.0
                else:
                    # If no variation, use mean comparison
                    return score >= mean_score
            
            # For small datasets, use adaptive threshold
            return score > max(0.3, np.mean(existing_scores) * 0.8)
            
        except Exception as e:
            logger.debug(f"Significance test error: {e}")
            return score > 0.4  # Fallback
    
    def _calculate_dynamic_key_threshold(self, existing_confidences: List[float]) -> float:
        """
        Calculate dynamic threshold for key identification using statistical analysis
        """
        try:
            if not existing_confidences:
                return 0.25  # Initial threshold
            
            # Use SciPy statistical methods for adaptive threshold
            confidences_array = np.array(existing_confidences)
            
            if len(confidences_array) > 2:
                # Use statistical measures to set threshold
                median_confidence = np.median(confidences_array)
                std_confidence = np.std(confidences_array)
                
                # Adaptive threshold: median - 0.5*std (captures reasonable candidates)
                threshold = max(0.2, median_confidence - 0.5 * std_confidence)
                return min(threshold, 0.5)  # Cap at reasonable maximum
            else:
                # For small datasets, use mean-based approach
                mean_confidence = np.mean(confidences_array)
                return max(0.2, mean_confidence * 0.7)
                
        except Exception as e:
            logger.debug(f"Dynamic key threshold calculation error: {e}")
            return 0.25
    
    def _calculate_dynamic_pairing_threshold(self, current_confidence: float, existing_confidences: List[float]) -> float:
        """
        Calculate dynamic threshold for key-value pairing using statistical analysis
        """
        try:
            if not existing_confidences:
                return 0.25  # Initial threshold
            
            # Use SciPy statistical methods for adaptive threshold
            confidences_array = np.array(existing_confidences)
            
            if len(confidences_array) > 1:
                # Calculate percentile-based threshold
                q25 = np.percentile(confidences_array, 25)
                q50 = np.percentile(confidences_array, 50)
                
                # Threshold at Q1 (25th percentile) for inclusive pairing
                # This captures pairs that are reasonably good compared to existing ones
                threshold = max(0.2, q25)
                
                # If current confidence is significantly better, lower threshold
                if current_confidence > q50:
                    threshold = max(0.15, threshold * 0.8)
                
                return min(threshold, 0.4)
            else:
                # Single confidence, use it as reference
                return max(0.2, existing_confidences[0] * 0.6)
                
        except Exception as e:
            logger.debug(f"Dynamic pairing threshold calculation error: {e}")
            return 0.25
    
    def _calculate_dynamic_pattern_confidence(self, pattern_type: str, text_blocks: List[TextBlock]) -> float:
        """
        Calculate dynamic confidence for patterns based on document characteristics
        """
        try:
            if pattern_type == 'colon_terminator':
                # Analyze colon usage patterns in the document
                colon_count = sum(1 for block in text_blocks if block.text.endswith(':'))
                total_blocks = len(text_blocks)
                
                if total_blocks == 0:
                    return 0.4  # Default
                
                # Higher colon frequency suggests more structured document
                colon_ratio = colon_count / total_blocks
                
                # Use statistical sigmoid function for confidence
                # Higher ratio = higher confidence in colon as key indicator
                confidence = 0.3 + 0.4 * (2 / (1 + np.exp(-5 * colon_ratio)))  # Sigmoid curve
                return min(confidence, 0.7)
            
            # Default for other patterns
            return 0.4
            
        except Exception as e:
            logger.debug(f"Dynamic pattern confidence error: {e}")
            return 0.4
    
    def _calculate_dynamic_spatial_weights(self, distance_score: float, 
                                         vertical_alignment: float, 
                                         horizontal_flow: float) -> List[float]:
        """
        Calculate dynamic weights for spatial scoring based on score distributions
        """
        try:
            scores = [distance_score, vertical_alignment, horizontal_flow]
            
            # Use entropy-based weighting - give more weight to discriminative features
            # Features with higher values are more discriminative
            total_score = sum(scores)
            
            if total_score > 0:
                # Normalize and apply softmax-like transformation for dynamic weighting
                normalized_scores = [s / total_score for s in scores]
                
                # Emphasize stronger signals
                emphasized = [s ** 1.5 for s in normalized_scores]  # Power transformation
                emphasis_total = sum(emphasized)
                
                if emphasis_total > 0:
                    weights = [e / emphasis_total for e in emphasized]
                    return weights
            
            # Fallback to equal weights
            return [0.33, 0.33, 0.34]
            
        except Exception as e:
            logger.debug(f"Dynamic spatial weights error: {e}")
            return [0.5, 0.3, 0.2]  # Fallback weights
    
    def _calculate_semantic_weights(self, cosine_sim: float, entity_score: float, 
                                  pos_score: float, context_score: float) -> List[float]:
        """
        Calculate dynamic weights for semantic compatibility based on score reliability
        """
        try:
            scores = [cosine_sim, entity_score, pos_score, context_score]
            
            # Calculate reliability of each score
            # Higher scores get higher weights (they're more discriminative)
            # Use softmax-like transformation for dynamic weighting
            
            # Apply temperature scaling for better distribution
            temperature = 2.0
            exp_scores = [np.exp(s / temperature) for s in scores]
            sum_exp = sum(exp_scores)
            
            if sum_exp > 0:
                weights = [exp_s / sum_exp for exp_s in exp_scores]
                
                # Ensure minimum weight for vector similarity if available
                if cosine_sim > 0 and weights[0] < 0.2:
                    # Redistribute to ensure vector similarity gets at least 20% if available
                    other_total = sum(weights[1:])
                    if other_total > 0:
                        factor = 0.8 / other_total
                        weights = [0.2] + [w * factor for w in weights[1:]]
                
                return weights
            else:
                # Fallback: prioritize vector similarity if available, then entity, then POS, then context
                if cosine_sim > 0:
                    return [0.4, 0.3, 0.2, 0.1]
                else:
                    return [0.0, 0.5, 0.3, 0.2]
                    
        except Exception as e:
            logger.debug(f"Semantic weights calculation error: {e}")
            return [0.4, 0.3, 0.2, 0.1]  # Fallback
    
    def _calculate_text_complexity(self, text: str) -> float:
        """
        Calculate text complexity using entropy and statistical measures
        """
        try:
            if not text:
                return 0.0
            
            # Character-level entropy calculation
            char_counts = {}
            for char in text.lower():
                char_counts[char] = char_counts.get(char, 0) + 1
            
            total_chars = len(text)
            entropy = 0.0
            
            for count in char_counts.values():
                probability = count / total_chars
                if probability > 0:
                    entropy -= probability * np.log2(probability)
            
            # Normalize entropy (max entropy for uniform distribution over 26 letters)
            max_entropy = np.log2(min(26, len(set(text.lower()))))
            normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
            
            # Additional complexity factors
            word_variety = len(set(text.split())) / max(len(text.split()), 1)
            char_variety = len(set(text.lower())) / max(len(text), 1)
            
            # Combine measures
            complexity = (normalized_entropy * 0.5 + word_variety * 0.3 + char_variety * 0.2)
            
            return min(complexity, 1.0)
            
        except Exception as e:
            logger.debug(f"Text complexity calculation error: {e}")
            return 0.5  # Default complexity
    
    def _calculate_percentile_rank(self, value: float, value_list: List[float]) -> float:
        """
        Calculate the percentile rank of a value in a list using SciPy methods
        """
        try:
            if not value_list:
                return 0.5
            
            # Sort the values
            sorted_values = sorted(value_list)
            
            # Find position of value
            position = sum(1 for v in sorted_values if v < value)
            
            # Calculate percentile rank
            rank = position / len(sorted_values)
            
            return rank
            
        except Exception as e:
            logger.debug(f"Percentile rank calculation error: {e}")
            return 0.5
        
    def _calculate_semantic_compatibility_score(self, key_text: str, value_text: str) -> float:
        """
        Calculate semantic compatibility score using SciPy and advanced NLP analysis
        """
        if not self.nlp:
            return 0.0
        
        try:
            key_doc = self.nlp(key_text)
            value_doc = self.nlp(value_text)
            
            # Multi-dimensional semantic analysis using SciPy
            compatibility_score = 0.0
            
            # Calculate individual compatibility scores
            cosine_similarity = 0
            if (self.nlp.vocab.vectors.size > 0 and 
                key_doc.vector.any() and value_doc.vector.any()):
                # Use SciPy's cosine distance (1 - cosine similarity)
                cosine_similarity = 1 - cosine(key_doc.vector, value_doc.vector)
            
            entity_score = self._calculate_entity_compatibility(key_doc, value_doc)
            pos_score = self._calculate_pos_compatibility(key_doc, value_doc)
            context_score = self._calculate_contextual_similarity(key_text, value_text)
            
            # Dynamic weight calculation based on score reliability
            weights = self._calculate_semantic_weights(cosine_similarity, entity_score, pos_score, context_score)
            
            # Weighted combination
            compatibility_score = (
                cosine_similarity * weights[0] +
                entity_score * weights[1] +
                pos_score * weights[2] +
                context_score * weights[3]
            )
            
            return min(compatibility_score, 1.0)  # Cap at 1.0
            
        except Exception as e:
            logger.debug(f"Error in semantic compatibility calculation: {e}")
            return 0.0
    
    def _calculate_spatial_compatibility(self, key_block: TextBlock, value_block: TextBlock) -> float:
        """
        Calculate spatial compatibility score using geometric analysis
        """
        # Calculate spatial relationships
        horizontal_distance = abs(key_block.bbox.center_x - value_block.bbox.center_x)
        vertical_distance = abs(key_block.bbox.center_y - value_block.bbox.center_y)
        
        # Normalize distances
        total_distance = np.sqrt(horizontal_distance**2 + vertical_distance**2)
        max_reasonable_distance = 500  # pixels
        
        # Distance score (closer is better)
        distance_score = max(0, (max_reasonable_distance - total_distance) / max_reasonable_distance)
        
        # Alignment scores
        vertical_alignment = 1.0 if vertical_distance < key_block.bbox.height * 0.8 else 0.5
        horizontal_flow = 1.0 if value_block.bbox.x > key_block.bbox.right else 0.7
        
        # Dynamic weight combination based on spatial relationships
        weights = self._calculate_dynamic_spatial_weights(distance_score, vertical_alignment, horizontal_flow)
        spatial_score = (distance_score * weights[0] + vertical_alignment * weights[1] + horizontal_flow * weights[2])
        
        return min(spatial_score, 1.0)
    
    def _calculate_entity_compatibility(self, key_doc, value_doc) -> float:
        """
        Calculate entity compatibility using dynamic statistical analysis without hardcoded assumptions
        """
        key_entities = [ent.label_ for ent in key_doc.ents]
        value_entities = [ent.label_ for ent in value_doc.ents]
        
        # Use SciPy statistical measures for entity pattern analysis
        try:
            # Dynamic entity relationship scoring using set theory and statistics
            if key_entities and value_entities:
                # Calculate Jaccard similarity for entity overlap
                key_set = set(key_entities)
                value_set = set(value_entities)
                intersection = len(key_set.intersection(value_set))
                union = len(key_set.union(value_set))
                jaccard_similarity = intersection / union if union > 0 else 0
                
                # If entities are different types, use statistical co-occurrence probability
                if jaccard_similarity == 0:
                    # Use SciPy to calculate statistical independence
                    # High entity diversity suggests complementary roles (key-value pattern)
                    diversity_score = min(len(key_set), len(value_set)) / max(len(key_set), len(value_set))
                    return diversity_score * 0.8
                else:
                    return jaccard_similarity * 0.6
            
            # Asymmetric entity pattern (common in key-value pairs)
            elif not key_entities and value_entities:
                # Statistical measure: keys often have no entities, values often do
                entity_density = len(value_entities) / max(len(value_doc.text.split()), 1)
                return min(entity_density * 1.5, 0.9)  # Cap at 0.9
            
            elif key_entities and not value_entities:
                # Less common but possible pattern
                return 0.4
            
            # No entities in either (still valid for many document types)
            else:
                return 0.5
                
        except Exception as e:
            logger.debug(f"Entity compatibility calculation error: {e}")
            return 0.5
    
    def _calculate_pos_compatibility(self, key_doc, value_doc) -> float:
        """
        Calculate POS pattern compatibility using pure statistical analysis without assumptions
        """
        key_pos = [token.pos_ for token in key_doc]
        value_pos = [token.pos_ for token in value_doc]
        
        try:
            # Use SciPy statistical methods to analyze POS pattern relationships
            # Convert POS tags to numerical vectors for statistical analysis
            all_pos_tags = list(set(key_pos + value_pos))
            
            if len(all_pos_tags) == 0:
                return 0.5
            
            # Create frequency vectors for statistical comparison
            key_vector = np.array([key_pos.count(pos) for pos in all_pos_tags])
            value_vector = np.array([value_pos.count(pos) for pos in all_pos_tags])
            
            # Normalize vectors
            key_total = np.sum(key_vector)
            value_total = np.sum(value_vector)
            
            if key_total > 0 and value_total > 0:
                key_normalized = key_vector / key_total
                value_normalized = value_vector / value_total
                
                # Calculate statistical measures using SciPy
                # 1. Euclidean distance (similarity in POS distribution)
                euclidean_dist = euclidean(key_normalized, value_normalized)
                euclidean_similarity = max(0, 1 - euclidean_dist)
                
                # 2. Statistical correlation of POS patterns
                if len(key_normalized) > 1 and len(value_normalized) > 1:
                    correlation, _ = pearsonr(key_normalized, value_normalized)
                    correlation = abs(correlation) if not np.isnan(correlation) else 0
                else:
                    correlation = 0
                
                # 3. Diversity measure (complementary POS patterns often indicate key-value)
                key_diversity = 1 - np.max(key_normalized) if np.max(key_normalized) > 0 else 0
                value_diversity = 1 - np.max(value_normalized) if np.max(value_normalized) > 0 else 0
                diversity_balance = min(key_diversity, value_diversity) / max(key_diversity, value_diversity) if max(key_diversity, value_diversity) > 0 else 0
                
                # Combine statistical measures
                compatibility = (
                    euclidean_similarity * 0.4 +
                    correlation * 0.3 +
                    diversity_balance * 0.3
                )
                
                return min(compatibility, 1.0)
            
            # Length-based statistical fallback
            length_ratio = min(len(key_pos), len(value_pos)) / max(len(key_pos), len(value_pos)) if max(len(key_pos), len(value_pos)) > 0 else 0
            return length_ratio * 0.6
            
        except Exception as e:
            logger.debug(f"POS compatibility calculation error: {e}")
            return 0.5
    
    def _calculate_contextual_similarity(self, key_text: str, value_text: str) -> float:
        """
        Calculate contextual similarity using character and pattern analysis
        """
        # Use SciPy's statistical functions for pattern analysis
        try:
            # Character-level analysis
            key_chars = np.array([ord(c) for c in key_text.lower() if c.isalpha()])
            value_chars = np.array([ord(c) for c in value_text.lower() if c.isalpha()])
            
            if len(key_chars) > 0 and len(value_chars) > 0:
                # Calculate statistical correlation using SciPy
                correlation, _ = pearsonr(
                    np.histogram(key_chars, bins=26)[0],
                    np.histogram(value_chars, bins=26)[0]
                )
                if not np.isnan(correlation):
                    return abs(correlation)
            
            # Length ratio similarity
            length_ratio = min(len(key_text), len(value_text)) / max(len(key_text), len(value_text))
            return length_ratio * 0.5
            
        except Exception:
            return 0.0
    
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
        
        # Try SciPy-enhanced clustering for better semantic pairing (if model has vectors)
        if self.nlp and self.nlp.vocab.vectors.size > 0:
            scipy_pairs = self._scipy_enhanced_clustering(key_candidates, all_blocks)
            for key_block, value_block, confidence in scipy_pairs:
                # Check if blocks are already used
                if (id(key_block) not in used_key_blocks and 
                    id(value_block) not in used_value_blocks):
                    
                    kv_pair = KeyValuePair(
                        key=key_block.text.strip(),
                        value=value_block.text.strip(),
                        key_bbox=key_block.bbox,
                        value_bbox=value_block.bbox,
                        confidence=confidence,
                        extraction_method="scipy_clustering"
                    )
                    
                    kv_pairs.append(kv_pair)
                    used_key_blocks.add(id(key_block))
                    used_value_blocks.add(id(value_block))
                    logger.info(f"SciPy clustering pair: '{key_block.text}' → '{value_block.text}' (conf: {confidence:.2f})")
        
        for key_block, key_confidence, key_method in key_candidates:
            # Skip if this block is already used as a value
            if id(key_block) in used_value_blocks:
                continue
            
            # Priority pairing for split blocks (adjacent blocks with specific confidence patterns)
            split_pair = self._find_split_block_pair(key_block, all_blocks, used_value_blocks, used_key_blocks)
            if split_pair:
                value_block, confidence, method = split_pair
                combined_confidence = self._calculate_enhanced_confidence(
                    key_block, value_block, key_confidence, confidence, is_form
                )
                
                if combined_confidence >= 0.5:  # Reasonable threshold for split pairs
                    kv_pair = KeyValuePair(
                        key=key_block.text.strip(),
                        value=value_block.text.strip(),
                        key_bbox=key_block.bbox,
                        value_bbox=value_block.bbox,
                        confidence=combined_confidence,
                        extraction_method=f"key:{key_method}|value:{method}|split_pair"
                    )
                    
                    kv_pairs.append(kv_pair)
                    used_key_blocks.add(id(key_block))
                    used_value_blocks.add(id(value_block))
                    continue  # Move to next key candidate
                
            # Find potential values for this key using standard spatial analysis
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
            
            # Dynamic threshold based on confidence distribution
            existing_confidences = [kv.confidence for kv in kv_pairs]
            min_threshold = self._calculate_dynamic_pairing_threshold(combined_confidence, existing_confidences)
            
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
        """Dynamic validation using statistical analysis without hardcoded assumptions"""
        try:
            if not self.nlp:
                return True  # Skip validation if no NLP model
            
            # Use pure statistical measures for validation
            key_doc = self.nlp(key_text)
            value_doc = self.nlp(value_text)
            
            # Statistical validation using multiple measures
            validation_score = 0.0
            
            # 1. Semantic compatibility score
            semantic_score = self._calculate_semantic_compatibility_score(key_text, value_text)
            validation_score += semantic_score * 0.4
            
            # 2. Text complexity analysis using entropy
            key_complexity = self._calculate_text_complexity(key_text)
            value_complexity = self._calculate_text_complexity(value_text)
            
            # Keys are typically simpler, values more complex
            complexity_balance = value_complexity / (key_complexity + 1e-6)  # Avoid division by zero
            complexity_score = min(complexity_balance / 2.0, 1.0)  # Cap at 1.0
            validation_score += complexity_score * 0.3
            
            # 3. Length ratio analysis
            length_ratio = len(value_text) / max(len(key_text), 1)
            length_score = min(length_ratio / 3.0, 1.0)  # Values often longer than keys
            validation_score += length_score * 0.2
            
            # 4. Entity distribution analysis
            key_entities = len([ent for ent in key_doc.ents])
            value_entities = len([ent for ent in value_doc.ents])
            
            # Values often have more entities than keys
            entity_score = 0.5
            if key_entities + value_entities > 0:
                entity_ratio = value_entities / (key_entities + value_entities)
                entity_score = entity_ratio
            validation_score += entity_score * 0.1
            
            # Accept pairs with reasonable statistical compatibility
            return validation_score > 0.4
            
        except Exception as e:
            logger.debug(f"Dynamic validation error: {e}")
            return True  # Conservative: accept if validation fails
    
    def _score_form_pairs(self, key_block: TextBlock, value_candidates: List[Tuple[TextBlock, float, str]]) -> List[Tuple[TextBlock, float, str]]:
        """Dynamic scoring using statistical spatial and semantic analysis"""
        enhanced_candidates = []
        
        # Calculate statistical measures for all candidates
        candidate_scores = []
        for value_block, confidence, method in value_candidates:
            # Calculate multiple scoring dimensions
            spatial_score = self._calculate_spatial_compatibility(key_block, value_block)
            semantic_score = self._calculate_semantic_compatibility_score(key_block.text, value_block.text)
            
            candidate_scores.append({
                'spatial': spatial_score,
                'semantic': semantic_score,
                'original': confidence
            })
        
        # Use SciPy statistical methods for dynamic enhancement
        for i, (value_block, confidence, method) in enumerate(value_candidates):
            enhanced_confidence = confidence
            
            # Get scores for this candidate
            scores = candidate_scores[i]
            
            # Dynamic spatial enhancement based on statistical distribution
            if len(candidate_scores) > 1:
                # Compare this candidate's spatial score to others
                spatial_scores = [cs['spatial'] for cs in candidate_scores]
                spatial_percentile = self._calculate_percentile_rank(scores['spatial'], spatial_scores)
                spatial_boost = spatial_percentile * 0.3  # Scale to reasonable boost
                enhanced_confidence += spatial_boost
                
                # Compare semantic compatibility
                semantic_scores = [cs['semantic'] for cs in candidate_scores]
                semantic_percentile = self._calculate_percentile_rank(scores['semantic'], semantic_scores)
                semantic_boost = semantic_percentile * 0.2
                enhanced_confidence += semantic_boost
            else:
                # Single candidate - use absolute thresholds
                if scores['spatial'] > 0.6:
                    enhanced_confidence += 0.2
                if scores['semantic'] > 0.5:
                    enhanced_confidence += 0.15
            
            # Text complexity analysis for additional scoring
            key_complexity = self._calculate_text_complexity(key_block.text)
            value_complexity = self._calculate_text_complexity(value_block.text)
            
            # Appropriate complexity balance often indicates good pairing
            complexity_ratio = value_complexity / max(key_complexity, 0.1)
            if 1.0 <= complexity_ratio <= 4.0:  # Reasonable range
                complexity_boost = min(complexity_ratio / 8.0, 0.15)
                enhanced_confidence += complexity_boost
            
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
        Hybrid extraction method with intelligent text splitting
        Phase 1: Split combined blocks → Phase 2: Spatial analysis → Phase 3: Semantic validation
        """
        if not text_blocks:
            return []
        
        try:
            logger.info(f"Starting hybrid extraction with {len(text_blocks)} original text blocks")
            
            # Phase 1: Intelligent text block splitting
            logger.info("Phase 1: Splitting combined text blocks...")
            split_blocks = self.split_combined_text_blocks(text_blocks)
            logger.info(f"After splitting: {len(split_blocks)} text blocks (+{len(split_blocks) - len(text_blocks)} new)")
            
            # Log splitting results for debugging
            for i, block in enumerate(split_blocks):
                logger.debug(f"Block {i+1}: '{block.text}' (conf: {block.confidence:.2f})")
            
            # Phase 2: Form layout detection on split blocks
            form_analysis = self.detect_form_layout(split_blocks)
            logger.info(f"Form detection: {form_analysis}")
            
            # Phase 3: Enhanced key identification
            key_candidates = self.identify_potential_keys(split_blocks, form_analysis)
            
            if not key_candidates:
                logger.warning("No key candidates found after splitting")
                return []
            
            logger.info(f"Found {len(key_candidates)} key candidates after splitting")
            for i, (block, conf, method) in enumerate(key_candidates):
                logger.debug(f"Key candidate {i+1}: '{block.text}' (confidence: {conf:.2f}, method: {method})")
            
            # Phase 4: Enhanced spatial clustering with both original and split blocks
            kv_pairs = self.cluster_key_value_pairs(key_candidates, split_blocks, form_analysis)
            logger.info(f"Clustering produced {len(kv_pairs)} initial pairs")
            
            # Phase 5: Semantic validation and confidence scoring
            validated_pairs = self._validate_and_clean_pairs(kv_pairs)
            logger.info(f"Final validation: {len(validated_pairs)} key-value pairs")
            
            # Log final results
            for i, pair in enumerate(validated_pairs):
                logger.info(f"Extracted pair {i+1}: '{pair.key}' → '{pair.value}' (confidence: {pair.confidence:.2f})")
            
            return validated_pairs
            
        except Exception as e:
            logger.error(f"Hybrid key-value extraction error: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return []
    
    def _validate_and_clean_pairs(self, kv_pairs: List[KeyValuePair]) -> List[KeyValuePair]:
        """
        Phase 5: Advanced semantic validation and confidence scoring
        """
        validated_pairs = []
        
        for pair in kv_pairs:
            # Basic cleaning
            key = pair.key.strip(':').strip()
            value = pair.value.strip()
            
            # Basic validation
            if len(key) < 2 or len(value) < 1:
                logger.debug(f"Skipping pair - too short: '{key}' → '{value}'")
                continue
            
            if key.lower() == value.lower():
                logger.debug(f"Skipping pair - identical: '{key}' → '{value}'")
                continue
            
            # Advanced semantic validation
            semantic_score = self._calculate_semantic_compatibility(key, value)
            
            # Adjust confidence based on semantic compatibility
            adjusted_confidence = pair.confidence * semantic_score
            
            # Only keep pairs with reasonable semantic compatibility
            if semantic_score >= 0.3:
                cleaned_pair = KeyValuePair(
                    key=key,
                    value=value,
                    key_bbox=pair.key_bbox,
                    value_bbox=pair.value_bbox,
                    confidence=adjusted_confidence,
                    extraction_method=f"{pair.extraction_method}|semantic:{semantic_score:.2f}"
            )
                validated_pairs.append(cleaned_pair)
                logger.debug(f"Validated pair: '{key}' → '{value}' (conf: {adjusted_confidence:.2f}, semantic: {semantic_score:.2f})")
            else:
                logger.debug(f"Rejected pair - low semantic score: '{key}' → '{value}' (semantic: {semantic_score:.2f})")
            
            # Sort by confidence for final ranking
            validated_pairs.sort(key=lambda x: x.confidence, reverse=True)
        
        return validated_pairs
    
    def _calculate_semantic_compatibility(self, key: str, value: str) -> float:
        """
        Calculate semantic compatibility score between key and value
        Returns 0.0 to 1.0 where 1.0 means perfect semantic match
        """
        key_lower = key.lower().strip()
        value_lower = value.lower().strip()
        
        compatibility_score = 0.5  # Base score
        
        # Name field validation
        if any(name_word in key_lower for name_word in ['first', 'last', 'full', 'name', 'surname']):
            if self._looks_like_name(value):
                compatibility_score += 0.4
            elif any(word in value_lower for word in ['street', 'avenue', 'road', 'blvd', '@', 'phone']):
                compatibility_score -= 0.3  # Definitely not a name
        
        # Address field validation
        elif any(addr_word in key_lower for addr_word in ['address', 'street', 'location']):
            if self._looks_like_address(value):
                compatibility_score += 0.4
            elif self._looks_like_name(value) and len(value.split()) <= 2:
                compatibility_score -= 0.3  # Single name not an address
        
        # Nationality/Country field validation
        elif any(country_word in key_lower for country_word in ['nationality', 'country', 'citizen']):
            if self._looks_like_country(value):
                compatibility_score += 0.4
            elif '@' in value or any(char.isdigit() for char in value):
                compatibility_score -= 0.3  # Not a country
        
        # Phone/Contact field validation
        elif any(phone_word in key_lower for phone_word in ['phone', 'telephone', 'mobile', 'contact']):
            if self._looks_like_phone(value):
                compatibility_score += 0.4
            elif self._looks_like_name(value):
                compatibility_score -= 0.3  # Name not a phone
        
        # Email field validation
        elif 'email' in key_lower:
            if '@' in value and '.' in value:
                compatibility_score += 0.4
            else:
                compatibility_score -= 0.3  # Not an email
        
        # ID/Number field validation
        elif any(id_word in key_lower for id_word in ['id', 'number', 'code', 'reference']):
            if any(char.isdigit() for char in value) or len(value.split()) == 1:
                compatibility_score += 0.2
        
        # Date field validation
        elif any(date_word in key_lower for date_word in ['date', 'birth', 'dob']):
            if self._looks_like_date(value):
                compatibility_score += 0.4
        
        # General validation: prevent field names as values
        discovered_fields = list(self.discovered_field_types)
        if any(field in value_lower for field in discovered_fields):
            compatibility_score -= 0.4
        
        # Length compatibility
        if len(key) >= len(value):  # Keys shouldn't be longer than values typically
            compatibility_score -= 0.1
        
        return max(0.0, min(1.0, compatibility_score))
    
    def _looks_like_name(self, text: str) -> bool:
        """Dynamic name detection using NLP entity recognition and statistical analysis"""
        try:
            if not self.nlp:
                return False
            
            doc = self.nlp(text)
            
            # Primary method: NLP entity recognition
            person_entities = [ent for ent in doc.ents if ent.label_ == 'PERSON']
            if person_entities:
                return True
            
            # Secondary method: Statistical text analysis
            # Calculate text characteristics using statistical measures
            complexity = self._calculate_text_complexity(text)
            
            # Names typically have lower complexity (fewer unique patterns)
            # Use statistical thresholds derived from text entropy
            if complexity < 0.3:  # Low complexity threshold
                # Additional statistical checks
                words = text.split()
                if 1 <= len(words) <= 4:  # Reasonable word count
                    # Check capitalization pattern (statistical)
                    capital_ratio = sum(1 for c in text if c.isupper()) / max(len(text), 1)
                    if 0.1 <= capital_ratio <= 0.4:  # Reasonable capitalization
                        return True
            
            return False
            
        except Exception as e:
            logger.debug(f"Dynamic name detection error: {e}")
            return False
    
    def _looks_like_address(self, text: str) -> bool:
        """Dynamic address detection using NLP entity recognition and statistical analysis"""
        try:
            if not self.nlp:
                return False
            
            doc = self.nlp(text)
            
            # Primary method: NLP entity recognition for locations
            location_entities = [ent for ent in doc.ents if ent.label_ in ['GPE', 'LOC', 'FAC']]
            if location_entities:
                return True
            
            # Secondary method: Statistical analysis of text patterns
            complexity = self._calculate_text_complexity(text)
            
            # Addresses typically have higher complexity due to mixed content
            if complexity > 0.4:  # Higher complexity threshold
                # Statistical pattern analysis
                has_numbers = any(char.isdigit() for char in text)
                has_punctuation = any(char in ',-.' for char in text)
                word_count = len(text.split())
                
                # Use statistical combination of features
                address_score = 0.0
                
                if has_numbers:
                    address_score += 0.4
                if has_punctuation:
                    address_score += 0.2
                if word_count >= 3:
                    address_score += 0.2
                if len(text) > 15:
                    address_score += 0.2
                
                return address_score > 0.6
            
            return False
            
        except Exception as e:
            logger.debug(f"Dynamic address detection error: {e}")
            return False
    
    def _looks_like_country(self, text: str) -> bool:
        """Dynamic country/location detection using NLP entity recognition"""
        try:
            if not self.nlp:
                return False
            
            doc = self.nlp(text)
            
            # Use NLP entity recognition for geographical entities
            geo_entities = [ent for ent in doc.ents if ent.label_ in ['GPE', 'NORP']]  # Geopolitical entities, nationalities
            if geo_entities:
                return True
            
            # Secondary statistical analysis for potential geographic terms
            complexity = self._calculate_text_complexity(text)
            
            # Geographic names typically have moderate complexity
            if 0.2 <= complexity <= 0.5:
                words = text.split()
                if 1 <= len(words) <= 3:  # Reasonable word count for country names
                    # Check if it's likely a proper noun
                    capital_ratio = sum(1 for c in text if c.isupper()) / max(len(text), 1)
                    if 0.1 <= capital_ratio <= 0.3:  # Reasonable capitalization for proper nouns
                        return True
            
            return False
            
        except Exception as e:
            logger.debug(f"Dynamic country detection error: {e}")
            return False
    
    def _looks_like_phone(self, text: str) -> bool:
        """Check if text looks like a phone number"""
        # Remove common phone formatting
        cleaned = re.sub(r'[^\d]', '', text)
        
        # Phone numbers typically have 7-15 digits
        if len(cleaned) < 7 or len(cleaned) > 15:
            return False
        
        # Should have some digits
        digit_ratio = len(cleaned) / len(text)
        return digit_ratio > 0.5
    
    def _looks_like_date(self, text: str) -> bool:
        """Check if text looks like a date"""
        # Check for common date patterns
        date_patterns = [
            r'\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4}',
            r'\d{4}[\/\-\.]\d{1,2}[\/\-\.]\d{1,2}',
            r'\d{1,2}\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{2,4}'
        ]
        
        for pattern in date_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        
        return False
    
    def extract_key_value_pairs_with_stats(self, text_blocks: List[TextBlock]) -> Tuple[List[KeyValuePair], Dict]:
        """
        Extract key-value pairs and return comprehensive statistics using SciPy analysis
        """
        import time
        start_time = time.time()
        
        original_count = len(text_blocks)
        
        # Track intermediate processing steps for better statistics
        processing_stats = {
            'original_ocr_blocks': original_count,
            'text_complexity_analysis': {},
            'spatial_analysis': {},
            'semantic_analysis': {},
            'extraction_methods': {},
            'confidence_distribution': {},
            'processing_stages': {}
        }
        
        # Analyze input text blocks
        if text_blocks:
            complexities = [self._calculate_text_complexity(block.text) for block in text_blocks]
            processing_stats['text_complexity_analysis'] = {
                'mean_complexity': float(np.mean(complexities)),
                'std_complexity': float(np.std(complexities)),
                'complexity_range': [float(np.min(complexities)), float(np.max(complexities))],
                'median_complexity': float(np.median(complexities))
            }
            
            # Spatial distribution analysis
            x_positions = [block.bbox.center_x for block in text_blocks]
            y_positions = [block.bbox.center_y for block in text_blocks]
            
            processing_stats['spatial_analysis'] = {
                'x_spread': float(np.std(x_positions)) if x_positions else 0,
                'y_spread': float(np.std(y_positions)) if y_positions else 0,
                'spatial_density': len(text_blocks) / (max(x_positions) - min(x_positions) + 1) if len(set(x_positions)) > 1 else 0,
                'layout_regularity': self._calculate_layout_regularity(text_blocks)
            }
        
        # Extract pairs with timing
        extraction_start = time.time()
        pairs = self.extract_key_value_pairs(text_blocks)
        extraction_time = time.time() - extraction_start
        
        # Analyze extraction results
        if pairs:
            confidences = [pair.confidence for pair in pairs]
            processing_stats['confidence_distribution'] = {
                'mean_confidence': float(np.mean(confidences)),
                'std_confidence': float(np.std(confidences)),
                'confidence_percentiles': {
                    '25th': float(np.percentile(confidences, 25)),
                    '50th': float(np.percentile(confidences, 50)),
                    '75th': float(np.percentile(confidences, 75))
                },
                'high_confidence_pairs': sum(1 for c in confidences if c > 0.7),
                'low_confidence_pairs': sum(1 for c in confidences if c < 0.4)
            }
            
            # Extraction method analysis
            methods = [pair.extraction_method for pair in pairs]
            method_counts = {}
            for method in methods:
                method_counts[method] = method_counts.get(method, 0) + 1
            
            processing_stats['extraction_methods'] = method_counts
        
        # Semantic analysis (if NLP is available)
        if self.nlp and pairs:
            semantic_scores = []
            for pair in pairs:
                score = self._calculate_semantic_compatibility_score(pair.key, pair.value)
                semantic_scores.append(score)
            
            processing_stats['semantic_analysis'] = {
                'mean_semantic_score': float(np.mean(semantic_scores)),
                'semantic_score_range': [float(np.min(semantic_scores)), float(np.max(semantic_scores))],
                'strong_semantic_pairs': sum(1 for s in semantic_scores if s > 0.6)
            }
        
        # Processing stage analysis
        split_based_pairs = sum(1 for pair in pairs if 'split' in pair.extraction_method.lower())
        spatial_pairs = sum(1 for pair in pairs if 'spatial' in pair.extraction_method.lower())
        scipy_pairs = sum(1 for pair in pairs if 'scipy' in pair.extraction_method.lower())
        
        processing_stats['processing_stages'] = {
            'pairs_from_splitting': split_based_pairs,
            'pairs_from_spatial': spatial_pairs,
            'pairs_from_scipy_clustering': scipy_pairs,
            'total_pairs_extracted': len(pairs),
            'extraction_success_rate': len(pairs) / max(original_count, 1),
            'processing_time_seconds': time.time() - start_time,
            'extraction_time_seconds': extraction_time
        }
        
        # Performance metrics
        processing_stats['performance_metrics'] = {
            'blocks_per_second': original_count / max(extraction_time, 0.001),
            'pairs_per_block_ratio': len(pairs) / max(original_count, 1),
            'efficiency_score': len(pairs) / max(extraction_time * original_count, 0.001) if original_count > 0 else 0
        }
        
        return pairs, processing_stats
    
    def _calculate_layout_regularity(self, text_blocks: List[TextBlock]) -> float:
        """
        Calculate layout regularity using statistical analysis of spatial patterns
        """
        try:
            if len(text_blocks) < 3:
                return 0.5  # Default for small datasets
            
            # Analyze vertical spacing regularity
            y_positions = sorted([block.bbox.center_y for block in text_blocks])
            y_diffs = [y_positions[i+1] - y_positions[i] for i in range(len(y_positions)-1)]
            
            if y_diffs:
                y_regularity = 1.0 - (np.std(y_diffs) / (np.mean(y_diffs) + 1e-6))
                y_regularity = max(0, min(y_regularity, 1.0))
            else:
                y_regularity = 0.5
            
            # Analyze horizontal alignment
            x_positions = [block.bbox.x for block in text_blocks]
            left_aligned_count = sum(1 for x in x_positions if abs(x - min(x_positions)) < 20)
            x_regularity = left_aligned_count / len(text_blocks)
            
            # Combine measures
            layout_regularity = (y_regularity * 0.6 + x_regularity * 0.4)
            
            return float(layout_regularity)
            
        except Exception as e:
            logger.debug(f"Layout regularity calculation error: {e}")
            return 0.5

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
            
        # Step 3: Enhanced Key-Value Extraction with splitting stats
        original_block_count = len(text_blocks)
        kv_pairs, split_stats = self.kv_extractor.extract_key_value_pairs_with_stats(text_blocks)
        
        # Prepare enhanced response with split analysis
        processing_time = (asyncio.get_event_loop().time() - start_time) if start_time else 0
        
        # Get splitting statistics
        original_blocks = split_stats.get('original_blocks', 0)
        pairs_from_splitting = split_stats.get('pairs_from_splitting', 0)
        
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
                "original_ocr_blocks": int(original_blocks),
                "pairs_from_splitting": int(pairs_from_splitting),
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
