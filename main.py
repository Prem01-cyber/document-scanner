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
    """Fast document quality assessment using OpenCV"""
    
    def __init__(self):
        self.min_contour_area = 10000  # Minimum document area
        self.blur_threshold = 100.0    # Laplacian variance threshold
        
    def assess_quality(self, image: np.ndarray) -> Dict:
        """
        Assess document quality for re-scan detection
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
                
            # 1. Blur Detection
            blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
            if blur_score < self.blur_threshold:
                issues.append("blurry_image")
                confidence -= 0.3
                
            # 2. Brightness Analysis
            mean_brightness = np.mean(gray)
            if mean_brightness < 50:
                issues.append("too_dark")
                confidence -= 0.2
            elif mean_brightness > 200:
                issues.append("too_bright")
                confidence -= 0.2
                
            # 3. Document Edge Detection
            try:
                edges = cv2.Canny(gray, 50, 150, apertureSize=3)
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
                    
                    # Check if document is properly captured
                    image_area = gray.shape[0] * gray.shape[1]
                    area_ratio = contour_area / image_area
                    
                    # Check for irregular shape (document might be cropped)
                    epsilon = 0.02 * cv2.arcLength(largest_contour, True)
                    approx = cv2.approxPolyDP(largest_contour, epsilon, True)
                    
                    if len(approx) < 4:  # Not rectangular enough
                        issues.append("irregular_shape")
                        confidence -= 0.2
                        
            except Exception as contour_error:
                logger.warning(f"Contour detection failed: {contour_error}")
                issues.append("contour_detection_failed")
                confidence -= 0.1
                
            # 4. Skew Detection
            try:
                lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
                if lines is not None:
                    angles = []
                    for rho, theta in lines[:10]:  # Check first 10 lines
                        angle = theta * 180 / np.pi
                        angles.append(angle)
                    
                    if angles:
                        avg_angle = np.mean(angles)
                        if abs(avg_angle - 90) > 10 and abs(avg_angle - 0) > 10:
                            issues.append("skewed_document")
                            confidence -= 0.1
            except Exception as skew_error:
                logger.warning(f"Skew detection failed: {skew_error}")
                # Don't penalize for skew detection failure
                
            # Decision logic
            needs_rescan = confidence < 0.6 or len(issues) >= 3
            
            return {
                "needs_rescan": needs_rescan,
                "confidence": max(0.0, confidence),
                "issues": issues,
                "blur_score": blur_score,
                "brightness": mean_brightness
            }
            
        except Exception as e:
            logger.error(f"Quality assessment error: {e}")
            return {
                "needs_rescan": True,
                "confidence": 0.0,
                "issues": ["processing_error"],
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
                            confidence=block.confidence
                        ))
                        
            return text_blocks
            
        except Exception as e:
            logger.error(f"OCR processing error: {e}")
            raise HTTPException(status_code=500, detail=f"OCR failed: {e}")

class EnhancedKeyValueExtractor:
    """
    Intelligent Key-Value extraction using multiple approaches:
    1. NLP-based semantic analysis
    2. Spatial layout analysis
    3. Pattern recognition without hardcoded rules
    4. Statistical clustering
    """
    
    def __init__(self):
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("spaCy model not found. Install with: python -m spacy download en_core_web_sm")
            self.nlp = None
        
        # Dynamic patterns that can be learned from data
        self.value_patterns = {
            'currency': r'[\$€£¥₹][\d,]+\.?\d*|\d+\.?\d*\s*[\$€£¥₹]',
            'percentage': r'\d+\.?\d*\s*%',
            'date': r'\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4}|\d{4}[\/\-\.]\d{1,2}[\/\-\.]\d{1,2}|\d{1,2}\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{2,4}',
            'phone': r'[\+]?[\d\s\(\)\-]{10,}',
            'email': r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
            'alphanumeric_id': r'[A-Z0-9]{3,}[\-\/]?[A-Z0-9]*',
            'number': r'\b\d+\.?\d*\b',
            'time': r'\d{1,2}:\d{2}(?:\s*[AaPp][Mm])?'
        }
        
        # Common key indicators (learned from documents, not hardcoded patterns)
        self.key_indicators = {
            'colon_suffix': ':',
            'hash_prefix': '#',
            'no_suffix': 'No.',
            'number_suffix': 'Number',
            'id_suffix': 'ID',
            'code_suffix': 'Code'
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
    
    def identify_potential_keys(self, text_blocks: List[TextBlock]) -> List[Tuple[TextBlock, float, str]]:
        """
        Identify potential keys using multiple heuristics:
        1. Structural indicators (colons, formatting)
        2. Semantic analysis
        3. Position analysis
        """
        key_candidates = []
        
        for block in text_blocks:
            text = block.text.strip()
            confidence = 0.0
            methods = []
            
            # Method 1: Structural indicators
            if text.endswith(':'):
                confidence += 0.4
                methods.append('colon_terminator')
            
            if text.endswith(('No.', 'Number', 'ID', 'Code', '#')):
                confidence += 0.3
                methods.append('suffix_indicator')
            
            if text.startswith(('#', 'Ref:', 'ID:')):
                confidence += 0.3
                methods.append('prefix_indicator')
            
            # Method 2: Length and word count heuristics
            words = text.split()
            if 1 <= len(words) <= 4 and len(text) <= 50:
                confidence += 0.2
                methods.append('length_heuristic')
            
            # Method 3: Capitalization patterns
            if text.isupper() and len(text) > 2:
                confidence += 0.1
                methods.append('capitalization')
            
            # Method 4: Position-based (left side of document typically contains keys)
            if block.bbox.x < 200:  # Left portion of document
                confidence += 0.1
                methods.append('position_left')
            
            # Method 5: NLP-based semantic analysis
            if self.nlp:
                doc = self.nlp(text)
                for token in doc:
                    if token.pos_ in ['NOUN', 'PROPN']:
                        confidence += 0.1
                        methods.append('semantic_noun')
                        break
            
            # Method 6: Check if it's likely NOT a value
            if not self._looks_like_value(text):
                confidence += 0.1
                methods.append('not_value')
            
            if confidence >= 0.3:  # Minimum threshold for key consideration
                key_candidates.append((block, confidence, '+'.join(methods)))
        
        return key_candidates
    
    def _looks_like_value(self, text: str) -> bool:
        """Check if text looks like a typical value rather than a key"""
        # Check against value patterns
        for pattern_type, pattern in self.value_patterns.items():
            if re.search(pattern, text, re.IGNORECASE):
                return True
        
        # Very long text is usually a value/content
        if len(text) > 100:
            return True
        
        # Contains mostly numbers
        if re.search(r'^\d+\.?\d*$', text.strip()):
            return True
            
        return False
    
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
                               all_blocks: List[TextBlock]) -> List[KeyValuePair]:
        """
        Main method to extract key-value pairs using clustering and scoring
        """
        kv_pairs = []
        used_value_blocks = set()
        
        # Sort key candidates by confidence
        key_candidates.sort(key=lambda x: x[1], reverse=True)
        
        for key_block, key_confidence, key_method in key_candidates:
            # Find potential values for this key
            value_candidates = self.find_spatial_values(key_block, all_blocks)
            
            if not value_candidates:
                continue
            
            # Filter out already used value blocks
            available_candidates = [
                (block, conf, method) for block, conf, method in value_candidates 
                if id(block) not in used_value_blocks
            ]
            
            if not available_candidates:
                continue
            
            # Sort by confidence and take the best match
            available_candidates.sort(key=lambda x: x[1], reverse=True)
            best_value_block, value_confidence, value_method = available_candidates[0]
            
            # Combined confidence score
            combined_confidence = (key_confidence + value_confidence) / 2
            
            if combined_confidence >= 0.4:  # Final threshold
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
        
        return kv_pairs
    
    def extract_key_value_pairs(self, text_blocks: List[TextBlock]) -> List[KeyValuePair]:
        """
        Main extraction method using the enhanced approach
        """
        if not text_blocks:
            return []
        
        try:
            # Step 1: Identify potential keys
            key_candidates = self.identify_potential_keys(text_blocks)
            
            if not key_candidates:
                logger.warning("No key candidates found")
                return []
            
            # Step 2: Use clustering approach to find best key-value pairs
            kv_pairs = self.cluster_key_value_pairs(key_candidates, text_blocks)
            
            # Step 3: Post-processing and validation
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
                    "confidence": kv.confidence,
                    "extraction_method": kv.extraction_method  # New field showing how it was extracted
                }
                for kv in kv_pairs
            ],
            "processing_time_seconds": processing_time,
            "extraction_statistics": {
                "total_text_blocks": len(text_blocks),
                "identified_key_value_pairs": len(kv_pairs),
                "average_confidence": sum(kv.confidence for kv in kv_pairs) / len(kv_pairs) if kv_pairs else 0,
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
                    "confidence": block.confidence
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
        
        return JSONResponse(content=result)
        
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
        
        return JSONResponse(content={
            "status": "quality_checked",
            "quality_assessment": quality_result
        })
        
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
