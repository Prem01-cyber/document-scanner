# Hybrid Document Processor
# Updated processor that uses hybrid extraction (adaptive + LLM fallback)

import numpy as np
import cv2
from typing import Dict, List, Union
import logging
import asyncio
import time
from fastapi import HTTPException
from prometheus_client import Histogram
from .logging_utils import configure_json_logging
import base64
from skimage.filters import threshold_sauvola

from quality.adaptive_quality_checker import AdaptiveDocumentQualityChecker
from .hybrid_kv_extractor import HybridKeyValueExtractor, ExtractionStrategy, HybridExtractionResult
from .llm_kv_extractor import LLMProvider, LLMKeyValueExtractor, LLMKeyValuePair
from .adaptive_kv_extractor import KeyValuePair
from .config import adaptive_config

logger = logging.getLogger(__name__)

class GoogleOCRProcessor:
    """Google Cloud Vision API integration (unchanged)"""
    
    def __init__(self):
        try:
            from google.cloud import vision
            self.client = vision.ImageAnnotatorClient()
        except Exception as e:
            logger.error(f"Failed to initialize Google Cloud Vision client: {e}")
            raise HTTPException(status_code=500, detail="Google Cloud Vision API not configured")
    
    def extract_text_with_bounds(self, image_bytes: bytes):
        """Extract text with bounding boxes - returns structured blocks, raw text, and preprocessed image bytes
        Applies basic preprocessing (deskew, denoise, adaptive threshold) before OCR.
        """
        try:
            from google.cloud import vision
            from .adaptive_kv_extractor import TextBlock, BoundingBox
            
            # Preprocess image before OCR
            nparr = np.frombuffer(image_bytes, np.uint8)
            img_cv = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img_cv is None:
                raise Exception("Invalid image data for OCR")

            processed = self._preprocess_for_ocr(img_cv)
            success, processed_buf = cv2.imencode('.png', processed)
            if not success:
                raise Exception("Failed to encode preprocessed image")
            processed_png_bytes = processed_buf.tobytes()
            image = vision.Image(content=processed_png_bytes)
            response = self.client.document_text_detection(image=image)
            
            if response.error.message:
                raise Exception(f"OCR Error: {response.error.message}")
            
            text_blocks = []
            raw_text_parts = []
            
            # Process structured blocks for adaptive extraction
            for page in response.full_text_annotation.pages:
                for block in page.blocks:
                    block_text = ""
                    
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
                        raw_text_parts.append(block_text.strip())
            
            # Also get raw text for LLM extraction
            raw_text = response.full_text_annotation.text if response.full_text_annotation else ""
            
            # If raw text is empty, use concatenated blocks
            if not raw_text:
                raw_text = "\n".join(raw_text_parts)
            
            return text_blocks, raw_text, processed_png_bytes
            
        except Exception as e:
            logger.error(f"OCR processing error: {e}")
            raise HTTPException(status_code=500, detail=f"OCR failed: {e}")

    def _preprocess_for_ocr(self, image_bgr: np.ndarray) -> np.ndarray:
        """Apply perspective correction, denoise, illumination normalization, deskew, advanced binarization,
        morphology cleanup, and line removal to improve OCR robustness.
        Returns a single-channel 8-bit preprocessed image.
        """
        # 1) Grayscale
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

        # 2) (Optional) Perspective correction by detecting document contour
        corrected = self._perspective_correction(gray)

        # 3) If very small, upscale slightly for better readability
        corrected = self._maybe_super_res(corrected)

        # 4) Denoise (fast non-local means) and illumination normalization
        denoised = cv2.fastNlMeansDenoising(corrected, h=10, templateWindowSize=7, searchWindowSize=21)
        norm = self._illumination_normalize(denoised)

        # 5) Light sharpening (unsharp mask)
        sharp = self._unsharp_mask(norm, amount=1.0, threshold=3)

        # 6) Estimate skew and deskew
        deskewed = self._deskew_by_hough(sharp)

        # 7) Advanced binarization: compare Otsu, Sauvola, and adaptive Gaussian; choose best by edge metric
        binary = self._advanced_binarization(deskewed)

        # 8) Morphological cleanup
        cleaned = self._morphology_cleanup(binary)

        # 9) Line removal (table lines) to avoid interfering with OCR
        no_lines = self._remove_lines(cleaned)

        return no_lines

    def _perspective_correction(self, gray: np.ndarray) -> np.ndarray:
        img = gray
        try:
            small = cv2.resize(gray, (0, 0), fx=0.5, fy=0.5) if max(gray.shape) > 1400 else gray
            edges = cv2.Canny(small, 50, 150)
            contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            h_s, w_s = small.shape[:2]
            best = None
            best_area = 0
            for cnt in contours:
                peri = cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
                if len(approx) == 4:
                    area = cv2.contourArea(approx)
                    if area > best_area:
                        best_area = area
                        best = approx
            if best is not None and best_area > 0.2 * (w_s * h_s):
                # order points and warp
                pts = best.reshape(4, 2).astype(np.float32)
                # scale back if resized
                if small is not gray:
                    scale_x = gray.shape[1] / w_s
                    scale_y = gray.shape[0] / h_s
                    pts[:, 0] *= scale_x
                    pts[:, 1] *= scale_y
                warped = self._four_point_transform(gray, pts)
                return warped
        except Exception:
            pass
        return img

    def _four_point_transform(self, img: np.ndarray, pts: np.ndarray) -> np.ndarray:
        # order points: top-left, top-right, bottom-right, bottom-left
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        (tl, tr, br, bl) = rect
        widthA = np.linalg.norm(br - bl)
        widthB = np.linalg.norm(tr - tl)
        maxWidth = int(max(widthA, widthB))
        heightA = np.linalg.norm(tr - br)
        heightB = np.linalg.norm(tl - bl)
        maxHeight = int(max(heightA, heightB))
        dst = np.array([[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]], dtype="float32")
        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(img, M, (maxWidth, maxHeight), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        return warped

    def _illumination_normalize(self, gray: np.ndarray) -> np.ndarray:
        # CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(gray)
        # Background estimation by large Gaussian blur and division
        bg = cv2.GaussianBlur(cl, (0, 0), sigmaX=21, sigmaY=21)
        norm = cv2.divide(cl, bg, scale=128)
        return norm

    def _unsharp_mask(self, img: np.ndarray, amount: float = 1.0, threshold: int = 3) -> np.ndarray:
        blurred = cv2.GaussianBlur(img, (0, 0), sigmaX=1.0)
        sharp = cv2.addWeighted(img, 1 + amount, blurred, -amount, 0)
        if threshold > 0:
            low_contrast_mask = (cv2.absdiff(img, blurred) < threshold).astype(np.uint8) * 255
            sharp = np.where(low_contrast_mask == 255, img, sharp).astype(np.uint8)
        return sharp

    def _deskew_by_hough(self, gray: np.ndarray) -> np.ndarray:
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=180)
        angle_deg = 0.0
        if lines is not None and len(lines) > 0:
            angles = []
            for rho, theta in lines[:, 0]:
                deg = (theta * 180.0 / np.pi) - 90.0
                if -45 < deg < 45:
                    angles.append(deg)
            if angles:
                angle_deg = float(np.median(angles))
        if abs(angle_deg) > 0.3:
            h, w = gray.shape
            center = (w // 2, h // 2)
            rot_mat = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
            return cv2.warpAffine(gray, rot_mat, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        return gray

    def _advanced_binarization(self, gray: np.ndarray) -> np.ndarray:
        # Otsu
        _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # Sauvola
        try:
            win = 31 if min(gray.shape) > 300 else 15
            sau_t = threshold_sauvola(gray, window_size=win)
            sau = (gray > sau_t).astype(np.uint8) * 255
        except Exception:
            sau = otsu
        # Adaptive Gaussian
        ada = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 15)

        # Choose best by edge density heuristic
        def edge_score(img_bin: np.ndarray) -> float:
            e = cv2.Canny(img_bin, 50, 150)
            ratio = np.mean(img_bin == 255)
            # prefer reasonable white ratio and many edges
            return float(e.sum()) / 255.0 - abs(ratio - 0.6) * 1e5

        candidates = [(otsu, edge_score(otsu)), (sau, edge_score(sau)), (ada, edge_score(ada))]
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[0][0]

    def _morphology_cleanup(self, img_bin: np.ndarray) -> np.ndarray:
        # Remove tiny noise and close small gaps
        kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        opened = cv2.morphologyEx(img_bin, cv2.MORPH_OPEN, kernel_open, iterations=1)
        kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel_close, iterations=1)
        return closed

    def _remove_lines(self, img_bin: np.ndarray) -> np.ndarray:
        try:
            inv = 255 - img_bin
            # Horizontal lines
            hori_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 1))
            detect_h = cv2.morphologyEx(inv, cv2.MORPH_OPEN, hori_kernel, iterations=1)
            # Vertical lines
            vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 30))
            detect_v = cv2.morphologyEx(inv, cv2.MORPH_OPEN, vert_kernel, iterations=1)
            lines = cv2.bitwise_or(detect_h, detect_v)
            # Subtract lines from inverted, then invert back
            cleaned_inv = cv2.subtract(inv, lines)
            cleaned = 255 - cleaned_inv
            return cleaned
        except Exception:
            return img_bin

    def _maybe_super_res(self, gray: np.ndarray) -> np.ndarray:
        h, w = gray.shape[:2]
        if min(h, w) < 700:
            return cv2.resize(gray, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
        return gray

class HybridDocumentProcessor:
    """
    Document processor with hybrid extraction system
    Intelligently combines adaptive and LLM methods for optimal results
    """
    
    def __init__(self, 
                 extraction_strategy: ExtractionStrategy = ExtractionStrategy.ADAPTIVE_FIRST,
                 llm_provider: LLMProvider = LLMProvider.OPENAI,
                 adaptive_confidence_threshold: float = 0.5,
                 min_pairs_threshold: int = 2,
                 enable_learning: bool = True,
                 llm_request_timeout_seconds: int = 40,
                 llm_max_retries: int = 2):
        
        # Initialize components
        try:
            configure_json_logging()
        except Exception:
            pass
        self.quality_checker = AdaptiveDocumentQualityChecker()
        self.ocr_processor = GoogleOCRProcessor()
        
        # Initialize hybrid extractor with configuration
        self.kv_extractor = HybridKeyValueExtractor(
            strategy=extraction_strategy,
            adaptive_confidence_threshold=adaptive_confidence_threshold,
            min_pairs_threshold=min_pairs_threshold,
            llm_provider=llm_provider,
            llm_request_timeout_seconds=llm_request_timeout_seconds,
            llm_max_retries=llm_max_retries,
            enable_learning=enable_learning
        )
        
        # Processing statistics
        self.processing_history = []
        self.total_processing_time = 0
        self.successful_extractions = 0

        # Metrics
        self._hist_total = Histogram("scanner_pipeline_seconds", "Total pipeline time in seconds")
        
    def process_image_bytes(self, image_bytes: bytes, document_type: str = "document") -> Dict:
        """Process image from bytes with hybrid extraction"""
        try:
            # Convert bytes to OpenCV image
            nparr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                raise ValueError("Invalid image format")
                
            return self.process_image(image, image_bytes, document_type)
            
        except Exception as e:
            logger.error(f"Hybrid image processing error: {e}")
            raise HTTPException(status_code=400, detail=f"Image processing failed: {e}")
    
    def process_image(self, image: np.ndarray, image_bytes: bytes, document_type: str = "document") -> Dict:
        """
        Complete hybrid processing pipeline
        """
        start_time = time.time()
        processing_audit = []
        
        processing_audit.append("🚀 Starting hybrid document processing")
        
        # Step 1: Adaptive Quality Assessment
        processing_audit.append("📋 Step 1: Quality assessment")
        quality_result = self.quality_checker.assess_quality(image, document_type=document_type)
        
        if quality_result["needs_rescan"]:
            processing_audit.append("❌ Quality insufficient - requesting rescan")
            rescan_decision = quality_result.get("rescan_decision", {})
            return {
                "status": "rescan_needed",
                "quality_assessment": quality_result,
                "message": rescan_decision.get("user_message", "Document quality is insufficient. Please rescan."),
                "rescan_reasons": rescan_decision.get("rescan_reasons", []),
                "rescan_urgency": rescan_decision.get("rescan_urgency", "medium"),
                "processing_audit": processing_audit,
                "hybrid_info": {
                    "quality_check_only": True,
                    "extraction_not_attempted": True
                }
            }
            
        processing_audit.append(f"✅ Quality OK (confidence: {quality_result['confidence']:.3f})")
        
        # Step 2: OCR Processing
        processing_audit.append("📄 Step 2: OCR text extraction")
        try:
            ocr_start = time.time()
            text_blocks, raw_text, preproc_png = self.ocr_processor.extract_text_with_bounds(image_bytes)
            ocr_time = time.time() - ocr_start
            
            processing_audit.append(f"✅ OCR completed: {len(text_blocks)} blocks, {len(raw_text)} chars in {ocr_time:.3f}s (preprocessed)")
            
            # Step 2.5: Enhanced quality assessment with text blocks for cut-off detection
            try:
                enhanced_quality_result = self.quality_checker.assess_quality(image, text_blocks=text_blocks, document_type=document_type)
                
                # If enhanced assessment finds new cut-off issues, handle accordingly
                if enhanced_quality_result["needs_rescan"] and not quality_result["needs_rescan"]:
                    processing_audit.append("⚠️ Enhanced quality check detected cut-off issues")
                    rescan_decision = enhanced_quality_result.get("rescan_decision", {})
                    return {
                        "status": "rescan_needed",
                        "quality_assessment": enhanced_quality_result,
                        "message": rescan_decision.get("user_message", "Cut-off issues detected. Please rescan."),
                        "rescan_reasons": rescan_decision.get("rescan_reasons", []),
                        "rescan_urgency": rescan_decision.get("rescan_urgency", "high"),
                        "processing_audit": processing_audit,
                        "enhanced_check_triggered": True
                    }
                elif enhanced_quality_result["confidence"] < quality_result["confidence"] - 0.1:
                    # Use enhanced result if significantly lower confidence
                    quality_result = enhanced_quality_result
                    processing_audit.append(f"📋 Updated quality assessment (confidence: {quality_result['confidence']:.3f})")
            except Exception as enhanced_qc_error:
                processing_audit.append(f"⚠️ Enhanced quality check failed: {enhanced_qc_error}")
                logger.warning(f"Enhanced quality check failed: {enhanced_qc_error}")
            
        except Exception as e:
            processing_audit.append(f"❌ OCR failed: {e}")
            return {
                "status": "ocr_failed",
                "error": str(e),
                "quality_assessment": quality_result,
                "processing_audit": processing_audit
            }
        
        # Step 3: Hybrid Key-Value Extraction
        processing_audit.append("🧠 Step 3: Hybrid key-value extraction")
        
        try:
            extraction_start = time.time()
            hybrid_result = self.kv_extractor.extract_key_value_pairs(
                text_blocks, raw_text, document_type
            )
            extraction_time = time.time() - extraction_start
            
            processing_audit.append(f"✅ Hybrid extraction completed in {extraction_time:.3f}s")
            processing_audit.extend(hybrid_result.audit_trail)
            
        except Exception as e:
            processing_audit.append(f"❌ Hybrid extraction failed: {e}")
            logger.error(f"Hybrid extraction error: {e}")
            return {
                "status": "extraction_failed",
                "error": str(e),
                "quality_assessment": quality_result,
                "processing_audit": processing_audit
            }
        
        # Step 4: Process Results and Statistics
        total_processing_time = time.time() - start_time
        self.total_processing_time += total_processing_time
        try:
            self._hist_total.observe(total_processing_time)
        except Exception:
            pass
        
        # Update success tracking
        if hybrid_result.pairs:
            self.successful_extractions += 1
        
        # Calculate comprehensive statistics
        extraction_stats = self._calculate_extraction_statistics(hybrid_result, ocr_time, extraction_time)
        
        # Store processing history
        self._store_processing_session(hybrid_result, quality_result, total_processing_time, document_type)
        
        # Learn from this session
        if hybrid_result.pairs:
            self._learn_from_session(hybrid_result, quality_result, document_type)
        
        processing_audit.append(f"📊 Final: {len(hybrid_result.pairs)} pairs extracted via {hybrid_result.primary_method}")
        
        # Prepare comprehensive response
        return {
            "status": "success",
            "quality_assessment": quality_result,
            "ocr_stats": {
                "text_blocks_count": len(text_blocks),
                "raw_text_length": len(raw_text),
                "processing_time_seconds": ocr_time
            },
            "preprocessed_image_b64": base64.b64encode(preproc_png).decode("ascii") if preproc_png else None,
            "key_value_pairs": self._format_extraction_results(hybrid_result.pairs),
            "extraction_metadata": {
                "primary_method": hybrid_result.primary_method,
                "fallback_used": hybrid_result.fallback_used,
                "extraction_time_seconds": hybrid_result.extraction_time_seconds,
                "confidence_scores": hybrid_result.confidence_scores,
                "method_comparison": hybrid_result.method_comparison,
                "strategy_used": self.kv_extractor.strategy.value
            },
            "llm_usage": hybrid_result.llm_usage,
            "processing_statistics": extraction_stats,
            "processing_time_seconds": total_processing_time,
            "processing_audit": processing_audit,
            "cut_off_analysis": quality_result.get("cut_off_analysis", {}),
            "rescan_decision": quality_result.get("rescan_decision", {}),
            "hybrid_system_info": {
                "adaptive_learning_enabled": self.kv_extractor.enable_learning,
                "llm_providers_available": list(self.kv_extractor.llm_extractor.available_providers.keys()),
                "current_strategy": self.kv_extractor.strategy.value,
                "confidence_threshold": self.kv_extractor.adaptive_confidence_threshold,
                "min_pairs_threshold": self.kv_extractor.min_pairs_threshold
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
            ],
            "raw_text_extracted": raw_text[:1000] + "..." if len(raw_text) > 1000 else raw_text  # Truncate for response size
        }
    
    def _format_extraction_results(self, pairs: List[Union[KeyValuePair, LLMKeyValuePair]]) -> List[Dict]:
        """Format extraction results for API response"""
        formatted_pairs = []
        
        for pair in pairs:
            # Handle both adaptive and LLM pairs
            if hasattr(pair, 'key_bbox') and hasattr(pair, 'value_bbox'):
                # Adaptive KeyValuePair with bounding boxes
                formatted_pair = {
                    "key": pair.key,
                    "value": pair.value,
                    "key_bbox": {
                        "x": pair.key_bbox.x,
                        "y": pair.key_bbox.y,
                        "width": pair.key_bbox.width,
                        "height": pair.key_bbox.height
                    },
                    "value_bbox": {
                        "x": pair.value_bbox.x,
                        "y": pair.value_bbox.y,
                        "width": pair.value_bbox.width,
                        "height": pair.value_bbox.height
                    },
                    "confidence": float(pair.confidence),
                    "extraction_method": pair.extraction_method,
                    "source": "adaptive"
                }
            else:
                # LLM KeyValuePair without bounding boxes
                formatted_pair = {
                    "key": pair.key,
                    "value": pair.value,
                    "key_bbox": None,  # LLM doesn't provide spatial info
                    "value_bbox": None,
                    "confidence": float(pair.confidence),
                    "extraction_method": pair.extraction_method,
                    "source": "llm",
                    "llm_provider": getattr(pair, 'llm_provider', 'unknown')
                }
            
            formatted_pairs.append(formatted_pair)
        
        return formatted_pairs
    
    def _calculate_extraction_statistics(self, hybrid_result: HybridExtractionResult, 
                                       ocr_time: float, extraction_time: float) -> Dict:
        """Calculate comprehensive extraction statistics"""
        
        pairs_count = len(hybrid_result.pairs)
        
        # Calculate confidence statistics
        if hybrid_result.pairs:
            confidences = [pair.confidence for pair in hybrid_result.pairs]
            confidence_stats = {
                "average_confidence": float(np.mean(confidences)),
                "min_confidence": float(np.min(confidences)),
                "max_confidence": float(np.max(confidences)),
                "std_confidence": float(np.std(confidences)),
                "high_confidence_pairs": sum(1 for c in confidences if c > 0.7),
                "low_confidence_pairs": sum(1 for c in confidences if c < 0.4)
            }
        else:
            confidence_stats = {
                "average_confidence": 0.0,
                "min_confidence": 0.0,
                "max_confidence": 0.0,
                "std_confidence": 0.0,
                "high_confidence_pairs": 0,
                "low_confidence_pairs": 0
            }
        
        # Method analysis
        method_stats = {
            "primary_method_used": hybrid_result.primary_method,
            "fallback_triggered": hybrid_result.fallback_used,
            "extraction_strategy": self.kv_extractor.strategy.value,
            "method_confidence_scores": hybrid_result.confidence_scores
        }
        
        # Performance metrics
        performance_stats = {
            "total_pairs_extracted": pairs_count,
            "extraction_time_seconds": extraction_time,
            "ocr_time_seconds": ocr_time,
            "pairs_per_second": pairs_count / max(extraction_time, 0.001),
            "total_session_count": len(self.processing_history) + 1,
            "lifetime_success_rate": self.successful_extractions / max(len(self.processing_history) + 1, 1)
        }
        
        return {
            "confidence_analysis": confidence_stats,
            "method_analysis": method_stats,
            "performance_metrics": performance_stats,
            "hybrid_system_performance": self.kv_extractor.get_performance_statistics()
        }
    
    def _store_processing_session(self, hybrid_result: HybridExtractionResult, 
                                quality_result: Dict, processing_time: float, document_type: str):
        """Store session data for learning and analytics"""
        
        session_data = {
            "timestamp": time.time(),
            "document_type": document_type,
            "pairs_extracted": len(hybrid_result.pairs),
            "primary_method": hybrid_result.primary_method,
            "fallback_used": hybrid_result.fallback_used,
            "average_confidence": np.mean([p.confidence for p in hybrid_result.pairs]) if hybrid_result.pairs else 0.0,
            "quality_confidence": quality_result.get("confidence", 0.0),
            "processing_time": processing_time,
            "extraction_time": hybrid_result.extraction_time_seconds,
            "confidence_scores": hybrid_result.confidence_scores,
            "strategy_used": self.kv_extractor.strategy.value
        }
        
        self.processing_history.append(session_data)
        
        # Limit history size
        if len(self.processing_history) > 100:
            self.processing_history = self.processing_history[-100:]
    
    def _learn_from_session(self, hybrid_result: HybridExtractionResult, 
                          quality_result: Dict, document_type: str):
        """Learn from successful processing sessions"""
        
        try:
            # Let the adaptive config learn from overall session results
            session_results = {
                "average_confidence": np.mean([p.confidence for p in hybrid_result.pairs]) if hybrid_result.pairs else 0.0,
                "quality_assessment": quality_result,
                "extraction_statistics": {
                    "total_pairs": len(hybrid_result.pairs),
                    "high_confidence_pairs": sum(1 for p in hybrid_result.pairs if p.confidence > 0.7),
                    "extraction_methods_used": [hybrid_result.primary_method]
                }
            }
            
            adaptive_config.adapt_from_document_processing(session_results)
            
            # The hybrid extractor handles its own learning internally
            
            logger.info(f"Learning completed for {document_type} document processing")
            
        except Exception as e:
            logger.debug(f"Session learning error: {e}")
    
    def get_processing_analytics(self) -> Dict:
        """Get comprehensive processing analytics"""
        
        if not self.processing_history:
            return {"status": "no_data", "message": "No processing sessions recorded yet"}
        
        recent_sessions = self.processing_history[-20:]
        
        # Overall performance
        avg_pairs = np.mean([s["pairs_extracted"] for s in recent_sessions])
        avg_confidence = np.mean([s["average_confidence"] for s in recent_sessions])
        avg_processing_time = np.mean([s["processing_time"] for s in recent_sessions])
        
        # Method distribution
        method_distribution = {}
        for session in recent_sessions:
            method = session["primary_method"]
            method_distribution[method] = method_distribution.get(method, 0) + 1
        
        # Strategy effectiveness
        strategy_performance = {}
        for session in recent_sessions:
            strategy = session["strategy_used"]
            if strategy not in strategy_performance:
                strategy_performance[strategy] = {"sessions": 0, "total_pairs": 0, "total_confidence": 0.0}
            
            strategy_performance[strategy]["sessions"] += 1
            strategy_performance[strategy]["total_pairs"] += session["pairs_extracted"]
            strategy_performance[strategy]["total_confidence"] += session["average_confidence"]
        
        # Calculate averages
        for strategy, stats in strategy_performance.items():
            if stats["sessions"] > 0:
                stats["avg_pairs"] = stats["total_pairs"] / stats["sessions"]
                stats["avg_confidence"] = stats["total_confidence"] / stats["sessions"]
        
        # Trend analysis
        if len(self.processing_history) >= 10:
            early_sessions = self.processing_history[:5]
            recent_sessions_trend = self.processing_history[-5:]
            
            early_avg_pairs = np.mean([s["pairs_extracted"] for s in early_sessions])
            recent_avg_pairs = np.mean([s["pairs_extracted"] for s in recent_sessions_trend])
            
            early_avg_conf = np.mean([s["average_confidence"] for s in early_sessions])
            recent_avg_conf = np.mean([s["average_confidence"] for s in recent_sessions_trend])
            
            trend_analysis = {
                "pairs_improvement": recent_avg_pairs - early_avg_pairs,
                "confidence_improvement": recent_avg_conf - early_avg_conf,
                "trend_direction": "improving" if (recent_avg_pairs > early_avg_pairs and recent_avg_conf > early_avg_conf) else "stable"
            }
        else:
            trend_analysis = {"status": "insufficient_data_for_trends"}
        
        return {
            "total_sessions": len(self.processing_history),
            "recent_performance": {
                "avg_pairs_extracted": float(avg_pairs),
                "avg_confidence": float(avg_confidence),
                "avg_processing_time": float(avg_processing_time),
                "success_rate": self.successful_extractions / len(self.processing_history)
            },
            "method_distribution": method_distribution,
            "strategy_performance": strategy_performance,
            "trend_analysis": trend_analysis,
            "current_configuration": {
                "strategy": self.kv_extractor.strategy.value,
                "confidence_threshold": self.kv_extractor.adaptive_confidence_threshold,
                "min_pairs_threshold": self.kv_extractor.min_pairs_threshold,
                "learning_enabled": self.kv_extractor.enable_learning
            },
            "hybrid_system_analytics": self.kv_extractor.get_performance_statistics()
        }
    
    def optimize_extraction_strategy(self) -> Dict:
        """Analyze performance and optimize extraction strategy"""
        
        try:
            # Get current strategy recommendation from hybrid extractor
            recommended_strategy = self.kv_extractor.optimize_strategy()
            current_strategy = self.kv_extractor.strategy
            
            analytics = self.get_processing_analytics()
            
            optimization_result = {
                "current_strategy": current_strategy.value,
                "recommended_strategy": recommended_strategy.value,
                "strategy_changed": recommended_strategy != current_strategy,
                "performance_analysis": analytics.get("strategy_performance", {}),
                "optimization_reasoning": []
            }
            
            # Update strategy if recommended
            if recommended_strategy != current_strategy:
                self.kv_extractor.set_strategy(recommended_strategy)
                optimization_result["optimization_reasoning"].append(
                    f"Strategy updated from {current_strategy.value} to {recommended_strategy.value} based on performance analysis"
                )
                logger.info(f"Extraction strategy optimized: {current_strategy.value} -> {recommended_strategy.value}")
            else:
                optimization_result["optimization_reasoning"].append(
                    f"Current strategy {current_strategy.value} is performing optimally"
                )
            
            return optimization_result
            
        except Exception as e:
            logger.error(f"Strategy optimization error: {e}")
            return {
                "status": "optimization_failed",
                "error": str(e),
                "current_strategy": self.kv_extractor.strategy.value
            }
    
    def update_configuration(self, 
                           strategy: ExtractionStrategy = None,
                           confidence_threshold: float = None,
                           min_pairs_threshold: int = None,
                           llm_provider: LLMProvider = None) -> Dict:
        """Update hybrid extractor configuration"""
        
        updates = []
        
        try:
            if strategy is not None:
                self.kv_extractor.set_strategy(strategy)
                updates.append(f"Strategy updated to {strategy.value}")
            
            if confidence_threshold is not None:
                self.kv_extractor.adaptive_confidence_threshold = confidence_threshold
                updates.append(f"Confidence threshold updated to {confidence_threshold}")
            
            if min_pairs_threshold is not None:
                self.kv_extractor.min_pairs_threshold = min_pairs_threshold
                updates.append(f"Min pairs threshold updated to {min_pairs_threshold}")
            
            if llm_provider is not None:
                # Reinitialize LLM extractor with new provider
                self.kv_extractor.llm_extractor = LLMKeyValueExtractor(primary_provider=llm_provider)
                updates.append(f"LLM provider updated to {llm_provider.value}")
            
            logger.info(f"Configuration updated: {', '.join(updates)}")
            
            return {
                "status": "configuration_updated",
                "updates": updates,
                "current_configuration": {
                    "strategy": self.kv_extractor.strategy.value,
                    "confidence_threshold": self.kv_extractor.adaptive_confidence_threshold,
                    "min_pairs_threshold": self.kv_extractor.min_pairs_threshold,
                    "llm_provider": self.kv_extractor.llm_extractor.primary_provider.value
                }
            }
            
        except Exception as e:
            logger.error(f"Configuration update error: {e}")
            return {
                "status": "configuration_update_failed",
                "error": str(e)
            }