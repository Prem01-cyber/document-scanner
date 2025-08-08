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
    
    def extract_text_with_bounds(self, image_bytes: bytes, preprocess_options: dict | None = None):
        """Extract text with bounding boxes returned in PREPROCESSED image coordinates.
        Returns (blocks, words, raw_text, preprocessed_png_bytes).
        """
        try:
            from google.cloud import vision
            from .adaptive_kv_extractor import TextBlock, BoundingBox
            
            # Preprocess image before OCR
            nparr = np.frombuffer(image_bytes, np.uint8)
            img_cv = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img_cv is None:
                raise Exception("Invalid image data for OCR")

            processed, H_src_to_proc = self._preprocess_for_ocr(img_cv, preprocess_options)
            success, processed_buf = cv2.imencode('.png', processed)
            if not success:
                raise Exception("Failed to encode preprocessed image")
            processed_png_bytes = processed_buf.tobytes()
            image = vision.Image(content=processed_png_bytes)
            response = self.client.document_text_detection(image=image)
            
            if response.error.message:
                raise Exception(f"OCR Error: {response.error.message}")
            
            text_blocks = []
            word_items = []
            raw_text_parts = []
            
            # Process structured blocks for adaptive extraction (coords in processed space)
            for page in response.full_text_annotation.pages:
                for block in page.blocks:
                    block_text = ""
                    
                    for paragraph in block.paragraphs:
                        for word in paragraph.words:
                            word_text = "".join([symbol.text for symbol in word.symbols])
                            block_text += word_text + " "
                            # Word-level bbox in processed space
                            w_vertices = word.bounding_box.vertices
                            wxs = np.array([v.x for v in w_vertices], dtype=np.float32)
                            wys = np.array([v.y for v in w_vertices], dtype=np.float32)
                            h_p, w_p = processed.shape[:2]
                            wx_min = int(np.clip(wxs.min(), 0, w_p - 1))
                            wy_min = int(np.clip(wys.min(), 0, h_p - 1))
                            wx_max = int(np.clip(wxs.max(), 0, w_p - 1))
                            wy_max = int(np.clip(wys.max(), 0, h_p - 1))
                            word_items.append({
                                "text": word_text.strip(),
                                "bbox": {
                                    "x": int(wx_min),
                                    "y": int(wy_min),
                                    "width": int(max(1, wx_max - wx_min)),
                                    "height": int(max(1, wy_max - wy_min))
                                }
                            })
                    
                    # Bounding box is already in processed image space
                    vertices = block.bounding_box.vertices
                    xs = np.array([v.x for v in vertices], dtype=np.float32)
                    ys = np.array([v.y for v in vertices], dtype=np.float32)
                    h_p, w_p = processed.shape[:2]
                    x_min = int(np.clip(xs.min(), 0, w_p - 1))
                    y_min = int(np.clip(ys.min(), 0, h_p - 1))
                    x_max = int(np.clip(xs.max(), 0, w_p - 1))
                    y_max = int(np.clip(ys.max(), 0, h_p - 1))

                    bbox = BoundingBox(
                        x=x_min,
                        y=y_min,
                        width=max(1, x_max - x_min),
                        height=max(1, y_max - y_min)
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
            
            return text_blocks, word_items, raw_text, processed_png_bytes
            
        except Exception as e:
            logger.error(f"OCR processing error: {e}")
            raise HTTPException(status_code=500, detail=f"OCR failed: {e}")

    def _preprocess_for_ocr(self, image_bgr: np.ndarray, options: dict | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Apply perspective correction, denoise, illumination normalization, deskew, advanced binarization,
        morphology cleanup, and line removal to improve OCR robustness.
        Returns a single-channel 8-bit preprocessed image.
        """
        # Default options
        opts = options or {}
        auto = bool(opts.get("auto_mode", True))
        enable_persp = bool(opts.get("enable_perspective", True)) if not auto else True
        enable_lines = bool(opts.get("enable_line_removal", True)) if not auto else True
        enable_sharp = bool(opts.get("enable_sharpen", True)) if not auto else True

        # 1) Grayscale
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

        # 2) (Optional) Perspective correction by detecting document contour
        corrected, H_src_to_corr = self._perspective_correction(gray) if enable_persp else (gray, np.eye(3, dtype=np.float32))

        # 3) If very small, upscale slightly for better readability
        corrected = self._maybe_super_res(corrected)

        # 4) Denoise (fast non-local means) and illumination normalization
        denoised = cv2.fastNlMeansDenoising(corrected, h=10, templateWindowSize=7, searchWindowSize=21)
        norm = self._illumination_normalize(denoised)

        # 5) Light sharpening (unsharp mask)
        sharp = self._unsharp_mask(norm, amount=1.0, threshold=3) if enable_sharp else norm

        # 6) Estimate skew and deskew
        deskewed, H_corr_to_desk = self._deskew_by_hough(sharp)

        # 7) Advanced binarization: compare Otsu, Sauvola, and adaptive Gaussian; choose best by edge metric
        binary = self._advanced_binarization(deskewed)

        # 8) Morphological cleanup
        cleaned = self._morphology_cleanup(binary)

        # 9) Line removal (table lines) to avoid interfering with OCR
        no_lines = self._remove_lines(cleaned) if enable_lines else cleaned

        # Compose homography from source to processed for bbox back-mapping
        H_src_to_proc = H_corr_to_desk @ H_src_to_corr
        return no_lines, H_src_to_proc

    def _perspective_correction(self, gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        img = gray
        H = np.eye(3, dtype=np.float32)
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
                warped, H = self._four_point_transform(gray, pts)
                return warped, H
        except Exception:
            pass
        return img, H

    def _four_point_transform(self, img: np.ndarray, pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
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
        return warped, M

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

    def _deskew_by_hough(self, gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
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
            warped = cv2.warpAffine(gray, rot_mat, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            # Promote to 3x3 homography
            H = np.array([[rot_mat[0,0], rot_mat[0,1], rot_mat[0,2]],
                          [rot_mat[1,0], rot_mat[1,1], rot_mat[1,2]],
                          [0, 0, 1]], dtype=np.float32)
            return warped, H
        return gray, np.eye(3, dtype=np.float32)

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
        
    def process_image_bytes(self, image_bytes: bytes, document_type: str = "document", preprocess_options: dict | None = None, roi: dict | None = None, append_mode: bool = False, previous_pairs: List[Dict] | None = None) -> Dict:
        """Process image from bytes with hybrid extraction"""
        try:
            # Convert bytes to OpenCV image
            nparr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                raise ValueError("Invalid image format")
                
            return self.process_image(image, image_bytes, document_type, preprocess_options, roi, append_mode, previous_pairs)
            
        except Exception as e:
            logger.error(f"Hybrid image processing error: {e}")
            raise HTTPException(status_code=400, detail=f"Image processing failed: {e}")
    
    def process_image(self, image: np.ndarray, image_bytes: bytes, document_type: str = "document", preprocess_options: dict | None = None, roi: dict | None = None, append_mode: bool = False, previous_pairs: List[Dict] | None = None) -> Dict:
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
        
        # Optional ROI crop for focused OCR
        full_image_for_ocr = image
        roi_info = None
        if roi:
            try:
                x, y, w, h = int(roi.get("x", 0)), int(roi.get("y", 0)), int(roi.get("width", 0)), int(roi.get("height", 0))
                h_img, w_img = image.shape[:2]
                x = max(0, min(x, w_img - 1))
                y = max(0, min(y, h_img - 1))
                w = max(1, min(w, w_img - x))
                h = max(1, min(h, h_img - y))
                full_image_for_ocr = image[y:y+h, x:x+w]
                roi_info = {"x": x, "y": y, "width": w, "height": h}
                processing_audit.append(f"🎯 ROI applied: x={x}, y={y}, w={w}, h={h}")
            except Exception as e:
                processing_audit.append(f"⚠️ ROI invalid, using full image: {e}")

        # Step 2: OCR Processing
        processing_audit.append("📄 Step 2: OCR text extraction")
        try:
            ocr_start = time.time()
            if roi_info is not None:
                ok, buf = cv2.imencode('.png', full_image_for_ocr)
                if not ok:
                    raise Exception("Failed to encode ROI for OCR")
                roi_bytes = buf.tobytes()
                text_blocks, word_items, raw_text, preproc_png = self.ocr_processor.extract_text_with_bounds(roi_bytes, preprocess_options or {})
                # Offset OCR bbox by ROI origin
                for tb in text_blocks:
                    tb.bbox.x += roi_info["x"]
                    tb.bbox.y += roi_info["y"]
                for wi in word_items:
                    wi["bbox"]["x"] += roi_info["x"]
                    wi["bbox"]["y"] += roi_info["y"]
            else:
                text_blocks, word_items, raw_text, preproc_png = self.ocr_processor.extract_text_with_bounds(image_bytes, preprocess_options or {})
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
        
        # Merge mode: append new pairs to previous
        formatted_pairs = self._format_extraction_results(hybrid_result.pairs, text_blocks, word_items)
        if append_mode and previous_pairs:
            existing = {(p.get("key",""), p.get("value","")) for p in previous_pairs}
            for fp in formatted_pairs:
                tup = (fp.get("key",""), fp.get("value",""))
                if tup not in existing:
                    previous_pairs.append(fp)
            formatted_pairs = previous_pairs

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
            "key_value_pairs": formatted_pairs,
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
            "extracted_words": word_items,
            "raw_text_extracted": raw_text[:1000] + "..." if len(raw_text) > 1000 else raw_text  # Truncate for response size
        }
    
    def _format_extraction_results(self, pairs: List[Union[KeyValuePair, LLMKeyValuePair]], text_blocks=None, word_items=None) -> List[Dict]:
        """Format extraction results for API response, assigning approximate
        bounding boxes for LLM-only pairs by aligning to OCR text blocks when available."""
        formatted_pairs = []
        text_blocks = text_blocks or []
        word_items = word_items or []

        # Lightweight matcher to map a string to the best OCR block
        def _normalize(s: str) -> str:
            return ''.join(ch.lower() for ch in str(s) if ch.isalnum() or ch.isspace()).strip()

        def _digits_only(s: str) -> str:
            return ''.join(ch for ch in str(s) if ch.isdigit())

        def _block_center(blk):
            return (blk.bbox.x + blk.bbox.width/2.0, blk.bbox.y + blk.bbox.height/2.0)

        def _best_block_for_text(query: str, near_block=None, image_wh=None, prefer_below_or_right=False):
            if not query or not text_blocks:
                return None
            q = _normalize(query)
            q_tokens = {t for t in q.replace('*', '').split() if t}
            q_digits = _digits_only(query)
            best = None
            best_score = 0.0
            for blk in text_blocks:
                try:
                    blk_text_raw = str(blk.text)
                    blk_text = _normalize(blk_text_raw)
                    blk_tokens = {t for t in blk_text.split() if t}
                    score = 0.0
                    # direct substring/containment
                    if q and (q in blk_text or blk_text in q):
                        score = max(len(q), len(blk_text)) / (len(q) + len(blk_text) + 1e-6)
                    # token overlap score
                    if q_tokens and blk_tokens:
                        inter = len(q_tokens & blk_tokens)
                        union = len(q_tokens | blk_tokens)
                        score = max(score, inter / max(union, 1))
                    # numeric exact match boost
                    blk_digits = _digits_only(blk_text_raw)
                    if q_digits and q_digits == blk_digits:
                        score = max(score, 1.0)
                    # partial digit match for dates, IDs, etc.
                    elif q_digits and blk_digits and len(q_digits) >= 3 and len(blk_digits) >= 3:
                        common_digits = sum(1 for a, b in zip(q_digits, blk_digits) if a == b)
                        if common_digits >= min(3, len(q_digits) // 2):
                            score = max(score, 0.6)
                    # fuzzy matching for names and similar text
                    if len(q) >= 3 and len(blk_text) >= 3:
                        # Simple character similarity
                        common_chars = sum(1 for c in q if c in blk_text)
                        if common_chars >= min(3, len(q) // 2):
                            char_score = common_chars / max(len(q), len(blk_text))
                            score = max(score, char_score * 0.4)
                    # proximity bonus if a reference block is provided
                    if near_block is not None and image_wh is not None:
                        cx_ref, cy_ref = _block_center(near_block)
                        cx, cy = _block_center(blk)
                        w, h = image_wh
                        # normalized distance in [0, sqrt(2)]
                        dist = ((cx - cx_ref)**2 + (cy - cy_ref)**2) ** 0.5 / max((w**2 + h**2) ** 0.5, 1)
                        prox = 1.0 - min(max(dist, 0.0), 1.0)
                        score += 0.35 * prox
                        # prefer blocks to the right or just below the key label
                        if prefer_below_or_right and (cx >= cx_ref - 5 or cy >= cy_ref - 5):
                            score += 0.05
                except Exception:
                    continue
                if score > best_score:
                    best_score = score
                    best = blk
            # require a minimal score to avoid random boxes
            # Lowered threshold from 0.38 to 0.20 to be more permissive for LLM-extracted text
            if best is not None and best_score >= 0.20:
                logger.debug(f"Text mapping: '{query}' -> '{best.text}' (score: {best_score:.3f})")
                return best
            if best is not None:
                logger.debug(f"Text mapping rejected: '{query}' -> '{best.text}' (score: {best_score:.3f} < 0.20)")
            else:
                logger.debug(f"Text mapping failed: '{query}' -> no suitable block found")
            return None

        # Try to refine a coarse block using word spans that best cover the query
        def _refine_bbox_with_words(query: str, coarse_block):
            if not query or not word_items:
                return coarse_block
            q = _normalize(query)
            if not q:
                return coarse_block
            words_in_block = []
            bx0 = coarse_block.bbox.x
            by0 = coarse_block.bbox.y
            bx1 = bx0 + coarse_block.bbox.width
            by1 = by0 + coarse_block.bbox.height
            for wi in word_items:
                wb = wi["bbox"]
                wx0, wy0 = wb["x"], wb["y"]
                wx1, wy1 = wx0 + wb["width"], wy0 + wb["height"]
                if wx0 >= bx0 and wy0 >= by0 and wx1 <= bx1 and wy1 <= by1:
                    words_in_block.append((wi["text"], wb))
            if not words_in_block:
                return coarse_block
            # greedy cover by words
            selected = []
            rem = q.split()
            for t, wb in words_in_block:
                nt = _normalize(t)
                if not nt:
                    continue
                hit = False
                for tok in list(rem):
                    if tok and (tok in nt or nt in tok):
                        rem.remove(tok)
                        hit = True
                if hit:
                    selected.append(wb)
                if not rem:
                    break
            if not selected:
                return coarse_block
            # build tight bbox
            xs0 = min(wb["x"] for wb in selected)
            ys0 = min(wb["y"] for wb in selected)
            xs1 = max(wb["x"] + wb["width"] for wb in selected)
            ys1 = max(wb["y"] + wb["height"] for wb in selected)
            class _BBox:
                def __init__(self, x, y, w, h):
                    self.x=x; self.y=y; self.width=w; self.height=h
            class _Blk:
                def __init__(self, bbox):
                    self.bbox=bbox
            return _Blk(_BBox(xs0, ys0, xs1 - xs0, ys1 - ys0))
        
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
                # First anchor the key by text similarity
                key_blk = _best_block_for_text(getattr(pair, 'key', ''))
                # Then search for the value with both text match and proximity to the key (if found)
                img_wh = None
                try:
                    # infer overall image width/height from available blocks
                    if text_blocks:
                        max_right = max(tb.bbox.x + tb.bbox.width for tb in text_blocks)
                        max_bottom = max(tb.bbox.y + tb.bbox.height for tb in text_blocks)
                        img_wh = (max_right, max_bottom)
                except Exception:
                    img_wh = None
                val_blk = _best_block_for_text(getattr(pair, 'value', ''), near_block=key_blk, image_wh=img_wh, prefer_below_or_right=True)
                # tighten boxes with word spans when possible
                if key_blk is not None:
                    key_blk = _refine_bbox_with_words(getattr(pair, 'key', ''), key_blk)
                if val_blk is not None:
                    val_blk = _refine_bbox_with_words(getattr(pair, 'value', ''), val_blk)
                def _bbox_dict(blk):
                    return {
                        "x": blk.bbox.x,
                        "y": blk.bbox.y,
                        "width": blk.bbox.width,
                        "height": blk.bbox.height
                    } if blk is not None else None
                formatted_pair = {
                    "key": pair.key,
                    "value": pair.value,
                    "key_bbox": _bbox_dict(key_blk),
                    "value_bbox": _bbox_dict(val_blk),
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