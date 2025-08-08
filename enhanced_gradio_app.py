#!/usr/bin/env python3
"""
Enhanced Gradio UI for Hybrid Document Scanner with Quality Assessment
Features both rule-based risk scoring and ML classification
"""

import gradio as gr
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import io
import json
import time
from typing import List, Dict, Optional, Tuple
import logging
from datetime import datetime
import traceback
from dotenv import load_dotenv
import base64

# Load environment variables from .env file
load_dotenv()

from src.hybrid_document_processor import HybridDocumentProcessor
from src.hybrid_kv_extractor import ExtractionStrategy
from src.llm_kv_extractor import LLMProvider
from quality.quality_data_collector import quality_data_collector

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import visual overlay for quality assessment
try:
    from quality.visual_overlay import create_quality_overlay
    VISUAL_OVERLAY_AVAILABLE = True
except ImportError:
    VISUAL_OVERLAY_AVAILABLE = False
    logger.warning("Visual overlay not available")

class EnhancedDocumentProcessor:
    """Enhanced document processor with quality assessment UI"""
    
    def __init__(self):
        self.processor = None
        self.processor_status = "❌ Not initialized"
        self.system_ready = False
        
        # Statistics
        self.stats = {
            "total_processed": 0,
            "successful": 0,
            "failed": 0,
            "total_pairs": 0,
            "avg_processing_time": 0.0,
            "last_processed": None,
            "quality_assessments": 0,
            "rescans_needed": 0
        }
        
        # Initialize processor
        self._initialize_processor()
    
    def _initialize_processor(self):
        """Initialize the document processor with error handling"""
        try:
            self.processor = HybridDocumentProcessor(
                extraction_strategy=ExtractionStrategy.ADAPTIVE_FIRST,
                enable_learning=True
            )
            self.processor_status = "✅ System Ready"
            self.system_ready = True
            logger.info("Document processor initialized successfully")
        except Exception as e:
            self.processor_status = f"❌ Initialization failed: {str(e)[:50]}..."
            self.system_ready = False
            logger.error(f"Failed to initialize processor: {e}")
    
    def process_document_with_quality(self, image, strategy="adaptive_first", confidence=0.5, llm_provider="ollama",
                                     roi: Dict = None, auto_mode: bool = True, enable_perspective: bool = True,
                                     enable_line_removal: bool = True, enable_sharpen: bool = True,
                                     append_mode: bool = False, previous_pairs: List[Dict] = None):
        """Process document with comprehensive quality assessment"""
        start_time = time.time()
        session_id = f"gradio_{int(time.time())}"
        
        # Update stats
        self.stats["total_processed"] += 1
        self.stats["last_processed"] = datetime.now().strftime("%H:%M:%S")
        
        if not self.system_ready:
            self.stats["failed"] += 1
            return self._create_error_response("System not ready. Please check status.")
        
        if image is None:
            self.stats["failed"] += 1
            return self._create_error_response("No image provided. Please upload a document.")
        
        try:
            # Validate image
            if hasattr(image, 'size'):
                if image.size[0] < 100 or image.size[1] < 100:
                    self.stats["failed"] += 1
                    return self._create_error_response("Image too small. Please upload a larger image (min 100x100).")
            
            # Convert image to bytes
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format='PNG')
            image_bytes = img_byte_arr.getvalue()
            
            # Configure processor
            try:
                strategy_enum = ExtractionStrategy(strategy)
                self.processor.kv_extractor.set_strategy(strategy_enum)
                self.processor.kv_extractor.adaptive_confidence_threshold = confidence
                
                # Set LLM provider if using LLM strategies
                if strategy in ["llm_first", "parallel", "confidence_based"]:
                    try:
                        provider_enum = LLMProvider(llm_provider)
                        self.processor.kv_extractor.llm_extractor.primary_provider = provider_enum
                    except (ValueError, AttributeError) as llm_error:
                        logger.warning(f"Could not set LLM provider {llm_provider}: {llm_error}")
                        
            except ValueError as e:
                self.stats["failed"] += 1
                return self._create_error_response(f"Invalid strategy: {strategy}")
            
            # Process document
            logger.info(f"🔄 Processing document with strategy: {strategy}, provider: {llm_provider}")
            preprocess_options = {
                "auto_mode": bool(auto_mode),
                "enable_perspective": bool(enable_perspective),
                "enable_line_removal": bool(enable_line_removal),
                "enable_sharpen": bool(enable_sharpen)
            }
            result = self.processor.process_image_bytes(
                image_bytes,
                "document",
                preprocess_options=preprocess_options,
                roi=roi,
                append_mode=bool(append_mode),
                previous_pairs=previous_pairs or []
            )
            processing_time = time.time() - start_time
            
            # Update quality stats
            self.stats["quality_assessments"] += 1
            if result.get("quality_assessment", {}).get("needs_rescan", False):
                self.stats["rescans_needed"] += 1
            
            # Log quality assessment data
            try:
                if result.get("quality_assessment"):
                    quality_assessment = result["quality_assessment"]
                    quality_metrics = quality_assessment.get("quality_risk_assessment", {}).get("quality_metrics", {})
                    rule_result = quality_assessment.get("quality_risk_assessment", {})
                    ml_result = quality_assessment.get("ml_assessment", {})
                    
                    quality_data_collector.log_quality_assessment(
                        quality_metrics=quality_metrics,
                        rule_based_result=rule_result,
                        ml_result=ml_result if ml_result.get("ml_available") else None,
                        document_type="document",
                        session_id=session_id
                    )
            except Exception as e:
                logger.warning(f"Failed to log quality assessment: {e}")
                # Don't fail the entire process for logging issues
            
            # Update processing time average
            total_time = self.stats["avg_processing_time"] * (self.stats["total_processed"] - 1) + processing_time
            self.stats["avg_processing_time"] = total_time / self.stats["total_processed"]
            
            if result["status"] == "success":
                self.stats["successful"] += 1
                pairs_count = len(result.get("key_value_pairs", []))
                self.stats["total_pairs"] += pairs_count
                
                return self._create_success_response(result, processing_time, image, session_id)
            else:
                self.stats["failed"] += 1
                error_msg = result.get("error", "Unknown processing error")
                return self._create_error_response(f"Processing failed: {error_msg}")
                
        except Exception as e:
            self.stats["failed"] += 1
            logger.error(f"Document processing error: {e}")
            return self._create_error_response(f"Unexpected error: {str(e)[:100]}...")
    
    def _create_success_response(self, result, processing_time, original_image, session_id):
        """Create a success response with quality assessment"""
        all_pairs = result.get("key_value_pairs", [])
        confidence_threshold = self.processor.kv_extractor.adaptive_confidence_threshold
        filtered_pairs = [pair for pair in all_pairs if pair.get("confidence", 0) >= confidence_threshold]

        # Decode preprocessed image (OCR input) if available
        pre_b64 = result.get("preprocessed_image_b64")
        pre_image = None
        if pre_b64:
            try:
                pre_image = Image.open(io.BytesIO(base64.b64decode(pre_b64)))
            except Exception:
                pre_image = None

        # Create annotated image on the preprocessed canvas when available,
        # because Google OCR bounding boxes are relative to that image
        base_image_for_annotation = pre_image if pre_image is not None else original_image
        annotated_image = self._create_enhanced_annotation(base_image_for_annotation, filtered_pairs, confidence_threshold)
        
        # Create quality overlay image if available
        quality_overlay_image = None
        if VISUAL_OVERLAY_AVAILABLE and result.get("quality_assessment"):
            try:
                # Convert PIL to numpy for overlay processing
                image_array = np.array(original_image)
                text_blocks = result.get("text_blocks", [])
                quality_overlay_array = create_quality_overlay(
                    image_array, result["quality_assessment"], text_blocks
                )
                quality_overlay_image = Image.fromarray(cv2.cvtColor(quality_overlay_array, cv2.COLOR_BGR2RGB))
            except Exception as e:
                logger.warning(f"Failed to create quality overlay: {e}")
                quality_overlay_image = annotated_image
        
        # Create table data
        table_data = []
        for i, pair in enumerate(filtered_pairs, 1):
            confidence = pair.get("confidence", 0)
            confidence_str = f"{confidence:.2f}"
            if confidence > 0.8:
                confidence_str += " 🟢"
            elif confidence > 0.5:
                confidence_str += " 🟡"
            else:
                confidence_str += " 🔴"
            
            table_data.append([
                i,
                pair.get("key", "")[:30],
                pair.get("value", "")[:40], 
                confidence_str,
                pair.get("source", "unknown")
            ])
        
        # Format quality assessment
        quality_info = self._format_quality_assessment(result.get("quality_assessment", {}))
        
        # Create comprehensive summary
        total_pairs = len(all_pairs)
        shown_pairs = len(filtered_pairs)
        summary = f"""### ✅ Processing Successful

**⏱️ Processing Time:** {processing_time:.2f} seconds  
**📊 Total Pairs Found:** {total_pairs}  
**✅ Pairs Shown (≥{confidence_threshold:.1f} confidence):** {shown_pairs}  
**🎯 Strategy:** {result.get('extraction_metadata', {}).get('strategy_used', 'unknown')}

{quality_info}

💡 **Tip:** Provide quality feedback to help improve our ML model!
"""
        
        return {
            "success": True,
            "image": annotated_image,
            "quality_overlay": quality_overlay_image or annotated_image,
            "preprocessed": pre_image,
            "table": table_data,
            "summary": summary,
            "quality_assessment": result.get("quality_assessment", {}),
            "session_id": session_id,
            "status": f"✅ Found {total_pairs} pairs, showing {shown_pairs}"
        }
    
    def _create_error_response(self, message):
        """Create an error response"""
        return {
            "success": False,
            "image": None,
            "table": [],
            "summary": f"### ❌ Processing Failed\\n\\n**Error:** {message}",
            "quality_assessment": {},
            "session_id": "",
            "status": f"❌ {message}"
        }
    
    def _format_quality_assessment(self, quality_assessment):
        """Format quality assessment for display"""
        if not quality_assessment:
            return "**📊 Quality Assessment:** Not available"
        
        confidence = quality_assessment.get("confidence", 0)
        needs_rescan = quality_assessment.get("needs_rescan", False)
        issues = quality_assessment.get("issues", [])
        
        # Quality status icon
        if needs_rescan:
            status_icon = "🔴"
            status_text = "Needs Rescan"
        elif confidence > 0.8:
            status_icon = "🟢"
            status_text = "Excellent Quality"
        elif confidence > 0.6:
            status_icon = "🟡"
            status_text = "Good Quality"
        else:
            status_icon = "🟠"
            status_text = "Marginal Quality"
        
        quality_info = f"""**📊 Quality Assessment:**
- **Status:** {status_icon} {status_text}
- **Confidence:** {confidence:.2f}
- **Issues:** {len(issues)} detected"""
        
        # Risk assessment details
        risk_assessment = quality_assessment.get("quality_risk_assessment", {})
        if risk_assessment:
            risk_score = risk_assessment.get("quality_risk_score", 0)
            risk_decision = risk_assessment.get("risk_decision", "unknown")
            quality_info += f"""
- **Risk Score:** {risk_score:.2f} (0=good, 1=poor)
- **Rule Decision:** {risk_decision.title()}"""
        
        # ML assessment details
        ml_assessment = quality_assessment.get("ml_assessment", {})
        if ml_assessment and ml_assessment.get("ml_available"):
            ml_prob = ml_assessment.get("ml_rescan_probability", 0)
            ml_pred = ml_assessment.get("ml_prediction", 0)
            agreement = ml_assessment.get("rule_vs_ml_agreement", None)
            
            ml_decision = "Rescan" if ml_pred == 1 else "Accept"
            agreement_icon = "✅" if agreement else "❌"
            
            quality_info += f"""
- **ML Decision:** {ml_decision} (prob: {ml_prob:.2f})
- **Agreement:** {agreement_icon} Rule-ML"""
        elif ml_assessment:
            ml_status = ml_assessment.get('ml_status', 'unknown')
            if 'model_unavailable' in ml_status:
                quality_info += f"""
- **ML Status:** Model not trained yet (run training script)"""
            else:
                quality_info += f"""
- **ML Status:** Not available ({ml_status})"""
        
        # User guidance
        rescan_decision = quality_assessment.get("rescan_decision", {})
        user_message = rescan_decision.get("user_message", "")
        if user_message and user_message != "Document quality assessment completed.":
            quality_info += f"""
- **📝 Guidance:** {user_message}"""
        
        return quality_info
    
    def _create_enhanced_annotation(self, image, pairs, confidence_threshold):
        """Create annotated image with bounding boxes"""
        if not pairs:
            img_array = np.array(image)
            h, w = img_array.shape[:2]
            
            overlay = img_array.copy()
            cv2.rectangle(overlay, (10, 10), (w-10, 60), (40, 40, 40), -1)
            cv2.addWeighted(img_array, 0.85, overlay, 0.15, 0, img_array)
            
            text = f"No pairs above {confidence_threshold:.1f} confidence threshold"
            cv2.putText(img_array, text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            return Image.fromarray(img_array)
        
        img_array = np.array(image)
        
        def get_confidence_color(confidence):
            if confidence > 0.8:
                return (0, 255, 0)      # Green
            elif confidence > 0.6:
                return (0, 200, 255)    # Orange
            elif confidence > 0.4:
                return (0, 150, 255)    # Light red
            else:
                return (0, 100, 255)    # Red
        
        # Draw bounding boxes
        display_index = 1
        for pair in pairs:
            key_bbox = pair.get("key_bbox")
            value_bbox = pair.get("value_bbox")
            # Skip non-spatial LLM pairs so overlay reflects only items with coordinates
            if not key_bbox and not value_bbox:
                continue

            confidence = pair.get("confidence", 0)
            color = get_confidence_color(confidence)
            thickness = 2 if confidence > 0.7 else 1

            if key_bbox:
                x, y, w, h = key_bbox["x"], key_bbox["y"], key_bbox["width"], key_bbox["height"]
                cv2.rectangle(img_array, (x, y), (x + w, y + h), color, thickness)
                cv2.putText(img_array, f"K{display_index}", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            if value_bbox:
                x, y, w, h = value_bbox["x"], value_bbox["y"], value_bbox["width"], value_bbox["height"]
                cv2.rectangle(img_array, (x, y), (x + w, y + h), color, thickness)
                cv2.putText(img_array, f"V{display_index}", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

                if key_bbox:
                    key_center = (key_bbox["x"] + key_bbox["width"]//2, key_bbox["y"] + key_bbox["height"]//2)
                    value_center = (x + w//2, y + h//2)
                    cv2.line(img_array, key_center, value_center, color, 1)

            display_index += 1
        
        return Image.fromarray(img_array)

    def _create_ocr_overlay(self, image, ocr_blocks, pairs):
        """Draw OCR text blocks and highlight those used by key/value pairs."""
        img = np.array(image).copy()
        # Draw all OCR blocks
        for block in ocr_blocks or []:
            bb = block.get("bbox", {})
            x, y, w, h = int(bb.get("x", 0)), int(bb.get("y", 0)), int(bb.get("width", 0)), int(bb.get("height", 0))
            if w <= 0 or h <= 0:
                continue
            cv2.rectangle(img, (x, y), (x + w, y + h), (160, 160, 160), 1)

        # Highlight extracted pair boxes
        for idx, pair in enumerate(pairs, start=1):
            for role, b in (("K", pair.get("key_bbox")), ("V", pair.get("value_bbox"))):
                if not b:
                    continue
                x, y, w, h = b["x"], b["y"], b["width"], b["height"]
                color = (0, 255, 0) if role == "K" else (0, 200, 255)
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                cv2.putText(img, f"{role}{idx}", (x, max(10, y-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        return Image.fromarray(img)
    
    def get_system_status(self):
        """Get comprehensive system status"""
        if not self.system_ready:
            return "❌ System not ready. Check logs and dependencies."
        
        try:
            # Get ML model status
            ml_status = "❌ Not Available"
            if hasattr(self.processor.quality_checker, 'ml_model_data') and self.processor.quality_checker.ml_model_data:
                model_type = self.processor.quality_checker.ml_model_data.get('model_type', 'unknown')
                performance = self.processor.quality_checker.ml_model_data.get('performance', {})
                roc_auc = performance.get('roc_auc', 0)
                ml_status = f"✅ {model_type.title()} (ROC AUC: {roc_auc:.3f})"
            
            # Get LLM status
            available_providers = list(self.processor.kv_extractor.llm_extractor.available_providers.keys())
            current_provider = self.processor.kv_extractor.llm_extractor.primary_provider.value
            
            status_text = f"""## 🔧 System Status

**Overall Status:** ✅ System Ready

**Components:**
- **Document Processor:** ✅ Operational
- **Quality Checker:** ✅ Rule-based + ML hybrid
- **ML Model:** {ml_status}
- **LLM Providers:** ✅ {len(available_providers)} available
- **Current LLM:** {current_provider}

**Session Statistics:**
- **Documents Processed:** {self.stats['total_processed']}
- **Successful:** {self.stats['successful']}
- **Failed:** {self.stats['failed']}
- **Quality Assessments:** {self.stats['quality_assessments']}
- **Rescans Needed:** {self.stats['rescans_needed']}
- **Total Pairs Extracted:** {self.stats['total_pairs']}
- **Average Processing Time:** {self.stats['avg_processing_time']:.2f}s
"""
            return status_text
            
        except Exception as e:
            return f"⚠️ Status check error: {str(e)[:100]}..."

def create_enhanced_interface():
    """Create enhanced Gradio interface with quality assessment"""
    
    processor = EnhancedDocumentProcessor()
    
    def process_document_ui(image, strategy, confidence, llm_provider,
                            use_roi, roi_x, roi_y, roi_w, roi_h,
                            auto_mode, enable_perspective, enable_line_removal, enable_sharpen,
                            append_mode, prev_pairs):
        """Main processing function for UI"""
        roi = None
        if use_roi:
            try:
                rx, ry, rw, rh = int(roi_x or 0), int(roi_y or 0), int(roi_w or 0), int(roi_h or 0)
                if rw > 0 and rh > 0:
                    roi = {"x": rx, "y": ry, "width": rw, "height": rh}
            except Exception:
                roi = None

        result = processor.process_document_with_quality(
            image, strategy, confidence, llm_provider,
            roi=roi,
            auto_mode=auto_mode,
            enable_perspective=enable_perspective,
            enable_line_removal=enable_line_removal,
            enable_sharpen=enable_sharpen,
            append_mode=append_mode,
            previous_pairs=prev_pairs or []
        )
        
        if result["success"]:
            # Format detailed quality assessment
            quality_display = format_detailed_quality_assessment(result["quality_assessment"])
            
            return (
                result["image"],
                result.get("quality_overlay", result["image"]),  # Quality overlay image
                result.get("preprocessed", result["image"]),     # Preprocessed image for OCR
                result["table"], 
                result["summary"],
                quality_display,
                result["session_id"]
            )
        else:
            return (
                None,
                None,  # No quality overlay for errors
                None,  # No preprocessed image for errors
                [],
                result["summary"],
                "Quality assessment not available due to processing error.",
                ""
            )
    
    def format_detailed_quality_assessment(quality_assessment):
        """Format detailed quality assessment for display"""
        if not quality_assessment:
            return "No quality assessment data available."
        
        display = f"""## 📊 Quality Assessment Report

### 🎯 Overall Assessment
- **Quality Confidence:** {quality_assessment.get('confidence', 0):.2f}
- **Needs Rescan:** {'Yes' if quality_assessment.get('needs_rescan', False) else 'No'}
- **Issues Detected:** {len(quality_assessment.get('issues', []))}
- **Blur Score:** {quality_assessment.get('blur_score', 0):.1f}
- **Brightness:** {quality_assessment.get('brightness', 128):.1f}
"""
        
        # Rule-based risk assessment
        risk_assessment = quality_assessment.get("quality_risk_assessment", {})
        if risk_assessment:
            risk_score = risk_assessment.get("quality_risk_score", 0)
            risk_decision = risk_assessment.get("risk_decision", "unknown")
            risk_reasons = risk_assessment.get("risk_reasons", [])
            
            display += f"""
### 📋 Rule-Based Risk Analysis
- **Risk Score:** {risk_score:.3f} (0=excellent, 1=poor)
- **Decision:** {risk_decision.title()}
- **Risk Factors:** {', '.join(risk_reasons) if risk_reasons else 'None'}
"""
        
        # ML assessment
        ml_assessment = quality_assessment.get("ml_assessment", {})
        if ml_assessment:
            if ml_assessment.get("ml_available", False):
                ml_prob = ml_assessment.get("ml_rescan_probability", 0)
                ml_pred = ml_assessment.get("ml_prediction", 0)
                agreement = ml_assessment.get("rule_vs_ml_agreement", None)
                
                ml_decision = "Needs Rescan" if ml_pred == 1 else "Acceptable"
                agreement_text = "✅ Agree" if agreement else "❌ Disagree"
                
                display += f"""
### 🤖 ML Classification
- **ML Available:** Yes
- **ML Decision:** {ml_decision}
- **Rescan Probability:** {ml_prob:.3f}
- **Rule-ML Agreement:** {agreement_text}
"""
            else:
                display += f"""
### 🤖 ML Classification
- **ML Available:** No
- **Status:** {ml_assessment.get('ml_status', 'Model not trained/loaded')}
- **Note:** Train ML model using train_quality_classifier.py
"""
        
        # Cut-off analysis
        cutoff_analysis = quality_assessment.get("cut_off_analysis", {})
        if cutoff_analysis:
            edge_issues = cutoff_analysis.get("edge_cut_issues", [])
            text_issues = cutoff_analysis.get("text_edge_issues", [])
            density_violations = cutoff_analysis.get("density_violations", [])
            
            display += f"""
### ✂️ Cut-off Detection Analysis
- **Edge Cut Issues:** {', '.join(edge_issues) if edge_issues else 'None'}
- **Text Edge Issues:** {', '.join(text_issues) if text_issues else 'None'}
- **Density Violations:** {', '.join(density_violations) if density_violations else 'None'}
"""
        
        # Advanced quality assessment
        advanced_assessment = quality_assessment.get("advanced_quality_assessment", {})
        if advanced_assessment and advanced_assessment.get("advanced_features_available", False):
            overall_score = advanced_assessment.get("overall_quality_score", 0) or 0
            quality_class = advanced_assessment.get("quality_class", "unknown") or "unknown"
            
            # Ensure numeric values
            try:
                overall_score = float(overall_score)
                quality_class = str(quality_class)
            except (ValueError, TypeError):
                overall_score = 0.0
                quality_class = "unknown"
            
            # Quality class icon
            class_icons = {"excellent": "🌟", "good": "✅", "fair": "⚠️", "poor": "❌", "unknown": "❓"}
            class_icon = class_icons.get(quality_class, "❓")
            
            display += f"""
### 🔬 Advanced Quality Analysis
- **Overall Quality Score:** {overall_score:.3f}
- **Quality Class:** {class_icon} {quality_class.title()}
"""
            
            # Enhanced blur analysis
            blur_analysis = advanced_assessment.get("enhanced_blur_analysis", {})
            if blur_analysis:
                tenengrad = blur_analysis.get("tenengrad_energy", 0) or 0
                composite_blur = blur_analysis.get("composite_blur_confidence", 0) or 0
                brenner = blur_analysis.get("brenner_focus", 0) or 0
                
                # Ensure numeric values
                try:
                    tenengrad = float(tenengrad)
                    composite_blur = float(composite_blur)
                    brenner = float(brenner)
                except (ValueError, TypeError):
                    tenengrad = 0.0
                    composite_blur = 0.0
                    brenner = 0.0
                
                display += f"""
#### 🌫️ Enhanced Blur Detection
- **Tenengrad Energy:** {tenengrad:.1f} (higher = sharper)
- **Composite Blur Confidence:** {composite_blur:.3f}
- **Brenner Focus:** {brenner:.1f}
"""
            
            # Brightness analysis
            brightness_analysis = advanced_assessment.get("brightness_analysis", {})
            if brightness_analysis:
                skewness = brightness_analysis.get("brightness_skewness", 0) or 0
                entropy = brightness_analysis.get("brightness_entropy", 0) or 0
                overexposure = brightness_analysis.get("overexposure_ratio", 0) or 0
                underexposure = brightness_analysis.get("underexposure_ratio", 0) or 0
                contrast = brightness_analysis.get("rms_contrast", 0) or 0
                
                # Ensure numeric values
                try:
                    skewness = float(skewness)
                    entropy = float(entropy)
                    overexposure = float(overexposure)
                    underexposure = float(underexposure)
                    contrast = float(contrast)
                except (ValueError, TypeError):
                    skewness = 0.0
                    entropy = 0.0
                    overexposure = 0.0
                    underexposure = 0.0
                    contrast = 0.0
                
                display += f"""
#### 💡 Statistical Brightness Analysis
- **Brightness Skewness:** {skewness:.3f} (0=symmetric, ±1=skewed)
- **Brightness Entropy:** {entropy:.3f} (higher = more varied)
- **Overexposure Ratio:** {overexposure:.3f} ({overexposure*100:.1f}%)
- **Underexposure Ratio:** {underexposure:.3f} ({underexposure*100:.1f}%)
- **RMS Contrast:** {contrast:.1f}
"""
            
            # Text layout analysis
            layout_analysis = advanced_assessment.get("text_layout_analysis", {})
            if layout_analysis:
                # Safe value extraction with None handling
                spacing_var = layout_analysis.get("text_spacing_variance", 0) or 0
                alignment_score = layout_analysis.get("text_alignment_score", 0) or 0
                density_uniformity = layout_analysis.get("text_density_uniformity", 0) or 0
                
                # Ensure values are numeric
                try:
                    spacing_var = float(spacing_var)
                    alignment_score = float(alignment_score)
                    density_uniformity = float(density_uniformity)
                except (ValueError, TypeError):
                    spacing_var = 0.0
                    alignment_score = 0.0
                    density_uniformity = 0.0
                
                display += f"""
#### 📝 Text Layout Analysis
- **Text Spacing Variance:** {spacing_var:.2f} (lower = more uniform)
- **Text Alignment Score:** {alignment_score:.3f} (0=perfect, 1=scattered)
- **Text Density Uniformity:** {density_uniformity:.3f} (higher = more even)
"""
            
            # Morphological analysis
            morph_analysis = advanced_assessment.get("morphological_analysis", {})
            if morph_analysis:
                edge_density = morph_analysis.get("edge_density", 0) or 0
                components = morph_analysis.get("connected_components", 0) or 0
                border_density = morph_analysis.get("border_edge_density", 0) or 0
                
                # Ensure numeric values
                try:
                    edge_density = float(edge_density)
                    components = int(components)
                    border_density = float(border_density)
                except (ValueError, TypeError):
                    edge_density = 0.0
                    components = 0
                    border_density = 0.0
                
                display += f"""
#### 🔍 Morphological Edge Analysis
- **Edge Density:** {edge_density:.4f} ({edge_density*100:.2f}%)
- **Connected Components:** {components}
- **Border Edge Density:** {border_density:.4f} (higher = possible cut-off)
"""
            
            # Statistical modeling
            stats_modeling = advanced_assessment.get("statistical_modeling", {})
            if stats_modeling:
                risk_prob = stats_modeling.get("statistical_risk_probability", 0) or 0
                confidence = stats_modeling.get("statistical_confidence", 0) or 0
                dominant_factor = stats_modeling.get("dominant_risk_factor", "unknown") or "unknown"
                
                # Ensure numeric values
                try:
                    risk_prob = float(risk_prob)
                    confidence = float(confidence)
                    dominant_factor = str(dominant_factor)
                except (ValueError, TypeError):
                    risk_prob = 0.0
                    confidence = 0.0
                    dominant_factor = "unknown"
                
                display += f"""
#### 📊 Statistical Risk Modeling
- **Statistical Risk Probability:** {risk_prob:.3f}
- **Statistical Confidence:** {confidence:.3f}
- **Dominant Risk Factor:** {dominant_factor.replace('_', ' ').title()}
"""
        else:
            # Check what advanced packages are actually available
            try:
                import scipy
                scipy_status = "✅ Available"
            except ImportError:
                scipy_status = "❌ Missing"
            
            try:
                import sklearn
                sklearn_status = "✅ Available" 
            except ImportError:
                sklearn_status = "❌ Missing"
                
            try:
                import statsmodels
                statsmodels_status = "✅ Available"
            except ImportError:
                statsmodels_status = "❌ Missing"
            
            display += f"""
### 🔬 Advanced Quality Analysis
- **Status:** Not Available
- **SciPy:** {scipy_status}
- **Scikit-learn:** {sklearn_status}
- **Statsmodels:** {statsmodels_status}
- **Note:** If packages show as available but features don't work, restart the application
"""
        
        # Rescan recommendations
        rescan_decision = quality_assessment.get("rescan_decision", {})
        if rescan_decision:
            urgency = rescan_decision.get("rescan_urgency", "low")
            reasons = rescan_decision.get("rescan_reasons", [])
            user_message = rescan_decision.get("user_message", "")
            
            urgency_icon = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(urgency, "⚪")
            
            display += f"""
### 🚦 Rescan Recommendations
- **Urgency:** {urgency_icon} {urgency.title()}
- **Reasons:** {', '.join(reasons) if reasons else 'None'}
- **Guidance:** {user_message}
"""
        
        return display
    
    def provide_quality_feedback(session_id, feedback_text, was_quality_good):
        """Handle quality feedback from user"""
        if not session_id:
            return "❌ No session ID available. Process a document first."
        
        try:
            actual_needs_rescan = not was_quality_good
            quality_data_collector.update_ground_truth(
                session_id=session_id,
                actual_needs_rescan=actual_needs_rescan,
                user_feedback=feedback_text
            )
            
            return f"✅ Thank you! Feedback recorded for session {session_id}. This helps improve our ML quality assessment."
        except Exception as e:
            return f"❌ Failed to record feedback: {e}"
    
    def get_training_summary():
        """Get ML training data summary"""
        try:
            summary = quality_data_collector.get_training_data_summary()
            
            if "error" in summary:
                return f"❌ Error: {summary['error']}"
            
            total = summary.get("total_samples", 0)
            labeled = summary.get("labeled_samples", 0)
            progress = summary.get("labeling_progress", "0%")
            agreement_rate = summary.get("rule_ml_agreement_rate")
            
            display = f"""## 📊 ML Training Data Summary

**Data Collection Progress:**
- **Total Samples:** {total}
- **Labeled Samples:** {labeled}
- **Labeling Progress:** {progress}
"""
            
            if agreement_rate is not None:
                display += f"\n- **Rule-ML Agreement Rate:** {agreement_rate:.1%}"
            
            doc_types = summary.get("document_types", {})
            if doc_types:
                display += "\n\n**Document Types:**"
                for doc_type, count in doc_types.items():
                    display += f"\n- {doc_type}: {count} samples"
            
            if labeled >= 100:
                display += f"\n\n✅ **Ready for training!** You have {labeled} labeled samples."
                display += f"\nRun: `python train_quality_classifier.py` to train the ML model."
            else:
                needed = 100 - labeled
                display += f"\n\n⏳ **Need {needed} more labeled samples** for ML training."
                display += f"\nProvide feedback on quality assessments to help reach this goal."
            
            return display
            
        except Exception as e:
            return f"❌ Failed to get training summary: {e}"
    
    # Create the interface
    with gr.Blocks(
        title="🧠 Enhanced Document Scanner",
        theme=gr.themes.Soft(),
        css="""
        .quality-excellent { background-color: #d4edda; border-color: #c3e6cb; }
        .quality-good { background-color: #fff3cd; border-color: #ffeaa7; }
        .quality-poor { background-color: #f8d7da; border-color: #f5c6cb; }
        .main-container { max-width: 1400px; margin: 0 auto; }
        """
    ) as demo:
        
        gr.Markdown("""
        # 🧠 Enhanced Document Scanner
        
        **Advanced document processing with AI-powered quality assessment**
        
        Features hybrid quality analysis combining rule-based computer vision with machine learning.
        """)
        
        with gr.Row():
            # Left column - Input
            with gr.Column(scale=1):
                gr.Markdown("### 📄 Document Input")
                
                image_input = gr.Image(
                    type="pil",
                    label="Upload Document",
                    sources=["upload", "clipboard"],
                    interactive=True
                )
                
                gr.Markdown("### ⚙️ Processing Options")
                
                strategy_input = gr.Radio(
                    choices=[
                        ("🧠 Adaptive First", "adaptive_first"),
                        ("🤖 LLM First", "llm_first"),
                        ("⚡ Parallel", "parallel"),
                        ("🔧 Adaptive Only", "adaptive_only"),
                    ],
                    value="adaptive_first",
                    label="Extraction Strategy"
                )
                
                confidence_input = gr.Slider(
                    minimum=0.1,
                    maximum=1.0,
                    value=0.5,
                    step=0.1,
                    label="Confidence Threshold"
                )
                
                llm_provider_input = gr.Radio(
                    choices=[
                        ("🦙 Ollama (Local)", "ollama"),
                        ("🤖 OpenAI GPT", "openai"),
                        ("🧠 Anthropic Claude", "anthropic"),
                    ],
                    value="ollama",
                    label="LLM Provider"
                )

                gr.Markdown("### 🎛️ Preprocessing Controls")
                auto_mode_toggle = gr.Checkbox(value=True, label="Auto mode (pick best binarization)")
                enable_perspective_toggle = gr.Checkbox(value=True, label="Enable perspective correction")
                enable_line_removal_toggle = gr.Checkbox(value=True, label="Enable line removal")
                enable_sharpen_toggle = gr.Checkbox(value=True, label="Enable sharpening")

                gr.Markdown("### 🎯 Focus Region (optional)")
                use_roi_toggle = gr.Checkbox(value=False, label="Enable ROI")
                with gr.Row():
                    roi_x = gr.Number(value=0, label="x")
                    roi_y = gr.Number(value=0, label="y")
                    roi_w = gr.Number(value=0, label="width")
                    roi_h = gr.Number(value=0, label="height")
                append_mode_toggle = gr.Checkbox(value=False, label="Append to previous results")
                prev_pairs_state = gr.State([])
                
                process_btn = gr.Button(
                    "🚀 Process Document",
                    variant="primary",
                    size="lg"
                )
                
                # System controls
                gr.Markdown("### 🔧 System")
                
                with gr.Row():
                    status_btn = gr.Button("📊 Check Status", size="sm")
                    training_btn = gr.Button("🎓 Training Data", size="sm")
            
            # Right column - Results
            with gr.Column(scale=2):
                gr.Markdown("### 📊 Results")
                
                with gr.Tabs():
                    with gr.TabItem("🖼️ Annotated Image"):
                        result_image = gr.Image(
                            label="Processed Document",
                            interactive=False
                        )
                        
                        gr.Markdown("""
                        **Legend:** 🟢 High Confidence | 🟡 Medium | 🔴 Low  
                        **K#** = Key, **V#** = Value, Lines show connections
                        """)
                    
                    with gr.TabItem("🔍 Quality Overlay"):
                        quality_overlay_image = gr.Image(
                            label="Quality Assessment Overlay",
                            interactive=False
                        )
                    with gr.TabItem("🧪 Preprocessed (OCR Input)"):
                        preprocessed_image = gr.Image(
                            label="Preprocessed Image for OCR",
                            interactive=False
                        )
                        
                        gr.Markdown("""
                        **Quality Overlay Legend:**  
                        🔴 **Red Boxes** = Edge cuts detected  
                        🟠 **Orange Boxes** = Text near edges  
                        🟡 **Yellow Indicators** = Density violations  
                        **Top-right Badge** = Overall quality score  
                        **Gray Dashed Lines** = Safe margin guidelines
                        """)
                    
                    # (Removed OCR Blocks tab per request)

                    with gr.TabItem("📋 Extracted Data"):
                        result_table = gr.Dataframe(
                            headers=["#", "Key", "Value", "Confidence", "Source"],
                            label="Key-Value Pairs",
                            interactive=False
                        )
                    
                    with gr.TabItem("📈 Summary"):
                        result_summary = gr.Markdown()
                    
                    with gr.TabItem("📊 Quality Assessment"):
                        quality_display = gr.Markdown()
                        
                        # Quality feedback section
                        gr.Markdown("---")
                        gr.Markdown("### 💬 Quality Feedback")
                        gr.Markdown("Help improve our ML model by providing feedback:")
                        
                        session_id_display = gr.Textbox(
                            label="Session ID",
                            interactive=False,
                            visible=False
                        )
                        
                        quality_good = gr.Radio(
                            choices=[("✅ Quality was good", True), ("❌ Quality was poor", False)],
                            label="Was the document quality actually acceptable?",
                            info="This helps train our ML model"
                        )
                        
                        feedback_text = gr.Textbox(
                            label="Additional Feedback (Optional)",
                            placeholder="Any specific issues or comments...",
                            lines=2
                        )
                        
                        feedback_btn = gr.Button("📝 Submit Feedback", size="sm")
                        feedback_result = gr.Markdown()
                    
                    # Removed System Status and ML Training tabs from the main flow to reduce clutter
        
        # Tips section
        gr.Markdown("""
        ### 💡 Enhanced Features
        
        - **🔍 Quality Assessment:** Hybrid rule-based + ML analysis
        - **📊 Risk Scoring:** Computer vision metrics for document quality
        - **🤖 ML Classification:** Learns from user feedback over time
        - **📈 Continuous Learning:** Your feedback improves the system
        - **🎯 Smart Recommendations:** Actionable guidance for better scans
        
        **🚀 Quick Start:**
        1. Upload a document image
        2. Choose processing strategy
        3. Review quality assessment
        4. Provide feedback to improve ML model
        5. Check training progress in ML Training tab
        """)
        
        # Event handlers
        process_btn.click(
            fn=process_document_ui,
            inputs=[image_input, strategy_input, confidence_input, llm_provider_input,
                    use_roi_toggle, roi_x, roi_y, roi_w, roi_h,
                    auto_mode_toggle, enable_perspective_toggle, enable_line_removal_toggle, enable_sharpen_toggle,
                    append_mode_toggle, prev_pairs_state],
            outputs=[result_image, quality_overlay_image, preprocessed_image, result_table, result_summary, quality_display, session_id_display]
        )

        # (Removed redundant click handler that returned gr.update())
        
        feedback_btn.click(
            fn=provide_quality_feedback,
            inputs=[session_id_display, feedback_text, quality_good],
            outputs=[feedback_result]
        )
        
        # Remove handlers tied to removed tabs
    
    return demo

if __name__ == "__main__":
    print("🚀 Starting Enhanced Document Scanner Interface...")
    
    try:
        demo = create_enhanced_interface()
        
        print("✅ Enhanced interface ready!")
        print("🌐 Starting server on http://localhost:7861")
        print("📊 Features: Rule-based + ML quality assessment")
        print("🎓 ML training data collection enabled")
        
        demo.launch(
            server_name="0.0.0.0",
            server_port=7861,
            share=False,
            show_error=True,
            show_api=False
        )
        
    except Exception as e:
        print(f"❌ Failed to start enhanced interface: {e}")
        traceback.print_exc()