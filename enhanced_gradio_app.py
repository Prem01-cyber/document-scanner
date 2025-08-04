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

# Load environment variables from .env file
load_dotenv()

from src.hybrid_document_processor import HybridDocumentProcessor
from src.hybrid_kv_extractor import ExtractionStrategy
from src.llm_kv_extractor import LLMProvider
from quality.quality_data_collector import quality_data_collector

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
    
    def process_document_with_quality(self, image, strategy="adaptive_first", confidence=0.5, llm_provider="ollama"):
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
            result = self.processor.process_image_bytes(image_bytes, "document")
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
        
        # Create annotated image
        annotated_image = self._create_enhanced_annotation(original_image, filtered_pairs, confidence_threshold)
        
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
            quality_info += f"""
- **ML Status:** Not available ({ml_assessment.get('ml_status', 'unknown')})"""
        
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
        for i, pair in enumerate(pairs):
            confidence = pair.get("confidence", 0)
            color = get_confidence_color(confidence)
            thickness = 2 if confidence > 0.7 else 1
            
            key_bbox = pair.get("key_bbox")
            value_bbox = pair.get("value_bbox")
            
            if key_bbox:
                x, y, w, h = key_bbox["x"], key_bbox["y"], key_bbox["width"], key_bbox["height"]
                cv2.rectangle(img_array, (x, y), (x + w, y + h), color, thickness)
                cv2.putText(img_array, f"K{i+1}", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            if value_bbox:
                x, y, w, h = value_bbox["x"], value_bbox["y"], value_bbox["width"], value_bbox["height"]
                cv2.rectangle(img_array, (x, y), (x + w, y + h), color, thickness)
                cv2.putText(img_array, f"V{i+1}", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                
                # Connection line
                if key_bbox:
                    key_center = (key_bbox["x"] + key_bbox["width"]//2, key_bbox["y"] + key_bbox["height"]//2)
                    value_center = (x + w//2, y + h//2)
                    cv2.line(img_array, key_center, value_center, color, 1)
        
        return Image.fromarray(img_array)
    
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
    
    def process_document_ui(image, strategy, confidence, llm_provider):
        """Main processing function for UI"""
        result = processor.process_document_with_quality(image, strategy, confidence, llm_provider)
        
        if result["success"]:
            # Format detailed quality assessment
            quality_display = format_detailed_quality_assessment(result["quality_assessment"])
            
            return (
                result["image"],
                result["table"], 
                result["summary"],
                quality_display,
                result["session_id"]
            )
        else:
            return (
                None,
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
                    
                    with gr.TabItem("🔧 System Status"):
                        system_status = gr.Markdown()
                    
                    with gr.TabItem("🎓 ML Training"):
                        training_summary = gr.Markdown()
        
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
            inputs=[image_input, strategy_input, confidence_input, llm_provider_input],
            outputs=[result_image, result_table, result_summary, quality_display, session_id_display]
        )
        
        feedback_btn.click(
            fn=provide_quality_feedback,
            inputs=[session_id_display, feedback_text, quality_good],
            outputs=[feedback_result]
        )
        
        status_btn.click(
            fn=processor.get_system_status,
            outputs=[system_status]
        )
        
        training_btn.click(
            fn=get_training_summary,
            outputs=[training_summary]
        )
    
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