#!/usr/bin/env python3
"""
Modern Gradio UI for Hybrid Document Scanner
Clean, intuitive interface with robust error handling and user experience
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

from hybrid_document_processor import HybridDocumentProcessor
from hybrid_kv_extractor import ExtractionStrategy
from llm_kv_extractor import LLMProvider

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentProcessor:
    """Simplified and robust document processor for Gradio UI"""
    
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
            "last_processed": None
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
    
    def get_system_status(self):
        """Get comprehensive system status"""
        if not self.system_ready:
            return {
                "status": "❌ System Not Ready",
                "details": "Document processor failed to initialize",
                "recommendations": "Check logs and dependencies"
            }
        
        try:
            # Check LLM providers
            llm_info = self.processor.kv_extractor.llm_extractor.get_provider_info()
            llm_available = self.processor.kv_extractor.llm_extractor.is_available()
            current_provider = self.processor.kv_extractor.llm_extractor.primary_provider.value
            
            # Get available providers (just the names from the actual available_providers dict)
            available_providers = list(self.processor.kv_extractor.llm_extractor.available_providers.keys())
            available_provider_names = [p.value for p in available_providers]
            
            return {
                "status": "✅ System Ready",
                "processor": "✅ Operational",
                "llm_providers": f"✅ {len(available_providers)} available" if available_providers else "⚠️ None available",
                "available_providers": available_provider_names,
                "current_provider": current_provider,
                "provider_details": llm_info,
                "stats": self.stats
            }
            
        except Exception as e:
            return {
                "status": "⚠️ Partial System Ready",
                "details": f"Status check error: {str(e)[:50]}...",
                "recommendations": "Some features may not work properly"
            }
    
    def process_document(self, image, strategy="adaptive_first", confidence=0.5, llm_provider="ollama"):
        """Process document with comprehensive error handling"""
        start_time = time.time()
        
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
                        logger.info(f"Set LLM provider to: {llm_provider}")
                    except (ValueError, AttributeError) as llm_error:
                        logger.warning(f"Could not set LLM provider {llm_provider}: {llm_error}")
                        # Continue with default provider
                        
            except ValueError as e:
                self.stats["failed"] += 1
                return self._create_error_response(f"Invalid strategy: {strategy}")
            
            # Process document
            logger.info(f"🔄 Processing document with strategy: {strategy}, provider: {llm_provider}")
            result = self.processor.process_image_bytes(image_bytes, "document")
            processing_time = time.time() - start_time
            
            logger.info(f"✅ Document processing completed in {processing_time:.2f}s")
            
            # Update processing time average
            total_time = self.stats["avg_processing_time"] * (self.stats["total_processed"] - 1) + processing_time
            self.stats["avg_processing_time"] = total_time / self.stats["total_processed"]
            
            if result["status"] == "success":
                self.stats["successful"] += 1
                pairs_count = len(result.get("key_value_pairs", []))
                self.stats["total_pairs"] += pairs_count
                
                return self._create_success_response(result, processing_time, image)
            else:
                self.stats["failed"] += 1
                error_msg = result.get("error", "Unknown processing error")
                return self._create_error_response(f"Processing failed: {error_msg}")
                
        except Exception as e:
            self.stats["failed"] += 1
            logger.error(f"Document processing error: {e}")
            return self._create_error_response(f"Unexpected error: {str(e)[:100]}...")
    
    def _create_success_response(self, result, processing_time, original_image):
        """Create a success response with annotated image and data"""
        pairs = result.get("key_value_pairs", [])
        
        # Create annotated image
        annotated_image = self._create_simple_annotation(original_image, pairs)
        
        # Create table data
        table_data = []
        for i, pair in enumerate(pairs, 1):
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
        
        # Get LLM provider info if available
        llm_used = "Not used"
        try:
            if self.processor and hasattr(self.processor, 'kv_extractor'):
                llm_used = self.processor.kv_extractor.llm_extractor.primary_provider.value
        except:
            pass
        
        # Create summary
        summary = f"""### ✅ Processing Successful
        
**⏱️ Processing Time:** {processing_time:.2f} seconds  
**📊 Extracted Pairs:** {len(pairs)}  
**🎯 Strategy Used:** {result.get('extraction_metadata', {}).get('strategy_used', 'unknown')}  
**🔧 Primary Method:** {result.get('extraction_metadata', {}).get('primary_method', 'unknown')}  
**🤖 LLM Provider:** {llm_used}

**📈 Confidence Distribution:**
- High (>0.8): {sum(1 for p in pairs if p.get('confidence', 0) > 0.8)} pairs  
- Medium (0.5-0.8): {sum(1 for p in pairs if 0.5 < p.get('confidence', 0) <= 0.8)} pairs  
- Low (<0.5): {sum(1 for p in pairs if p.get('confidence', 0) <= 0.5)} pairs

**📋 Method Breakdown:**
{self._get_method_breakdown(pairs)}
"""
        
        return {
            "success": True,
            "image": annotated_image,
            "table": table_data,
            "summary": summary,
            "status": f"✅ Successfully extracted {len(pairs)} key-value pairs"
        }
    
    def _create_error_response(self, message):
        """Create an error response"""
        return {
            "success": False,
            "image": None,
            "table": [],
            "summary": f"### ❌ Processing Failed\n\n**Error:** {message}",
            "status": f"❌ {message}"
        }
    
    def _create_simple_annotation(self, image, pairs):
        """Create a clean, simple annotation of the image"""
        if not pairs:
            return image
        
        # Convert to OpenCV format
        img_array = np.array(image)
        
        # Colors for annotation
        colors = {
            "key": (0, 255, 255),     # Cyan for keys
            "value": (255, 255, 0),   # Yellow for values
            "connection": (255, 255, 255)  # White for connections
        }
        
        # Draw bounding boxes and connections
        for i, pair in enumerate(pairs):
            confidence = pair.get("confidence", 0)
            thickness = max(1, int(confidence * 3))
            
            key_bbox = pair.get("key_bbox")
            value_bbox = pair.get("value_bbox")
            
            # Draw key box
            if key_bbox:
                x, y, w, h = key_bbox["x"], key_bbox["y"], key_bbox["width"], key_bbox["height"]
                cv2.rectangle(img_array, (x, y), (x + w, y + h), colors["key"], thickness)
                cv2.putText(img_array, f"K{i+1}", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors["key"], 1)
            
            # Draw value box
            if value_bbox:
                x, y, w, h = value_bbox["x"], value_bbox["y"], value_bbox["width"], value_bbox["height"]
                cv2.rectangle(img_array, (x, y), (x + w, y + h), colors["value"], thickness)
                cv2.putText(img_array, f"V{i+1}", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors["value"], 1)
                
                # Draw connection line
                if key_bbox:
                    key_center = (key_bbox["x"] + key_bbox["width"]//2, key_bbox["y"] + key_bbox["height"]//2)
                    value_center = (x + w//2, y + h//2)
                    cv2.line(img_array, key_center, value_center, colors["connection"], 1)
        
        return Image.fromarray(img_array)
    
    def _get_method_breakdown(self, pairs):
        """Get breakdown of extraction methods used"""
        methods = {}
        for pair in pairs:
            method = pair.get("source", "unknown")
            methods[method] = methods.get(method, 0) + 1
        
        breakdown = []
        for method, count in methods.items():
            percentage = (count / len(pairs)) * 100 if pairs else 0
            breakdown.append(f"- {method.title()}: {count} pairs ({percentage:.1f}%)")
        
        return "\n".join(breakdown) if breakdown else "- No data available"
    
    def get_available_providers(self):
        """Get available LLM providers for UI"""
        if not self.system_ready or not self.processor:
            return ["ollama"], "ollama"  # fallback
        
        try:
            # Get available providers from the actual system
            available_providers = list(self.processor.kv_extractor.llm_extractor.available_providers.keys())
            provider_names = [p.value for p in available_providers]
            current_provider = self.processor.kv_extractor.llm_extractor.primary_provider.value
            
            # Create choices with nice labels
            provider_choices = []
            for provider_name in provider_names:
                if provider_name == "ollama":
                    provider_choices.append(("🦙 Ollama (Local)", "ollama"))
                elif provider_name == "openai":
                    provider_choices.append(("🤖 OpenAI GPT", "openai"))
                elif provider_name == "anthropic":
                    provider_choices.append(("🧠 Anthropic Claude", "anthropic"))
                elif provider_name == "azure_openai":
                    provider_choices.append(("☁️ Azure OpenAI", "azure_openai"))
                else:
                    provider_choices.append((f"🔧 {provider_name.title()}", provider_name))
            
            # If no providers available, fallback to all options (will show errors when used)
            if not provider_choices:
                provider_choices = [
                    ("🦙 Ollama (Local) - Not Available", "ollama"),
                    ("🤖 OpenAI GPT - Not Available", "openai"),
                    ("🧠 Anthropic Claude - Not Available", "anthropic"),
                    ("☁️ Azure OpenAI - Not Available", "azure_openai"),
                ]
                current_provider = "ollama"  # fallback
            
            return provider_choices, current_provider
            
        except Exception as e:
            logger.warning(f"Could not get provider info: {e}")
            # Fallback to default options
            return [
                ("🦙 Ollama (Local)", "ollama"),
                ("🤖 OpenAI GPT", "openai"),
                ("🧠 Anthropic Claude", "anthropic"),
                ("☁️ Azure OpenAI", "azure_openai"),
            ], "ollama"

def create_modern_interface():
    """Create a modern, clean Gradio interface"""
    
    processor = DocumentProcessor()
    
    def process_document_ui(image, strategy, confidence, llm_provider):
        """Main processing function for UI"""
        result = processor.process_document(image, strategy, confidence, llm_provider)
        
        if result["success"]:
            return (
                result["image"],
                result["table"], 
                result["summary"]
            )
        else:
            return (
                None,
                [],
                result["summary"]
            )
    
    def get_status():
        """Get system status for UI"""
        status = processor.get_system_status()
        
        # Get current LLM provider info
        current_llm = "Unknown"
        llm_details = ""
        
        if processor.system_ready:
            try:
                current_llm = processor.processor.kv_extractor.llm_extractor.primary_provider.value
                llm_info = processor.processor.kv_extractor.llm_extractor.get_provider_info()
                available_providers = list(processor.processor.kv_extractor.llm_extractor.available_providers.keys())
                
                llm_details = f"\n**🤖 LLM Provider Status:**\n"
                llm_details += f"- **Active Provider**: 🎯 **{current_llm.upper()}**\n"
                llm_details += f"- **Available Providers**: {len(available_providers)}\n\n"
                
                for provider, info in llm_info.items():
                    status_icon = "✅" if provider in [p.value for p in available_providers] else "❌"
                    model = info.get('model', 'unknown')
                    llm_details += f"- **{provider.title()}**: {status_icon} Model: {model}\n"
                    
            except Exception as e:
                llm_details = f"\n**LLM Error:** {str(e)[:50]}...\n"
        
        status_text = f"""## 🔧 System Status

**Overall Status:** {status['status']}

**Components:**
- **Document Processor:** {status.get('processor', 'Unknown')}
- **LLM Providers:** {status.get('llm_providers', 'Unknown')}
- **Current LLM:** {current_llm}

**Available Providers:** {', '.join(status.get('available_providers', ['None']))}

{llm_details}

**Session Statistics:**
- **Documents Processed:** {processor.stats['total_processed']}
- **Successful:** {processor.stats['successful']}
- **Failed:** {processor.stats['failed']}
- **Total Pairs Extracted:** {processor.stats['total_pairs']}
- **Average Processing Time:** {processor.stats['avg_processing_time']:.2f}s
- **Last Processed:** {processor.stats['last_processed'] or 'Never'}
"""
        return status_text
    
    def reset_stats():
        """Reset session statistics"""
        processor.stats = {
            "total_processed": 0,
            "successful": 0,
            "failed": 0,
            "total_pairs": 0,
            "avg_processing_time": 0.0,
            "last_processed": None
        }
        return "✅ Statistics reset successfully"
    

    
    # Create the interface
    with gr.Blocks(
        title="🧠 Document Scanner",
        theme=gr.themes.Soft(),
        css="""
        .main-container { max-width: 1200px; margin: 0 auto; }
        .status-bar { background: #f0f2f6; padding: 10px; border-radius: 8px; margin: 10px 0; }
        .error-box { background: #fee; border: 1px solid #fcc; padding: 10px; border-radius: 5px; }
        .success-box { background: #efe; border: 1px solid #cfc; padding: 10px; border-radius: 5px; }
        """
    ) as demo:
        
        gr.Markdown("""
        # 🧠 Hybrid Document Scanner
        
        **Professional document processing with intelligent key-value extraction**
        
        Upload a document image and extract structured information automatically.
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
                    label="Extraction Strategy",
                    info="Choose processing method"
                )
                
                confidence_input = gr.Slider(
                    minimum=0.1,
                    maximum=1.0,
                    value=0.5,
                    step=0.1,
                    label="Confidence Threshold",
                    info="Higher = fewer but more confident results"
                )
                
                # Get available providers dynamically
                provider_choices, default_provider = processor.get_available_providers()
                
                llm_provider_input = gr.Radio(
                    choices=provider_choices,
                    value=default_provider,
                    label="LLM Provider",
                    info="Choose AI provider for LLM strategies (only available providers shown)"
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
                    reset_btn = gr.Button("🔄 Reset Stats", size="sm")
            
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
                        **Legend:** 🟦 Keys | 🟨 Values | ⚪ Connections
                        """)
                    
                    with gr.TabItem("📋 Extracted Data"):
                        result_table = gr.Dataframe(
                            headers=["#", "Key", "Value", "Confidence", "Source"],
                            label="Key-Value Pairs",
                            interactive=False,
                            wrap=True
                        )
                    
                    with gr.TabItem("📈 Statistics"):
                        result_summary = gr.Markdown(
                            "Process a document to see detailed statistics"
                        )
                        
                    with gr.TabItem("🔧 System Status"):
                        system_status = gr.Markdown(
                            "Click 'Check Status' to view system information"
                        )
        
        # Examples
        gr.Markdown("### 📁 Quick Start")
        
        example_files = []
        import os
        if os.path.exists("test_images"):
            for file in sorted(os.listdir("test_images")):
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    example_files.append(os.path.join("test_images", file))
        
        if example_files:
            gr.Examples(
                examples=[[f] for f in example_files[:3]],
                inputs=[image_input],
                label="Try these sample documents"
            )
        
        # Tips
        gr.Markdown("""
        ### 💡 Tips for Best Results
        
        - **📸 Image Quality:** Use clear, well-lit images
        - **🔄 Strategy:** Try 'Adaptive First' for forms, 'LLM First' for complex documents  
        - **⚙️ Confidence:** Lower threshold (0.3) for more results, higher (0.7) for quality
        - **🤖 LLM Provider:** Interface shows only available providers
        - **🔧 Troubleshooting:** Check system status if processing fails
        
        **📋 Strategy Guide:**
        - **Adaptive First**: Fast, good for structured documents
        - **LLM First**: Slower but better for complex layouts  
        - **Parallel**: Best accuracy, uses both methods
        - **Adaptive Only**: Fastest, no LLM required
        
        **🔌 Provider Setup:**
        - **Ollama**: Run `ollama serve` and install models locally
        - **OpenAI**: Set `OPENAI_API_KEY` environment variable
        - **Anthropic**: Set `ANTHROPIC_API_KEY` environment variable  
        - **Azure**: Set Azure OpenAI credentials in environment
        """)
        
        # Event handlers
        process_btn.click(
            fn=process_document_ui,
            inputs=[image_input, strategy_input, confidence_input, llm_provider_input],
            outputs=[result_image, result_table, result_summary]
        )
        
        status_btn.click(
            fn=get_status,
            outputs=[system_status]
        )
        
        reset_btn.click(
            fn=reset_stats
        )
        

    
    return demo

if __name__ == "__main__":
    print("🚀 Starting Modern Document Scanner Interface...")
    
    try:
        demo = create_modern_interface()
        
        print("✅ Interface ready!")
        print("🌐 Starting server on http://localhost:7860")
        
        demo.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False,
            show_error=True,
            show_api=False
        )
        
    except Exception as e:
        print(f"❌ Failed to start interface: {e}")
        traceback.print_exc()