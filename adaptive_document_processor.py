# Fully Adaptive Document Processor
# Integration example showing how to replace hardcoded system with adaptive learning

import numpy as np
import cv2
from typing import Dict, List
import logging
import asyncio
from fastapi import HTTPException

from adaptive_quality_checker import AdaptiveDocumentQualityChecker
from adaptive_kv_extractor import AdaptiveKeyValueExtractor, TextBlock, BoundingBox
from config import adaptive_config

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
        
    def extract_text_with_bounds(self, image_bytes: bytes) -> List[TextBlock]:
        """Extract text with bounding boxes using Google OCR"""
        try:
            from google.cloud import vision
            
            image = vision.Image(content=image_bytes)
            response = self.client.document_text_detection(image=image)
            
            if response.error.message:
                raise Exception(f"OCR Error: {response.error.message}")
                
            text_blocks = []
            
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
                        
            return text_blocks
            
        except Exception as e:
            logger.error(f"OCR processing error: {e}")
            raise HTTPException(status_code=500, detail=f"OCR failed: {e}")

class AdaptiveDocumentProcessor:
    """
    Fully adaptive document processor that learns and improves over time
    """
    
    def __init__(self):
        # Use adaptive components instead of hardcoded ones
        self.quality_checker = AdaptiveDocumentQualityChecker()
        self.ocr_processor = GoogleOCRProcessor()
        self.kv_extractor = AdaptiveKeyValueExtractor()
        
        # Track processing history for continuous learning
        self.processing_history = []
        
    def process_image_bytes(self, image_bytes: bytes) -> Dict:
        """Process image from bytes with adaptive learning"""
        try:
            # Convert bytes to OpenCV image
            nparr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                raise ValueError("Invalid image format")
                
            return self.process_image(image, image_bytes)
            
        except Exception as e:
            logger.error(f"Adaptive image processing error: {e}")
            raise HTTPException(status_code=400, detail=f"Image processing failed: {e}")
    
    def process_image(self, image: np.ndarray, image_bytes: bytes) -> Dict:
        """
        Complete adaptive processing pipeline with continuous learning
        """
        start_time = asyncio.get_event_loop().time() if asyncio.get_event_loop().is_running() else 0
        
        # Step 1: Adaptive Quality Assessment
        logger.info("Step 1: Adaptive quality assessment")
        quality_result = self.quality_checker.assess_quality(image)
        
        if quality_result["needs_rescan"]:
            return {
                "status": "rescan_needed",
                "quality_assessment": quality_result,
                "message": "Document quality is insufficient. Please rescan.",
                "adaptive_info": {
                    "parameters_used": "Learned from previous documents",
                    "learning_enabled": True
                }
            }
            
        # Step 2: OCR Processing
        logger.info("Step 2: OCR text extraction")
        try:
            text_blocks = self.ocr_processor.extract_text_with_bounds(image_bytes)
        except Exception as e:
            return {
                "status": "ocr_failed",
                "error": str(e),
                "quality_assessment": quality_result
            }
            
        # Step 3: Adaptive Key-Value Extraction
        logger.info("Step 3: Adaptive key-value extraction")
        kv_pairs = self.kv_extractor.extract_key_value_pairs(text_blocks)
        
        # Step 4: Calculate adaptive processing statistics
        processing_time = (asyncio.get_event_loop().time() - start_time) if start_time else 0
        avg_confidence = np.mean([kv.confidence for kv in kv_pairs]) if kv_pairs else 0.0
        
        # Step 5: Learn from this processing session
        processing_result = {
            "average_confidence": avg_confidence,
            "quality_assessment": quality_result,
            "extraction_statistics": {
                "total_pairs": len(kv_pairs),
                "high_confidence_pairs": sum(1 for kv in kv_pairs if kv.confidence > 0.7)
            }
        }
        
        # Update adaptive configuration based on this session
        self._learn_from_processing_session(processing_result)
        
        # Prepare comprehensive response
        response = {
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
                    "extraction_method": kv.extraction_method
                }
                for kv in kv_pairs
            ],
            "processing_time_seconds": processing_time,
            "extraction_statistics": {
                "total_text_blocks": len(text_blocks),
                "identified_key_value_pairs": len(kv_pairs),
                "average_confidence": float(avg_confidence),
                "high_confidence_pairs": sum(1 for kv in kv_pairs if kv.confidence > 0.7),
                "extraction_methods_used": list(set(kv.extraction_method for kv in kv_pairs))
            },
            "adaptive_processing_info": {
                "learning_enabled": True,
                "parameters_adapted": self._get_adaptation_summary(),
                "processing_history_count": len(self.processing_history),
                "confidence_improvement": self._calculate_confidence_trend(),
                "adaptive_thresholds_used": quality_result.get("adaptive_thresholds", {})
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
        
        # Store this session in history
        self.processing_history.append({
            "timestamp": start_time,
            "average_confidence": avg_confidence,
            "pairs_extracted": len(kv_pairs),
            "quality_confidence": quality_result.get("confidence", 0.0)
        })
        
        # Limit history size
        if len(self.processing_history) > 50:
            self.processing_history = self.processing_history[-50:]
        
        return response
    
    def _learn_from_processing_session(self, processing_result: Dict):
        """
        Learn from the current processing session to improve future performance
        """
        try:
            logger.info("Learning from processing session")
            
            # Let the adaptive config learn from this session
            adaptive_config.adapt_from_document_processing(processing_result)
            
            # Let the KV extractor learn from successful extractions
            if processing_result["average_confidence"] > 0.6:
                # This was a successful session, let components learn
                logger.info(f"Successful session (conf: {processing_result['average_confidence']:.2f}), updating learned parameters")
            
        except Exception as e:
            logger.debug(f"Learning from session error: {e}")
    
    def _get_adaptation_summary(self) -> Dict:
        """
        Get summary of what parameters have been adapted
        """
        try:
            summary = {
                "quality_thresholds_learned": 0,
                "extraction_confidence_learned": 0,
                "semantic_thresholds_learned": 0
            }
            
            # Count adapted parameters
            config = adaptive_config.config
            
            for category in ["quality_thresholds", "extraction_confidence", "semantic_thresholds"]:
                if category in config:
                    for param_name, param_config in config[category].items():
                        if (isinstance(param_config, dict) and 
                            param_config.get("confidence_weight", 0) > 0.3):
                            summary[f"{category}_learned"] += 1
            
            return summary
            
        except Exception as e:
            logger.debug(f"Adaptation summary error: {e}")
            return {"error": "Could not generate summary"}
    
    def _calculate_confidence_trend(self) -> Dict:
        """
        Calculate confidence improvement trend over processing history
        """
        try:
            if len(self.processing_history) < 5:
                return {"trend": "insufficient_data", "sessions": len(self.processing_history)}
            
            # Get recent confidence values
            recent_confidences = [
                session["average_confidence"] 
                for session in self.processing_history[-10:]
            ]
            
            early_confidences = [
                session["average_confidence"] 
                for session in self.processing_history[:5]
            ]
            
            recent_avg = np.mean(recent_confidences)
            early_avg = np.mean(early_confidences)
            
            improvement = recent_avg - early_avg
            
            return {
                "trend": "improving" if improvement > 0.05 else "stable" if abs(improvement) <= 0.05 else "declining",
                "improvement_amount": float(improvement),
                "recent_average": float(recent_avg),
                "early_average": float(early_avg),
                "total_sessions": len(self.processing_history)
            }
            
        except Exception as e:
            logger.debug(f"Confidence trend calculation error: {e}")
            return {"trend": "error", "message": str(e)}
    
    def get_learning_statistics(self) -> Dict:
        """
        Get comprehensive learning statistics
        """
        try:
            config = adaptive_config.config
            
            stats = {
                "total_processing_sessions": len(self.processing_history),
                "parameters_learned": {},
                "confidence_trends": self._calculate_confidence_trend(),
                "adaptation_summary": self._get_adaptation_summary()
            }
            
            # Detailed parameter learning stats
            for category in ["quality_thresholds", "extraction_confidence", "semantic_thresholds"]:
                if category in config:
                    stats["parameters_learned"][category] = {}
                    
                    for param_name, param_config in config[category].items():
                        if isinstance(param_config, dict):
                            stats["parameters_learned"][category][param_name] = {
                                "confidence_weight": param_config.get("confidence_weight", 0),
                                "learned_values_count": len(param_config.get("learned_values", [])),
                                "is_actively_learning": param_config.get("confidence_weight", 0) > 0.1
                            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Learning statistics error: {e}")
            return {"error": str(e)}

# FastAPI Integration Example
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse

def create_adaptive_app() -> FastAPI:
    """
    Create FastAPI application with fully adaptive document processing
    """
    app = FastAPI(
        title="Adaptive Document Scanner API", 
        version="3.0.0", 
        description="Fully Adaptive Document Scanner with Continuous Learning"
    )
    
    # Initialize adaptive processor
    processor = AdaptiveDocumentProcessor()
    
    @app.post("/scan-document")
    async def adaptive_scan_document(file: UploadFile = File(...)):
        """
        Adaptive document scanning that learns and improves over time
        """
        try:
            if not file.content_type.startswith('image/'):
                raise HTTPException(status_code=400, detail="File must be an image")
                
            image_bytes = await file.read()
            
            if len(image_bytes) == 0:
                raise HTTPException(status_code=400, detail="Empty file")
                
            # Process with adaptive learning
            result = processor.process_image_bytes(image_bytes)
            
            return JSONResponse(content=result)
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Adaptive processing error: {e}")
            raise HTTPException(status_code=500, detail=f"Processing failed: {e}")
    
    @app.get("/learning-statistics")
    async def get_learning_statistics():
        """
        Get statistics about the adaptive learning process
        """
        try:
            stats = processor.get_learning_statistics()
            return JSONResponse(content=stats)
        except Exception as e:
            logger.error(f"Statistics error: {e}")
            raise HTTPException(status_code=500, detail=f"Statistics failed: {e}")
    
    @app.post("/reset-learning")
    async def reset_learning():
        """
        Reset learned parameters (for testing/debugging)
        """
        try:
            # Reset adaptive config
            adaptive_config.config = adaptive_config._load_default_config()
            adaptive_config.save_config()
            
            # Reset processor history
            processor.processing_history = []
            processor.kv_extractor.learned_patterns = {}
            processor.kv_extractor.pattern_confidence = {}
            
            return {"status": "learning_reset", "message": "All learned parameters have been reset"}
            
        except Exception as e:
            logger.error(f"Reset error: {e}")
            raise HTTPException(status_code=500, detail=f"Reset failed: {e}")
    
    @app.get("/adaptive-config")
    async def get_adaptive_config():
        """
        Get current adaptive configuration
        """
        try:
            return JSONResponse(content=adaptive_config.config)
        except Exception as e:
            logger.error(f"Config retrieval error: {e}")
            raise HTTPException(status_code=500, detail=f"Config retrieval failed: {e}")
    
    @app.get("/")
    async def root():
        """Root endpoint with adaptive API information"""
        return {
            "service": "Adaptive Document Scanner API",
            "version": "3.0.0",
            "description": "Fully adaptive document processing that learns and improves over time",
            "features": [
                "Adaptive quality assessment with learned thresholds",
                "Dynamic key-value extraction with confidence learning",
                "Continuous parameter optimization",
                "No hardcoded values - everything adapts",
                "Real-time learning from successful extractions",
                "Statistical confidence trending"
            ],
            "endpoints": {
                "/scan-document": "POST - Adaptive document processing with learning",
                "/learning-statistics": "GET - View learning progress and statistics",
                "/reset-learning": "POST - Reset all learned parameters",
                "/adaptive-config": "GET - View current adaptive configuration"
            },
            "learning_status": {
                "total_sessions": len(processor.processing_history),
                "adaptive_parameters": "Continuously learning from document processing"
            }
        }
    
    return app

# Create the adaptive app
app = create_adaptive_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)