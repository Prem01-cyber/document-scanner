# Hybrid Document Processor
# Updated processor that uses hybrid extraction (adaptive + LLM fallback)

import numpy as np
import cv2
from typing import Dict, List, Union
import logging
import asyncio
import time
from fastapi import HTTPException

from adaptive_quality_checker import AdaptiveDocumentQualityChecker
from hybrid_kv_extractor import HybridKeyValueExtractor, ExtractionStrategy, HybridExtractionResult
from llm_kv_extractor import LLMProvider, LLMKeyValueExtractor, LLMKeyValuePair
from adaptive_kv_extractor import KeyValuePair
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
    
    def extract_text_with_bounds(self, image_bytes: bytes):
        """Extract text with bounding boxes - returns both structured blocks and raw text"""
        try:
            from google.cloud import vision
            from adaptive_kv_extractor import TextBlock, BoundingBox
            
            image = vision.Image(content=image_bytes)
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
            
            return text_blocks, raw_text
            
        except Exception as e:
            logger.error(f"OCR processing error: {e}")
            raise HTTPException(status_code=500, detail=f"OCR failed: {e}")

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
                 enable_learning: bool = True):
        
        # Initialize components
        self.quality_checker = AdaptiveDocumentQualityChecker()
        self.ocr_processor = GoogleOCRProcessor()
        
        # Initialize hybrid extractor with configuration
        self.kv_extractor = HybridKeyValueExtractor(
            strategy=extraction_strategy,
            adaptive_confidence_threshold=adaptive_confidence_threshold,
            min_pairs_threshold=min_pairs_threshold,
            llm_provider=llm_provider,
            enable_learning=enable_learning
        )
        
        # Processing statistics
        self.processing_history = []
        self.total_processing_time = 0
        self.successful_extractions = 0
        
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
        quality_result = self.quality_checker.assess_quality(image)
        
        if quality_result["needs_rescan"]:
            processing_audit.append("❌ Quality insufficient - requesting rescan")
            return {
                "status": "rescan_needed",
                "quality_assessment": quality_result,
                "message": "Document quality is insufficient. Please rescan.",
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
            text_blocks, raw_text = self.ocr_processor.extract_text_with_bounds(image_bytes)
            ocr_time = time.time() - ocr_start
            
            processing_audit.append(f"✅ OCR completed: {len(text_blocks)} blocks, {len(raw_text)} chars in {ocr_time:.3f}s")
            
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
            "key_value_pairs": self._format_extraction_results(hybrid_result.pairs),
            "extraction_metadata": {
                "primary_method": hybrid_result.primary_method,
                "fallback_used": hybrid_result.fallback_used,
                "extraction_time_seconds": hybrid_result.extraction_time_seconds,
                "confidence_scores": hybrid_result.confidence_scores,
                "method_comparison": hybrid_result.method_comparison,
                "strategy_used": self.kv_extractor.strategy.value
            },
            "processing_statistics": extraction_stats,
            "processing_time_seconds": total_processing_time,
            "processing_audit": processing_audit,
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