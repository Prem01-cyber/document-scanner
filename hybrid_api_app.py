# Hybrid Document Scanner FastAPI Application
# Complete API application using the hybrid extraction system

import os
import json
import logging
from typing import Optional
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from hybrid_document_processor import HybridDocumentProcessor
from hybrid_kv_extractor import ExtractionStrategy
from llm_kv_extractor import LLMProvider
from config import adaptive_config

logger = logging.getLogger(__name__)

# Configuration models for API
class ExtractionConfig(BaseModel):
    strategy: Optional[str] = "adaptive_first"
    confidence_threshold: Optional[float] = 0.5
    min_pairs_threshold: Optional[int] = 2
    llm_provider: Optional[str] = "openai"
    document_type: Optional[str] = "document"

class StrategyUpdate(BaseModel):
    strategy: str
    confidence_threshold: Optional[float] = None
    min_pairs_threshold: Optional[int] = None
    llm_provider: Optional[str] = None

def create_hybrid_api() -> FastAPI:
    """
    Create the hybrid document scanner FastAPI application
    """
    
    app = FastAPI(
        title="Hybrid Document Scanner API",
        version="4.0.0",
        description="Intelligent document processing with hybrid adaptive + LLM extraction"
    )
    
    # Initialize the hybrid processor
    processor = HybridDocumentProcessor(
        extraction_strategy=ExtractionStrategy.ADAPTIVE_FIRST,
        llm_provider=LLMProvider.OPENAI,
        enable_learning=True
    )
    
    @app.post("/scan-document")
    async def hybrid_scan_document(
        file: UploadFile = File(...),
        strategy: str = Query("adaptive_first", description="Extraction strategy"),
        document_type: str = Query("document", description="Type of document being processed"),
        confidence_threshold: float = Query(0.5, description="Confidence threshold for adaptive extraction"),
        min_pairs: int = Query(2, description="Minimum pairs threshold")
    ):
        """
        🧠 Hybrid document scanning with intelligent extraction strategy
        
        Combines adaptive spatial analysis with LLM fallback for robust extraction
        """
        try:
            # Validate file
            if not file.content_type.startswith('image/'):
                raise HTTPException(status_code=400, detail="File must be an image")
            
            image_bytes = await file.read()
            if len(image_bytes) == 0:
                raise HTTPException(status_code=400, detail="Empty file")
            
            # Update processor configuration for this request
            try:
                strategy_enum = ExtractionStrategy(strategy)
                processor.kv_extractor.set_strategy(strategy_enum)
                processor.kv_extractor.adaptive_confidence_threshold = confidence_threshold
                processor.kv_extractor.min_pairs_threshold = min_pairs
                
                logger.info(f"Processing with strategy: {strategy}, threshold: {confidence_threshold}")
                
            except ValueError:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Invalid strategy. Use: {[s.value for s in ExtractionStrategy]}"
                )
            
            # Process document with hybrid system
            result = processor.process_image_bytes(image_bytes, document_type)
            
            # Ensure JSON serializable
            def ensure_serializable(obj):
                if hasattr(obj, '__dict__'):
                    return {key: ensure_serializable(value) for key, value in obj.__dict__.items()}
                elif isinstance(obj, dict):
                    return {key: ensure_serializable(value) for key, value in obj.items()}
                elif isinstance(obj, list):
                    return [ensure_serializable(item) for item in obj]
                elif isinstance(obj, (int, float, str, bool, type(None))):
                    return obj
                else:
                    return str(obj)
            
            serializable_result = ensure_serializable(result)
            
            return JSONResponse(content=serializable_result)
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Hybrid processing error: {e}")
            raise HTTPException(status_code=500, detail=f"Processing failed: {e}")
    
    @app.post("/scan-adaptive-only")
    async def scan_adaptive_only(
        file: UploadFile = File(...),
        document_type: str = Query("document", description="Document type")
    ):
        """
        🔧 Adaptive-only extraction (no LLM fallback)
        
        Uses only the adaptive spatial analysis system
        """
        try:
            if not file.content_type.startswith('image/'):
                raise HTTPException(status_code=400, detail="File must be an image")
            
            image_bytes = await file.read()
            
            # Set to adaptive-only strategy
            original_strategy = processor.kv_extractor.strategy
            processor.kv_extractor.set_strategy(ExtractionStrategy.ADAPTIVE_ONLY)
            
            try:
                result = processor.process_image_bytes(image_bytes, document_type)
                return JSONResponse(content=result)
            finally:
                # Restore original strategy
                processor.kv_extractor.set_strategy(original_strategy)
                
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Adaptive-only processing error: {e}")
            raise HTTPException(status_code=500, detail=f"Processing failed: {e}")
    
    @app.post("/scan-llm-only")
    async def scan_llm_only(
        file: UploadFile = File(...),
        document_type: str = Query("document", description="Document type"),
        llm_provider: str = Query("openai", description="LLM provider to use")
    ):
        """
        🤖 LLM-only extraction (no adaptive analysis)
        
        Uses only large language model for text understanding
        """
        try:
            if not file.content_type.startswith('image/'):
                raise HTTPException(status_code=400, detail="File must be an image")
            
            image_bytes = await file.read()
            
            # Validate and set LLM provider
            try:
                provider_enum = LLMProvider(llm_provider)
                if provider_enum not in processor.kv_extractor.llm_extractor.available_providers:
                    available = list(processor.kv_extractor.llm_extractor.available_providers.keys())
                    raise HTTPException(
                        status_code=400,
                        detail=f"LLM provider '{llm_provider}' not available. Available: {[p.value for p in available]}"
                    )
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid LLM provider. Use: {[p.value for p in LLMProvider]}"
                )
            
            # Set to LLM-only strategy
            original_strategy = processor.kv_extractor.strategy
            processor.kv_extractor.set_strategy(ExtractionStrategy.LLM_ONLY)
            
            try:
                result = processor.process_image_bytes(image_bytes, document_type)
                return JSONResponse(content=result)
            finally:
                # Restore original strategy
                processor.kv_extractor.set_strategy(original_strategy)
                
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"LLM-only processing error: {e}")
            raise HTTPException(status_code=500, detail=f"Processing failed: {e}")
    
    @app.post("/scan-parallel")
    async def scan_parallel(
        file: UploadFile = File(...),
        document_type: str = Query("document", description="Document type")
    ):
        """
        ⚡ Parallel extraction with intelligent merging
        
        Runs both adaptive and LLM extraction simultaneously and merges results
        """
        try:
            if not file.content_type.startswith('image/'):
                raise HTTPException(status_code=400, detail="File must be an image")
            
            image_bytes = await file.read()
            
            # Set to parallel strategy
            original_strategy = processor.kv_extractor.strategy
            processor.kv_extractor.set_strategy(ExtractionStrategy.PARALLEL)
            
            try:
                result = processor.process_image_bytes(image_bytes, document_type)
                return JSONResponse(content=result)
            finally:
                # Restore original strategy
                processor.kv_extractor.set_strategy(original_strategy)
                
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Parallel processing error: {e}")
            raise HTTPException(status_code=500, detail=f"Processing failed: {e}")
    
    @app.get("/analytics")
    async def get_processing_analytics():
        """
        📊 Get comprehensive processing analytics and performance metrics
        """
        try:
            analytics = processor.get_processing_analytics()
            return JSONResponse(content=analytics)
        except Exception as e:
            logger.error(f"Analytics error: {e}")
            raise HTTPException(status_code=500, detail=f"Analytics failed: {e}")
    
    @app.get("/hybrid-performance")
    async def get_hybrid_performance():
        """
        📈 Get hybrid system performance statistics
        """
        try:
            performance = processor.kv_extractor.get_performance_statistics()
            return JSONResponse(content=performance)
        except Exception as e:
            logger.error(f"Performance stats error: {e}")
            raise HTTPException(status_code=500, detail=f"Performance stats failed: {e}")
    
    @app.post("/optimize-strategy")
    async def optimize_extraction_strategy():
        """
        🎯 Automatically optimize extraction strategy based on performance
        """
        try:
            optimization_result = processor.optimize_extraction_strategy()
            return JSONResponse(content=optimization_result)
        except Exception as e:
            logger.error(f"Strategy optimization error: {e}")
            raise HTTPException(status_code=500, detail=f"Optimization failed: {e}")
    
    @app.post("/update-configuration")
    async def update_configuration(config: StrategyUpdate):
        """
        ⚙️  Update hybrid extraction configuration
        """
        try:
            # Parse and validate parameters
            strategy_enum = None
            if config.strategy:
                try:
                    strategy_enum = ExtractionStrategy(config.strategy)
                except ValueError:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Invalid strategy. Use: {[s.value for s in ExtractionStrategy]}"
                    )
            
            llm_provider_enum = None
            if config.llm_provider:
                try:
                    llm_provider_enum = LLMProvider(config.llm_provider)
                except ValueError:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Invalid LLM provider. Use: {[p.value for p in LLMProvider]}"
                    )
            
            # Update configuration
            result = processor.update_configuration(
                strategy=strategy_enum,
                confidence_threshold=config.confidence_threshold,
                min_pairs_threshold=config.min_pairs_threshold,
                llm_provider=llm_provider_enum
            )
            
            return JSONResponse(content=result)
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Configuration update error: {e}")
            raise HTTPException(status_code=500, detail=f"Configuration update failed: {e}")
    
    @app.get("/system-status")
    async def get_system_status():
        """
        🔍 Get comprehensive system status and capabilities
        """
        try:
            # Get LLM provider information
            llm_info = processor.kv_extractor.llm_extractor.get_provider_info()
            llm_available = processor.kv_extractor.llm_extractor.is_available()
            
            # Get adaptive config status
            adaptive_stats = adaptive_config.config
            
            # Get current processor configuration
            current_config = {
                "strategy": processor.kv_extractor.strategy.value,
                "confidence_threshold": processor.kv_extractor.adaptive_confidence_threshold,
                "min_pairs_threshold": processor.kv_extractor.min_pairs_threshold,
                "learning_enabled": processor.kv_extractor.enable_learning
            }
            
            status = {
                "system_version": "4.0.0",
                "hybrid_system_active": True,
                "components_status": {
                    "adaptive_quality_checker": "operational",
                    "google_ocr": "operational",
                    "adaptive_kv_extractor": "operational",
                    "llm_kv_extractor": "operational" if llm_available else "limited",
                    "hybrid_coordinator": "operational"
                },
                "llm_capabilities": {
                    "available": llm_available,
                    "providers": llm_info,
                    "primary_provider": processor.kv_extractor.llm_extractor.primary_provider.value
                },
                "adaptive_learning": {
                    "enabled": processor.kv_extractor.enable_learning,
                    "parameters_learned": sum(
                        1 for category in adaptive_stats.values() 
                        if isinstance(category, dict)
                        for param in category.values()
                        if isinstance(param, dict) and param.get("confidence_weight", 0) > 0.1
                    ),
                    "total_sessions": len(processor.processing_history)
                },
                "current_configuration": current_config,
                "supported_strategies": [s.value for s in ExtractionStrategy],
                "supported_llm_providers": [p.value for p in LLMProvider],
                "processing_statistics": {
                    "total_extractions": len(processor.processing_history),
                    "successful_extractions": processor.successful_extractions,
                    "total_processing_time": processor.total_processing_time
                }
            }
            
            return JSONResponse(content=status)
            
        except Exception as e:
            logger.error(f"System status error: {e}")
            raise HTTPException(status_code=500, detail=f"System status failed: {e}")
    
    @app.get("/available-strategies")
    async def get_available_strategies():
        """
        📋 Get information about available extraction strategies
        """
        strategies_info = {
            "adaptive_first": {
                "description": "Try adaptive extraction first, fallback to LLM if insufficient",
                "best_for": ["structured forms", "documents with clear layouts", "fast processing"],
                "pros": ["Fast", "Interpretable", "Good spatial understanding"],
                "cons": ["May struggle with complex layouts", "Requires structured text"]
            },
            "llm_first": {
                "description": "Try LLM extraction first, fallback to adaptive if insufficient",
                "best_for": ["unstructured documents", "complex layouts", "unusual formats"],
                "pros": ["Robust", "Handles complexity well", "Context understanding"],
                "cons": ["Slower", "Requires API access", "Higher cost"]
            },
            "parallel": {
                "description": "Run both methods simultaneously and merge results intelligently",
                "best_for": ["maximum accuracy", "comprehensive extraction", "unknown document types"],
                "pros": ["Best coverage", "Combines strengths", "High accuracy"],
                "cons": ["Higher resource usage", "Longer processing time"]
            },
            "confidence_based": {
                "description": "Analyze document and choose the best method automatically",
                "best_for": ["mixed document types", "automated processing", "optimal efficiency"],
                "pros": ["Intelligent selection", "Adaptive", "Efficient"],
                "cons": ["Prediction may not always be optimal"]
            },
            "adaptive_only": {
                "description": "Use only adaptive spatial analysis (no LLM)",
                "best_for": ["structured documents", "when LLM is unavailable", "cost-sensitive applications"],
                "pros": ["Fast", "No external dependencies", "Free"],
                "cons": ["Limited to structured layouts"]
            },
            "llm_only": {
                "description": "Use only LLM extraction (no adaptive analysis)",
                "best_for": ["unstructured text", "complex documents", "when spatial info not important"],
                "pros": ["Handles any text format", "Context aware"],
                "cons": ["No spatial information", "Requires API access"]
            }
        }
        
        return {
            "available_strategies": list(strategies_info.keys()),
            "strategy_details": strategies_info,
            "current_strategy": processor.kv_extractor.strategy.value,
            "recommendation": "Use 'adaptive_first' for most structured documents, 'parallel' for maximum accuracy"
        }
    
    @app.get("/")
    async def root():
        """
        🏠 Root endpoint with comprehensive API information
        """
        return {
            "service": "Hybrid Document Scanner API",
            "version": "4.0.0",
            "description": "Intelligent document processing with hybrid adaptive + LLM extraction",
            "key_features": [
                "🧠 Hybrid extraction system (adaptive + LLM)",
                "🔧 Adaptive spatial analysis with learned parameters",
                "🤖 Multiple LLM provider support (OpenAI, Anthropic, Ollama)",
                "⚡ Parallel processing with intelligent merging",
                "📊 Real-time performance analytics and optimization",
                "🎯 Automatic strategy selection and tuning",
                "📈 Continuous learning from successful extractions"
            ],
            "extraction_methods": {
                "adaptive": "Spatial analysis with learned confidence parameters",
                "llm": "Large language model text understanding",
                "hybrid": "Intelligent combination of both methods"
            },
            "api_endpoints": {
                "processing": {
                    "/scan-document": "🧠 Main hybrid extraction endpoint",
                    "/scan-adaptive-only": "🔧 Adaptive-only extraction",
                    "/scan-llm-only": "🤖 LLM-only extraction",
                    "/scan-parallel": "⚡ Parallel extraction with merging"
                },
                "analytics": {
                    "/analytics": "📊 Processing analytics",
                    "/hybrid-performance": "📈 Hybrid system performance",
                    "/system-status": "🔍 Comprehensive system status"
                },
                "configuration": {
                    "/optimize-strategy": "🎯 Auto-optimize extraction strategy",
                    "/update-configuration": "⚙️ Update system configuration",
                    "/available-strategies": "📋 Strategy information"
                }
            },
            "supported_strategies": [s.value for s in ExtractionStrategy],
            "supported_llm_providers": [p.value for p in LLMProvider],
            "current_status": {
                "strategy": processor.kv_extractor.strategy.value,
                "llm_available": processor.kv_extractor.llm_extractor.is_available(),
                "learning_enabled": processor.kv_extractor.enable_learning,
                "total_sessions": len(processor.processing_history)
            }
        }
    
    return app

# Create the application
app = create_hybrid_api()

if __name__ == "__main__":
    import uvicorn
    
    # Check environment setup
    print("🚀 Starting Hybrid Document Scanner API")
    print("=" * 50)
    
    # Check for required environment variables
    required_env = {
        "GOOGLE_APPLICATION_CREDENTIALS": "Google Cloud Vision API",
        "OPENAI_API_KEY": "OpenAI GPT (optional)",
        "ANTHROPIC_API_KEY": "Anthropic Claude (optional)"
    }
    
    print("📋 Environment Check:")
    for var, description in required_env.items():
        status = "✅ SET" if os.getenv(var) else "❌ NOT SET"
        print(f"   {var}: {status} ({description})")
    
    print("\n🔧 System Components:")
    try:
        processor = HybridDocumentProcessor()
        
        # Check LLM availability
        llm_providers = processor.kv_extractor.llm_extractor.get_provider_info()
        print(f"   Available LLM providers: {list(llm_providers.keys()) if llm_providers else 'None'}")
        
        # Check adaptive config
        config_params = sum(
            1 for category in adaptive_config.config.values() 
            if isinstance(category, dict)
            for param in category.values()
            if isinstance(param, dict) and param.get("confidence_weight", 0) > 0.1
        )
        print(f"   Learned parameters: {config_params}")
        
    except Exception as e:
        print(f"   ⚠️  Initialization warning: {e}")
    
    print(f"\n🌐 Starting server on http://localhost:8000")
    print("📚 API docs available at: http://localhost:8000/docs")
    print("🎯 Try the hybrid extraction at: http://localhost:8000/scan-document")
    
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")