# Hybrid Key-Value Extractor
# Coordinates between adaptive extraction and LLM fallback for optimal results

import time
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
import logging
from dataclasses import dataclass
from enum import Enum

from .adaptive_kv_extractor import AdaptiveKeyValueExtractor, TextBlock, KeyValuePair
from .llm_kv_extractor import LLMKeyValueExtractor, LLMKeyValuePair, LLMProvider
from prometheus_client import Histogram

logger = logging.getLogger(__name__)

class ExtractionStrategy(Enum):
    """Extraction strategies for hybrid system"""
    ADAPTIVE_ONLY = "adaptive_only"
    LLM_ONLY = "llm_only"
    ADAPTIVE_FIRST = "adaptive_first"  # Try adaptive, fallback to LLM
    LLM_FIRST = "llm_first"  # Try LLM, fallback to adaptive
    PARALLEL = "parallel"  # Run both and merge
    CONFIDENCE_BASED = "confidence_based"  # Choose based on confidence

@dataclass
class HybridExtractionResult:
    """Result from hybrid extraction with detailed metadata"""
    pairs: List[Union[KeyValuePair, LLMKeyValuePair]]
    primary_method: str
    fallback_used: bool
    extraction_time_seconds: float
    confidence_scores: Dict[str, float]
    method_comparison: Dict[str, Any]
    audit_trail: List[str]
    llm_usage: Optional[Dict[str, Any]] = None

class HybridKeyValueExtractor:
    """
    Intelligent hybrid extractor that combines adaptive and LLM methods
    """
    
    def __init__(self, 
                 strategy: ExtractionStrategy = ExtractionStrategy.ADAPTIVE_FIRST,
                 adaptive_confidence_threshold: float = 0.5,
                 min_pairs_threshold: int = 2,
                 llm_provider: LLMProvider = LLMProvider.OPENAI,
                 llm_request_timeout_seconds: int = 40,
                 llm_max_retries: int = 2,
                 enable_learning: bool = True):
        
        self.strategy = strategy
        self.adaptive_confidence_threshold = adaptive_confidence_threshold
        self.min_pairs_threshold = min_pairs_threshold
        self.enable_learning = enable_learning
        
        # Initialize extractors
        self.adaptive_extractor = AdaptiveKeyValueExtractor()
        self.llm_extractor = LLMKeyValueExtractor(
            primary_provider=llm_provider,
            request_timeout_seconds=llm_request_timeout_seconds,
            max_retries=llm_max_retries,
        )

        # Metrics
        self._hist_adaptive = Histogram("scanner_adaptive_seconds", "Adaptive extraction latency in seconds")
        self._hist_llm = Histogram("scanner_llm_seconds", "LLM extraction latency in seconds")
        
        # Performance tracking
        self.performance_stats = {
            "total_extractions": 0,
            "adaptive_successes": 0,
            "llm_fallbacks": 0,
            "parallel_runs": 0,
            "average_confidence": 0.0,
            "method_preference_learned": {},
            "processing_times": {
                "adaptive": [],
                "llm": [],
                "hybrid": []
            }
        }
        
        # Learning data for strategy optimization
        self.extraction_history = []
        
    def extract_key_value_pairs(self, 
                               text_blocks: List[TextBlock], 
                               raw_text: Optional[str] = None,
                               document_type: str = "document") -> HybridExtractionResult:
        """
        Main hybrid extraction method that intelligently chooses the best approach
        """
        start_time = time.time()
        self.performance_stats["total_extractions"] += 1
        
        audit_trail = []
        audit_trail.append(f"🔄 Starting hybrid extraction with strategy: {self.strategy.value}")
        
        # Ensure we have raw text for LLM extraction
        if raw_text is None:
            raw_text = " ".join([block.text for block in text_blocks])
        
        # Choose extraction strategy
        if self.strategy == ExtractionStrategy.ADAPTIVE_FIRST:
            result = self._adaptive_first_strategy(text_blocks, raw_text, document_type, audit_trail)
        
        elif self.strategy == ExtractionStrategy.LLM_FIRST:
            result = self._llm_first_strategy(text_blocks, raw_text, document_type, audit_trail)
        
        elif self.strategy == ExtractionStrategy.PARALLEL:
            result = self._parallel_strategy(text_blocks, raw_text, document_type, audit_trail)
        
        elif self.strategy == ExtractionStrategy.CONFIDENCE_BASED:
            result = self._confidence_based_strategy(text_blocks, raw_text, document_type, audit_trail)
        
        elif self.strategy == ExtractionStrategy.ADAPTIVE_ONLY:
            result = self._adaptive_only_strategy(text_blocks, audit_trail)
        
        elif self.strategy == ExtractionStrategy.LLM_ONLY:
            result = self._llm_only_strategy(raw_text, document_type, audit_trail)
        
        else:
            # Default to adaptive first
            result = self._adaptive_first_strategy(text_blocks, raw_text, document_type, audit_trail)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        result.extraction_time_seconds = processing_time
        result.audit_trail = audit_trail
        
        # Update performance statistics
        self._update_performance_stats(result, processing_time)
        
        # Learn from this extraction
        if self.enable_learning:
            self._learn_from_extraction(result, text_blocks, raw_text, document_type)
        
        audit_trail.append(f"✅ Hybrid extraction completed in {processing_time:.3f}s")
        audit_trail.append(f"📊 Extracted {len(result.pairs)} pairs using {result.primary_method}")
        
        return result
    
    def _adaptive_first_strategy(self, text_blocks: List[TextBlock], raw_text: str, 
                               document_type: str, audit_trail: List[str]) -> HybridExtractionResult:
        """
        Try adaptive extraction first, fallback to LLM if insufficient
        """
        audit_trail.append("🎯 Strategy: Adaptive first with LLM fallback")
        
        # Try adaptive extraction
        adaptive_start = time.time()
        adaptive_pairs = self.adaptive_extractor.extract_key_value_pairs(text_blocks)
        adaptive_time = time.time() - adaptive_start
        try:
            self._hist_adaptive.observe(adaptive_time)
        except Exception:
            pass
        
        audit_trail.append(f"🔧 Adaptive extraction: {len(adaptive_pairs)} pairs in {adaptive_time:.3f}s")
        
        # Calculate adaptive confidence
        adaptive_confidence = self._calculate_confidence(adaptive_pairs)
        audit_trail.append(f"📈 Adaptive confidence: {adaptive_confidence:.3f}")
        
        # Check if adaptive extraction is sufficient
        if (len(adaptive_pairs) >= self.min_pairs_threshold and 
            adaptive_confidence >= self.adaptive_confidence_threshold):
            
            audit_trail.append("✅ Adaptive extraction sufficient - using adaptive results")
            self.performance_stats["adaptive_successes"] += 1
            
            return HybridExtractionResult(
                pairs=adaptive_pairs,
                primary_method="adaptive",
                fallback_used=False,
                extraction_time_seconds=adaptive_time,
                confidence_scores={"adaptive": adaptive_confidence},
                method_comparison={"adaptive_pairs": len(adaptive_pairs)},
                audit_trail=[]
            )
        
        # Adaptive insufficient - try LLM fallback
        audit_trail.append("⚠️  Adaptive extraction insufficient - falling back to LLM")
        
        if not self.llm_extractor.is_available():
            audit_trail.append("❌ LLM not available - returning adaptive results anyway")
            return HybridExtractionResult(
                pairs=adaptive_pairs,
                primary_method="adaptive_forced",
                fallback_used=False,
                extraction_time_seconds=adaptive_time,
                confidence_scores={"adaptive": adaptive_confidence},
                method_comparison={"adaptive_pairs": len(adaptive_pairs), "llm_available": False},
                audit_trail=[]
            )
        
        # Run LLM extraction
        llm_start = time.time()
        llm_pairs = self.llm_extractor.extract_key_value_pairs(raw_text, document_type)
        llm_time = time.time() - llm_start
        try:
            self._hist_llm.observe(llm_time)
        except Exception:
            pass
        
        audit_trail.append(f"🤖 LLM extraction: {len(llm_pairs)} pairs in {llm_time:.3f}s")
        
        llm_confidence = self._calculate_llm_confidence(llm_pairs)
        audit_trail.append(f"📈 LLM confidence: {llm_confidence:.3f}")
        
        self.performance_stats["llm_fallbacks"] += 1
        
        # Choose best result
        if len(llm_pairs) > len(adaptive_pairs) and llm_confidence >= 0.6:
            audit_trail.append("🤖 Using LLM results (better coverage)")
            primary_method = "llm_fallback"
            final_pairs = llm_pairs
        else:
            audit_trail.append("🔧 Using adaptive results (LLM didn't improve)")
            primary_method = "adaptive_fallback"
            final_pairs = adaptive_pairs
        
        # Build LLM usage metadata if available
        llm_usage = None
        if llm_pairs:
            meta = getattr(llm_pairs[0], "response_metadata", {}) or {}
            llm_usage = {
                "model": meta.get("model"),
                "tokens_input": meta.get("tokens_input"),
                "tokens_output": meta.get("tokens_output"),
                "estimated_cost_usd": meta.get("estimated_cost_usd"),
                "latency_seconds": llm_time,
            }

        return HybridExtractionResult(
            pairs=final_pairs,
            primary_method=primary_method,
            fallback_used=True,
            extraction_time_seconds=adaptive_time + llm_time,
            confidence_scores={
                "adaptive": adaptive_confidence,
                "llm": llm_confidence
            },
            method_comparison={
                "adaptive_pairs": len(adaptive_pairs),
                "llm_pairs": len(llm_pairs),
                "adaptive_time": adaptive_time,
                "llm_time": llm_time
            },
            audit_trail=[],
            llm_usage=llm_usage,
        )
    
    def _llm_first_strategy(self, text_blocks: List[TextBlock], raw_text: str, 
                          document_type: str, audit_trail: List[str]) -> HybridExtractionResult:
        """
        Try LLM extraction first, fallback to adaptive if insufficient
        """
        audit_trail.append("🤖 Strategy: LLM first with adaptive fallback")
        
        if not self.llm_extractor.is_available():
            audit_trail.append("❌ LLM not available - falling back to adaptive")
            return self._adaptive_only_strategy(text_blocks, audit_trail)
        
        # Try LLM extraction first
        llm_start = time.time()
        llm_pairs = self.llm_extractor.extract_key_value_pairs(raw_text, document_type)
        llm_time = time.time() - llm_start
        
        audit_trail.append(f"🤖 LLM extraction: {len(llm_pairs)} pairs in {llm_time:.3f}s")
        
        llm_confidence = self._calculate_llm_confidence(llm_pairs)
        audit_trail.append(f"📈 LLM confidence: {llm_confidence:.3f}")
        
        # Check if LLM extraction is sufficient
        if len(llm_pairs) >= self.min_pairs_threshold and llm_confidence >= 0.6:
            audit_trail.append("✅ LLM extraction sufficient - using LLM results")
            
            return HybridExtractionResult(
                pairs=llm_pairs,
                primary_method="llm",
                fallback_used=False,
                extraction_time_seconds=llm_time,
                confidence_scores={"llm": llm_confidence},
                method_comparison={"llm_pairs": len(llm_pairs)},
                audit_trail=[]
            )
        
        # LLM insufficient - try adaptive fallback
        audit_trail.append("⚠️  LLM extraction insufficient - falling back to adaptive")
        
        adaptive_start = time.time()
        adaptive_pairs = self.adaptive_extractor.extract_key_value_pairs(text_blocks)
        adaptive_time = time.time() - adaptive_start
        try:
            self._hist_adaptive.observe(adaptive_time)
        except Exception:
            pass
        
        audit_trail.append(f"🔧 Adaptive extraction: {len(adaptive_pairs)} pairs in {adaptive_time:.3f}s")
        
        adaptive_confidence = self._calculate_confidence(adaptive_pairs)
        
        # Choose best result
        if len(adaptive_pairs) > len(llm_pairs):
            final_pairs = adaptive_pairs
            primary_method = "adaptive_fallback"
            audit_trail.append("🔧 Using adaptive results (better coverage)")
        else:
            final_pairs = llm_pairs
            primary_method = "llm_fallback"
            audit_trail.append("🤖 Using LLM results (adaptive didn't improve)")
        
        return HybridExtractionResult(
            pairs=final_pairs,
            primary_method=primary_method,
            fallback_used=True,
            extraction_time_seconds=llm_time + adaptive_time,
            confidence_scores={
                "llm": llm_confidence,
                "adaptive": adaptive_confidence
            },
            method_comparison={
                "llm_pairs": len(llm_pairs),
                "adaptive_pairs": len(adaptive_pairs)
            },
            audit_trail=[]
        )
    
    def _parallel_strategy(self, text_blocks: List[TextBlock], raw_text: str, 
                         document_type: str, audit_trail: List[str]) -> HybridExtractionResult:
        """
        Run both methods in parallel and merge results intelligently
        """
        audit_trail.append("⚡ Strategy: Parallel extraction with intelligent merging")
        
        # Run adaptive extraction
        adaptive_start = time.time()
        adaptive_pairs = self.adaptive_extractor.extract_key_value_pairs(text_blocks)
        adaptive_time = time.time() - adaptive_start
        
        audit_trail.append(f"🔧 Adaptive extraction: {len(adaptive_pairs)} pairs in {adaptive_time:.3f}s")
        
        # Run LLM extraction (if available)
        llm_pairs = []
        llm_time = 0
        llm_confidence = 0
        
        if self.llm_extractor.is_available():
            llm_start = time.time()
            llm_pairs = self.llm_extractor.extract_key_value_pairs(raw_text, document_type)
            llm_time = time.time() - llm_start
            try:
                self._hist_llm.observe(llm_time)
            except Exception:
                pass
            audit_trail.append(f"🤖 LLM extraction: {len(llm_pairs)} pairs in {llm_time:.3f}s")
            llm_confidence = self._calculate_llm_confidence(llm_pairs)
        else:
            audit_trail.append("❌ LLM not available - using only adaptive results")
        
        # Merge results intelligently
        merged_pairs = self._merge_extraction_results(adaptive_pairs, llm_pairs, audit_trail)
        
        adaptive_confidence = self._calculate_confidence(adaptive_pairs)
        
        self.performance_stats["parallel_runs"] += 1
        
        # LLM usage metadata
        llm_usage = None
        if llm_pairs:
            meta = getattr(llm_pairs[0], "response_metadata", {}) or {}
            llm_usage = {
                "model": meta.get("model"),
                "tokens_input": meta.get("tokens_input"),
                "tokens_output": meta.get("tokens_output"),
                "estimated_cost_usd": meta.get("estimated_cost_usd"),
                "latency_seconds": llm_time,
            }

        return HybridExtractionResult(
            pairs=merged_pairs,
            primary_method="parallel_merged",
            fallback_used=False,
            extraction_time_seconds=max(adaptive_time, llm_time),  # Parallel time
            confidence_scores={
                "adaptive": adaptive_confidence,
                "llm": llm_confidence,
                "merged": self._calculate_merged_confidence(merged_pairs)
            },
            method_comparison={
                "adaptive_pairs": len(adaptive_pairs),
                "llm_pairs": len(llm_pairs),
                "merged_pairs": len(merged_pairs),
                "adaptive_time": adaptive_time,
                "llm_time": llm_time
            },
            audit_trail=[],
            llm_usage=llm_usage,
        )
    
    def _confidence_based_strategy(self, text_blocks: List[TextBlock], raw_text: str, 
                                 document_type: str, audit_trail: List[str]) -> HybridExtractionResult:
        """
        Quick assessment to choose the best method for this document
        """
        audit_trail.append("🧠 Strategy: Confidence-based method selection")
        
        # Quick document analysis to predict best method
        predicted_method = self._predict_best_method(text_blocks, raw_text, audit_trail)
        
        if predicted_method == "adaptive":
            audit_trail.append("🔧 Predicted method: Adaptive (structured document)")
            return self._adaptive_first_strategy(text_blocks, raw_text, document_type, audit_trail)
        else:
            audit_trail.append("🤖 Predicted method: LLM (unstructured/complex document)")
            return self._llm_first_strategy(text_blocks, raw_text, document_type, audit_trail)
    
    def _adaptive_only_strategy(self, text_blocks: List[TextBlock], 
                              audit_trail: List[str]) -> HybridExtractionResult:
        """
        Use only adaptive extraction
        """
        audit_trail.append("🔧 Strategy: Adaptive only")
        
        adaptive_start = time.time()
        adaptive_pairs = self.adaptive_extractor.extract_key_value_pairs(text_blocks)
        adaptive_time = time.time() - adaptive_start
        
        adaptive_confidence = self._calculate_confidence(adaptive_pairs)
        
        return HybridExtractionResult(
            pairs=adaptive_pairs,
            primary_method="adaptive_only",
            fallback_used=False,
            extraction_time_seconds=adaptive_time,
            confidence_scores={"adaptive": adaptive_confidence},
            method_comparison={"adaptive_pairs": len(adaptive_pairs)},
            audit_trail=[]
        )
    
    def _llm_only_strategy(self, raw_text: str, document_type: str, 
                         audit_trail: List[str]) -> HybridExtractionResult:
        """
        Use only LLM extraction
        """
        audit_trail.append("🤖 Strategy: LLM only")
        
        if not self.llm_extractor.is_available():
            audit_trail.append("❌ LLM not available - cannot extract")
            return HybridExtractionResult(
                pairs=[],
                primary_method="llm_unavailable",
                fallback_used=False,
                extraction_time_seconds=0,
                confidence_scores={},
                method_comparison={"error": "LLM not available"},
                audit_trail=[]
            )
        
        llm_start = time.time()
        llm_pairs = self.llm_extractor.extract_key_value_pairs(raw_text, document_type)
        llm_time = time.time() - llm_start
        
        llm_confidence = self._calculate_llm_confidence(llm_pairs)
        
        llm_usage = None
        if llm_pairs:
            meta = getattr(llm_pairs[0], "response_metadata", {}) or {}
            llm_usage = {
                "model": meta.get("model"),
                "tokens_input": meta.get("tokens_input"),
                "tokens_output": meta.get("tokens_output"),
                "estimated_cost_usd": meta.get("estimated_cost_usd"),
                "latency_seconds": llm_time,
            }

        return HybridExtractionResult(
            pairs=llm_pairs,
            primary_method="llm_only",
            fallback_used=False,
            extraction_time_seconds=llm_time,
            confidence_scores={"llm": llm_confidence},
            method_comparison={"llm_pairs": len(llm_pairs)},
            audit_trail=[],
            llm_usage=llm_usage,
        )
    
    def _calculate_confidence(self, pairs: List[KeyValuePair]) -> float:
        """Calculate average confidence for adaptive pairs"""
        if not pairs:
            return 0.0
        return np.mean([pair.confidence for pair in pairs])
    
    def _calculate_llm_confidence(self, pairs: List[LLMKeyValuePair]) -> float:
        """Calculate average confidence for LLM pairs"""
        if not pairs:
            return 0.0
        return np.mean([pair.confidence for pair in pairs])
    
    def _calculate_merged_confidence(self, pairs: List[Union[KeyValuePair, LLMKeyValuePair]]) -> float:
        """Calculate confidence for merged pairs"""
        if not pairs:
            return 0.0
        return np.mean([pair.confidence for pair in pairs])
    
    def _merge_extraction_results(self, adaptive_pairs: List[KeyValuePair], 
                                llm_pairs: List[LLMKeyValuePair], 
                                audit_trail: List[str]) -> List[Union[KeyValuePair, LLMKeyValuePair]]:
        """
        Intelligently merge adaptive and LLM results
        """
        if not llm_pairs:
            audit_trail.append("🔧 No LLM pairs to merge - using adaptive only")
            return adaptive_pairs
        
        if not adaptive_pairs:
            audit_trail.append("🤖 No adaptive pairs to merge - using LLM only")
            return llm_pairs
        
        # Create merged result
        merged_pairs = []
        
        # Start with higher-confidence method as base
        adaptive_conf = self._calculate_confidence(adaptive_pairs)
        llm_conf = self._calculate_llm_confidence(llm_pairs)
        
        if adaptive_conf >= llm_conf:
            audit_trail.append(f"🔧 Using adaptive as base (conf: {adaptive_conf:.3f} vs {llm_conf:.3f})")
            base_pairs = adaptive_pairs
            supplement_pairs = llm_pairs
        else:
            audit_trail.append(f"🤖 Using LLM as base (conf: {llm_conf:.3f} vs {adaptive_conf:.3f})")
            base_pairs = llm_pairs
            supplement_pairs = adaptive_pairs
        
        # Add all base pairs
        merged_pairs.extend(base_pairs)
        
        # Add unique pairs from supplement
        base_keys = {pair.key.lower().strip() for pair in base_pairs}
        
        added_count = 0
        for supp_pair in supplement_pairs:
            supp_key = supp_pair.key.lower().strip()
            
            # Check if this key is already covered
            if supp_key not in base_keys:
                merged_pairs.append(supp_pair)
                base_keys.add(supp_key)
                added_count += 1
        
        audit_trail.append(f"🔀 Merged {len(base_pairs)} base + {added_count} supplemental = {len(merged_pairs)} total pairs")
        
        return merged_pairs
    
    def _predict_best_method(self, text_blocks: List[TextBlock], raw_text: str, 
                           audit_trail: List[str]) -> str:
        """
        Predict which extraction method will work best for this document
        """
        # Simple heuristics for method prediction
        score_adaptive = 0
        score_llm = 0
        
        # Text structure analysis
        total_text = " ".join([block.text for block in text_blocks])
        
        # Count structured indicators
        colons = total_text.count(':')
        lines_with_colons = sum(1 for block in text_blocks if ':' in block.text)
        
        if colons > 3 or lines_with_colons > 2:
            score_adaptive += 2
            audit_trail.append(f"📋 Structured indicators: {colons} colons, {lines_with_colons} lines with colons (+2 adaptive)")
        
        # Text complexity
        avg_block_length = np.mean([len(block.text) for block in text_blocks])
        if avg_block_length > 50:
            score_llm += 1
            audit_trail.append(f"📄 High text complexity: avg {avg_block_length:.1f} chars per block (+1 LLM)")
        else:
            score_adaptive += 1
            audit_trail.append(f"📄 Low text complexity: avg {avg_block_length:.1f} chars per block (+1 adaptive)")
        
        # Block count
        if len(text_blocks) > 20:
            score_llm += 1
            audit_trail.append(f"📊 Many text blocks: {len(text_blocks)} (+1 LLM)")
        else:
            score_adaptive += 1
            audit_trail.append(f"📊 Few text blocks: {len(text_blocks)} (+1 adaptive)")
        
        # Historical performance (if available)
        if hasattr(self, 'method_preference_learned'):
            if 'adaptive' in self.performance_stats.get('method_preference_learned', {}):
                pref = self.performance_stats['method_preference_learned']['adaptive']
                if pref > 0.6:
                    score_adaptive += 1
                    audit_trail.append(f"📈 Historical adaptive success: {pref:.2f} (+1 adaptive)")
        
        predicted = "adaptive" if score_adaptive >= score_llm else "llm"
        audit_trail.append(f"🎯 Method prediction: {predicted} (adaptive: {score_adaptive}, llm: {score_llm})")
        
        return predicted
    
    def _update_performance_stats(self, result: HybridExtractionResult, processing_time: float):
        """Update performance statistics"""
        # Update confidence tracking
        if result.pairs:
            pair_confidences = [pair.confidence for pair in result.pairs]
            current_avg = np.mean(pair_confidences)
            
            # Update rolling average
            total = self.performance_stats["total_extractions"]
            prev_avg = self.performance_stats["average_confidence"]
            self.performance_stats["average_confidence"] = (
                (prev_avg * (total - 1) + current_avg) / total
            )
        
        # Update timing stats
        self.performance_stats["processing_times"]["hybrid"].append(processing_time)
        
        # Limit history size
        for key in self.performance_stats["processing_times"]:
            if len(self.performance_stats["processing_times"][key]) > 100:
                self.performance_stats["processing_times"][key] = \
                    self.performance_stats["processing_times"][key][-100:]
    
    def _learn_from_extraction(self, result: HybridExtractionResult, 
                             text_blocks: List[TextBlock], raw_text: str, document_type: str):
        """Learn from extraction results to improve future predictions"""
        
        # Store extraction history
        history_entry = {
            "timestamp": time.time(),
            "method_used": result.primary_method,
            "pairs_extracted": len(result.pairs),
            "confidence": self._calculate_merged_confidence(result.pairs),
            "processing_time": result.extraction_time_seconds,
            "text_blocks_count": len(text_blocks),
            "text_length": len(raw_text),
            "fallback_used": result.fallback_used
        }
        
        self.extraction_history.append(history_entry)
        
        # Limit history size
        if len(self.extraction_history) > 100:
            self.extraction_history = self.extraction_history[-100:]
        
        # Update method preferences based on success
        if len(result.pairs) > 0 and self._calculate_merged_confidence(result.pairs) > 0.6:
            method_key = result.primary_method.split('_')[0]  # 'adaptive' or 'llm'
            
            if method_key not in self.performance_stats["method_preference_learned"]:
                self.performance_stats["method_preference_learned"][method_key] = 0.5
            
            # Increase preference for successful method
            current_pref = self.performance_stats["method_preference_learned"][method_key]
            self.performance_stats["method_preference_learned"][method_key] = min(1.0, current_pref + 0.1)
    
    def get_performance_statistics(self) -> Dict:
        """Get comprehensive performance statistics"""
        
        adaptive_stats = {}
        llm_stats = {}
        
        # Get adaptive extractor stats if available
        if hasattr(self.adaptive_extractor, 'get_extraction_statistics'):
            try:
                adaptive_stats = self.adaptive_extractor.get_extraction_statistics()
            except:
                pass
        
        # Get LLM extractor stats
        llm_stats = self.llm_extractor.get_extraction_statistics()
        
        return {
            "hybrid_performance": self.performance_stats,
            "extraction_history_count": len(self.extraction_history),
            "current_strategy": self.strategy.value,
            "adaptive_threshold": self.adaptive_confidence_threshold,
            "min_pairs_threshold": self.min_pairs_threshold,
            "component_stats": {
                "adaptive": adaptive_stats,
                "llm": llm_stats
            },
            "recent_performance": self._get_recent_performance_summary()
        }
    
    def _get_recent_performance_summary(self) -> Dict:
        """Get summary of recent performance"""
        if len(self.extraction_history) < 5:
            return {"status": "insufficient_data"}
        
        recent = self.extraction_history[-10:]
        
        return {
            "avg_pairs_extracted": np.mean([entry["pairs_extracted"] for entry in recent]),
            "avg_confidence": np.mean([entry["confidence"] for entry in recent]),
            "avg_processing_time": np.mean([entry["processing_time"] for entry in recent]),
            "fallback_usage_rate": np.mean([entry["fallback_used"] for entry in recent]),
            "method_distribution": {
                method: sum(1 for entry in recent if entry["method_used"].startswith(method))
                for method in ["adaptive", "llm", "parallel"]
            }
        }
    
    def optimize_strategy(self) -> ExtractionStrategy:
        """
        Analyze performance and recommend optimal strategy
        """
        if len(self.extraction_history) < 10:
            return self.strategy  # Not enough data
        
        recent = self.extraction_history[-20:]
        
        # Analyze method performance
        method_performance = {}
        for entry in recent:
            method = entry["method_used"].split('_')[0]
            if method not in method_performance:
                method_performance[method] = []
            
            # Performance score combines pairs found and confidence
            score = entry["pairs_extracted"] * entry["confidence"]
            method_performance[method].append(score)
        
        # Calculate average performance per method
        avg_performance = {}
        for method, scores in method_performance.items():
            avg_performance[method] = np.mean(scores)
        
        # Recommend strategy based on performance
        if len(avg_performance) == 1:
            return self.strategy  # Only one method used
        
        best_method = max(avg_performance, key=avg_performance.get)
        
        if best_method == "adaptive":
            return ExtractionStrategy.ADAPTIVE_FIRST
        elif best_method == "llm":
            return ExtractionStrategy.LLM_FIRST
        else:
            return ExtractionStrategy.PARALLEL
    
    def set_strategy(self, strategy: ExtractionStrategy):
        """Update extraction strategy"""
        self.strategy = strategy
        logger.info(f"Hybrid extraction strategy updated to: {strategy.value}")

# Convenience function for quick testing
def test_hybrid_extraction(text_blocks: List[TextBlock], raw_text: str = None) -> Dict:
    """
    Quick test function for hybrid extraction
    """
    extractor = HybridKeyValueExtractor()
    
    if raw_text is None:
        raw_text = " ".join([block.text for block in text_blocks])
    
    result = extractor.extract_key_value_pairs(text_blocks, raw_text)
    
    return {
        "pairs_extracted": len(result.pairs),
        "primary_method": result.primary_method,
        "fallback_used": result.fallback_used,
        "processing_time": result.extraction_time_seconds,
        "confidence_scores": result.confidence_scores,
        "audit_trail": result.audit_trail,
        "performance_stats": extractor.get_performance_statistics()
    }