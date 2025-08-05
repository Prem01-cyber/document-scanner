# Dynamic Configuration System for Document Scanner
# This replaces hardcoded values with adaptive, learnable parameters

import json
import os
from typing import Dict, Any
import numpy as np
import logging

logger = logging.getLogger(__name__)

class AdaptiveConfig:
    """
    Adaptive configuration system that learns from document processing history
    """
    
    def __init__(self, config_file: str = "scanner_config.json"):
        self.config_file = config_file
        self.config = self._load_default_config()
        self.processing_history = []
        self.confidence_learned_values = {}
        
        # Load existing config if available
        self._load_config()
    
    def _load_default_config(self) -> Dict[str, Any]:
        """Load default adaptive configuration"""
        return {
            # Quality Assessment - Dynamic Thresholds
            "quality_thresholds": {
                "min_contour_area_ratio": {
                    "default": 0.1,
                    "adaptive_range": [0.05, 0.3],
                    "learned_values": [],
                    "confidence_weight": 0.0
                },
                "base_blur_threshold": {
                    "default": 100.0,
                    "adaptive_range": [50.0, 200.0],
                    "learned_values": [],
                    "confidence_weight": 0.0
                },
                "dark_threshold_bounds": {
                    "default": [30, 80],
                    "adaptive_range": [15, 120],
                    "learned_values": [],
                    "confidence_weight": 0.0
                },
                "bright_threshold_bounds": {
                    "default": [180, 240],
                    "adaptive_range": [150, 255],
                    "learned_values": [],
                    "confidence_weight": 0.0
                },
                "skew_tolerance": {
                    "default": 8.0,
                    "adaptive_range": [3.0, 15.0],
                    "learned_values": [],
                    "confidence_weight": 0.0
                },
                "text_area_ratio_threshold": {
                    "default": 0.01,
                    "adaptive_range": [0.005, 0.05],
                    "learned_values": [],
                    "confidence_weight": 0.0
                }
            },
            
            # Key-Value Extraction - Dynamic Confidence Boosts
            "extraction_confidence": {
                "colon_terminator_boost": {
                    "default": 0.2,
                    "adaptive_range": [0.1, 0.5],
                    "learned_values": [],
                    "confidence_weight": 0.0
                },
                "suffix_indicator_boost": {
                    "default": 0.4,
                    "adaptive_range": [0.2, 0.7],
                    "learned_values": [],
                    "confidence_weight": 0.0
                },
                "form_left_position_threshold": {
                    "default": 150,
                    "adaptive_range": [50, 300],
                    "learned_values": [],
                    "confidence_weight": 0.0
                },
                "proximity_distance_threshold": {
                    "default": 300,
                    "adaptive_range": [150, 600],
                    "learned_values": [],
                    "confidence_weight": 0.0
                }
            },
            
            # Semantic Analysis - Dynamic Thresholds
            "semantic_thresholds": {
                "min_pairing_confidence": {
                    "default": 0.3,
                    "adaptive_range": [0.15, 0.6],
                    "learned_values": [],
                    "confidence_weight": 0.0
                },
                "high_confidence_split_key": {
                    "default": 0.9,
                    "adaptive_range": [0.7, 0.95],
                    "learned_values": [],
                    "confidence_weight": 0.0
                },
                "semantic_compatibility_threshold": {
                    "default": 0.4,
                    "adaptive_range": [0.2, 0.7],
                    "learned_values": [],
                    "confidence_weight": 0.0
                }
            },
            
            # Learning Parameters
            "learning_config": {
                "learning_rate": 0.1,
                "confidence_decay": 0.95,
                "min_samples_for_adaptation": 5,
                "max_history_size": 100
            }
        }
    
    def get_adaptive_value(self, category: str, parameter: str) -> float:
        """
        Get adaptive parameter value based on learning history
        """
        try:
            param_config = self.config[category][parameter]
            default_value = param_config["default"]
            learned_values = param_config["learned_values"]
            confidence_weight = param_config["confidence_weight"]
            
            if not learned_values or confidence_weight < 0.3:
                return default_value
            
            # Calculate weighted average of learned values
            recent_values = learned_values[-10:]  # Use recent values
            
            # Ensure recent_values is a flat numeric array
            try:
                # Convert to numpy array and flatten if needed
                values_array = np.array(recent_values, dtype=float).flatten()
                weights_array = np.linspace(0.5, 1.0, len(values_array))
                
                # Ensure shapes match
                if len(values_array) == len(weights_array) and len(values_array) > 0:
                    weighted_avg = np.average(values_array, weights=weights_array)
                else:
                    # Fallback to simple average if shapes don't match
                    weighted_avg = np.mean(values_array) if len(values_array) > 0 else default_value
            except (ValueError, TypeError) as e:
                logger.warning(f"Error in weighted average calculation: {e}. Using simple mean.")
                weighted_avg = np.mean([float(v) for v in recent_values if isinstance(v, (int, float))])
                if np.isnan(weighted_avg):
                    weighted_avg = default_value
            
            # Blend with default based on confidence
            adaptive_value = (
                default_value * (1 - confidence_weight) + 
                weighted_avg * confidence_weight
            )
            
            # Ensure within adaptive range
            min_val, max_val = param_config["adaptive_range"]
            return np.clip(adaptive_value, min_val, max_val)
            
        except KeyError:
            return self.config.get(category, {}).get(parameter, {}).get("default", 0.5)
    
    def learn_from_success(self, category: str, parameter: str, value: float, success_confidence: float):
        """
        Learn from successful document processing
        """
        try:
            param_config = self.config[category][parameter]
            
            # Add to learned values
            param_config["learned_values"].append(value)
            
            # Update confidence weight based on success
            learning_rate = self.config["learning_config"]["learning_rate"]
            param_config["confidence_weight"] = min(
                1.0,
                param_config["confidence_weight"] + learning_rate * success_confidence
            )
            
            # Limit history size
            max_history = self.config["learning_config"]["max_history_size"]
            if len(param_config["learned_values"]) > max_history:
                param_config["learned_values"] = param_config["learned_values"][-max_history:]
            
        except KeyError:
            pass  # Parameter not found, skip learning
    
    def adapt_from_document_processing(self, processing_result: Dict[str, Any]):
        """
        Adapt configuration based on document processing results
        """
        try:
            # Extract success metrics
            success_confidence = processing_result.get("average_confidence", 0.0)
            quality_confidence = processing_result.get("quality_assessment", {}).get("confidence", 0.0)
            
            # Only learn from successful processing
            if success_confidence > 0.6 and quality_confidence > 0.7:
                # Adapt quality thresholds if they worked well
                quality_stats = processing_result.get("quality_assessment", {})
                
                if "adaptive_thresholds" in quality_stats:
                    thresholds = quality_stats["adaptive_thresholds"]
                    
                    # Learn blur threshold
                    if "blur_threshold" in thresholds:
                        self.learn_from_success(
                            "quality_thresholds", 
                            "base_blur_threshold",
                            thresholds["blur_threshold"],
                            quality_confidence
                        )
                    
                    # Learn brightness thresholds
                    if "dark_threshold" in thresholds and "bright_threshold" in thresholds:
                        self.learn_from_success(
                            "quality_thresholds",
                            "dark_threshold_bounds",
                            [thresholds["dark_threshold"], thresholds["bright_threshold"]],
                            quality_confidence
                        )
                
                # Learn extraction confidence values
                extraction_stats = processing_result.get("extraction_statistics", {})
                methods_used = extraction_stats.get("extraction_methods_used", [])
                
                # Adapt confidence boosts based on successful methods
                for method in methods_used:
                    if "colon_terminator" in method:
                        current_boost = self.get_adaptive_value("extraction_confidence", "colon_terminator_boost")
                        self.learn_from_success(
                            "extraction_confidence",
                            "colon_terminator_boost", 
                            current_boost,
                            success_confidence
                        )
        
        except Exception as e:
            print(f"Adaptation error: {e}")
    
    def save_config(self):
        """Save current configuration to file"""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=2, default=str)
        except Exception as e:
            print(f"Config save error: {e}")
    
    def _load_config(self):
        """Load configuration from file if it exists"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    saved_config = json.load(f)
                    
                # Merge saved config with defaults
                self._deep_merge(self.config, saved_config)
        except Exception as e:
            print(f"Config load error: {e}")
    
    def _deep_merge(self, default_dict: Dict, update_dict: Dict):
        """Recursively merge dictionaries"""
        for key, value in update_dict.items():
            if key in default_dict and isinstance(default_dict[key], dict) and isinstance(value, dict):
                self._deep_merge(default_dict[key], value)
            else:
                default_dict[key] = value

# Global adaptive configuration instance
adaptive_config = AdaptiveConfig()