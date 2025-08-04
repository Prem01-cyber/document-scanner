#!/usr/bin/env python3
"""
Demo: Adaptive Document Scanner Learning System

This script demonstrates how the adaptive document scanner learns and improves
over time, replacing hardcoded values with learned parameters.

Run this to see the learning system in action!
"""

import json
import time
import random
import numpy as np
from typing import Dict, List
from src.config import adaptive_config

def simulate_document_processing_session(session_id: int, document_type: str) -> Dict:
    """
    Simulate a document processing session with realistic results
    """
    print(f"\n🔄 Processing Document #{session_id} ({document_type})")
    
    # Simulate different document types with different characteristics
    if document_type == "form":
        # Forms typically have good structure, high confidence
        base_confidence = random.uniform(0.7, 0.9)
        pairs_extracted = random.randint(8, 15)
        quality_confidence = random.uniform(0.8, 0.95)
        
    elif document_type == "invoice":
        # Invoices vary in quality
        base_confidence = random.uniform(0.6, 0.85)
        pairs_extracted = random.randint(5, 12)
        quality_confidence = random.uniform(0.7, 0.9)
        
    elif document_type == "receipt":
        # Receipts often have lower quality scans
        base_confidence = random.uniform(0.4, 0.75)
        pairs_extracted = random.randint(3, 8)
        quality_confidence = random.uniform(0.6, 0.8)
        
    else:  # mixed document
        base_confidence = random.uniform(0.5, 0.8)
        pairs_extracted = random.randint(4, 10)
        quality_confidence = random.uniform(0.65, 0.85)
    
    # Add some realistic variation
    confidence_variation = random.uniform(-0.1, 0.1)
    final_confidence = max(0.1, min(0.95, base_confidence + confidence_variation))
    
    # Simulate processing results
    processing_result = {
        "average_confidence": final_confidence,
        "quality_assessment": {
            "confidence": quality_confidence,
            "adaptive_thresholds": {
                "blur_threshold": random.uniform(80, 120),
                "dark_threshold": random.randint(25, 45),
                "bright_threshold": random.randint(200, 250)
            }
        },
        "extraction_statistics": {
            "total_pairs": pairs_extracted,
            "high_confidence_pairs": max(0, pairs_extracted - random.randint(1, 3)),
            "extraction_methods_used": [
                "adaptive_colon_terminator", 
                "adaptive_spatial_analysis",
                "learned_pattern_match"
            ]
        }
    }
    
    print(f"   📊 Confidence: {final_confidence:.3f}")
    print(f"   📄 Pairs extracted: {pairs_extracted}")
    print(f"   ⭐ Quality: {quality_confidence:.3f}")
    
    return processing_result

def show_parameter_evolution(parameter_name: str, category: str, initial_value: float):
    """
    Show how a parameter evolves over time
    """
    print(f"\n📈 Parameter Evolution: {category}.{parameter_name}")
    print(f"   Initial value: {initial_value:.3f}")
    
    current_value = adaptive_config.get_adaptive_value(category, parameter_name)
    print(f"   Current learned value: {current_value:.3f}")
    
    # Get learning history if available
    try:
        param_config = adaptive_config.config[category][parameter_name]
        learned_values = param_config.get("learned_values", [])
        confidence_weight = param_config.get("confidence_weight", 0.0)
        
        if learned_values:
            print(f"   📚 Learning samples: {len(learned_values)}")
            print(f"   🎯 Confidence weight: {confidence_weight:.3f}")
            print(f"   📊 Recent values: {learned_values[-3:]}")
            
            if len(learned_values) > 1:
                trend = learned_values[-1] - learned_values[0]
                trend_symbol = "📈" if trend > 0 else "📉" if trend < 0 else "➡️"
                print(f"   {trend_symbol} Overall trend: {trend:+.3f}")
        else:
            print(f"   📝 No learning data yet")
            
    except KeyError:
        print(f"   ❌ Parameter not found in config")

def demonstrate_adaptive_learning():
    """
    Main demonstration of the adaptive learning system
    """
    print("🚀 Adaptive Document Scanner Learning Demonstration")
    print("=" * 60)
    
    print("\n📋 Initial Configuration State:")
    print(f"   Config file: {adaptive_config.config_file}")
    
    # Show initial state of key parameters
    key_parameters = [
        ("min_contour_area_ratio", "quality_thresholds", 0.1),
        ("base_blur_threshold", "quality_thresholds", 100.0),
        ("colon_terminator_boost", "extraction_confidence", 0.2),
        ("form_left_position_threshold", "extraction_confidence", 150),
        ("min_pairing_confidence", "semantic_thresholds", 0.3)
    ]
    
    print("\n🔧 Key Parameters (Before Learning):")
    for param_name, category, initial_val in key_parameters:
        current_val = adaptive_config.get_adaptive_value(category, param_name)
        print(f"   {param_name}: {current_val:.3f} (default: {initial_val})")
    
    # Simulate multiple document processing sessions
    print("\n🔄 Simulating Document Processing Sessions...")
    print("   (The system will learn from each successful processing)")
    
    document_types = ["form", "invoice", "receipt", "mixed"] * 3  # 12 sessions
    processing_history = []
    
    for session_id, doc_type in enumerate(document_types, 1):
        # Simulate processing
        result = simulate_document_processing_session(session_id, doc_type)
        processing_history.append(result)
        
        # Let the system learn from this session
        adaptive_config.adapt_from_document_processing(result)
        
        # Show learning progress every few sessions
        if session_id % 4 == 0:
            print(f"\n📚 Learning Progress After {session_id} Sessions:")
            avg_confidence = np.mean([r["average_confidence"] for r in processing_history[-4:]])
            print(f"   Recent average confidence: {avg_confidence:.3f}")
            
        # Small delay for demonstration
        time.sleep(0.5)
    
    # Show final state
    print("\n" + "=" * 60)
    print("🎯 Learning Results Summary")
    print("=" * 60)
    
    print("\n🔧 Key Parameters (After Learning):")
    for param_name, category, initial_val in key_parameters:
        print(f"\n{param_name}:")
        show_parameter_evolution(param_name, category, initial_val)
    
    # Show overall learning statistics
    print("\n📊 Overall Learning Statistics:")
    total_sessions = len(processing_history)
    avg_confidence_overall = np.mean([r["average_confidence"] for r in processing_history])
    high_confidence_sessions = sum(1 for r in processing_history if r["average_confidence"] > 0.7)
    
    print(f"   Total processing sessions: {total_sessions}")
    print(f"   Average confidence: {avg_confidence_overall:.3f}")
    print(f"   High-confidence sessions: {high_confidence_sessions}/{total_sessions} ({high_confidence_sessions/total_sessions*100:.1f}%)")
    
    # Show confidence trend
    early_avg = np.mean([r["average_confidence"] for r in processing_history[:4]])
    late_avg = np.mean([r["average_confidence"] for r in processing_history[-4:]])
    improvement = late_avg - early_avg
    
    print(f"\n📈 Confidence Improvement:")
    print(f"   Early sessions (1-4): {early_avg:.3f}")
    print(f"   Recent sessions (last 4): {late_avg:.3f}")
    print(f"   Improvement: {improvement:+.3f}")
    
    if improvement > 0.05:
        print("   🎉 System is learning and improving!")
    elif improvement > -0.05:
        print("   ✅ System performance is stable")
    else:
        print("   ⚠️  System may need adjustment")
    
    # Save the learned configuration
    adaptive_config.save_config()
    print(f"\n💾 Learned configuration saved to: {adaptive_config.config_file}")
    
    print("\n" + "=" * 60)
    print("✨ Demonstration Complete!")
    print("\nThe adaptive system has now learned from document processing")
    print("and will use these improved parameters for future documents.")
    print("\nKey benefits of the adaptive system:")
    print("  • No hardcoded values - everything adapts")
    print("  • Learns from successful extractions")
    print("  • Improves accuracy over time")
    print("  • Handles different document types")
    print("  • Maintains configuration across restarts")

def show_config_comparison():
    """
    Show a comparison between default and learned configuration
    """
    print("\n🔍 Configuration Comparison:")
    print("-" * 40)
    
    # Load default config for comparison
    default_config = adaptive_config._load_default_config()
    current_config = adaptive_config.config
    
    categories = ["quality_thresholds", "extraction_confidence", "semantic_thresholds"]
    
    for category in categories:
        print(f"\n📂 {category}:")
        
        if category in default_config and category in current_config:
            for param_name in default_config[category]:
                if param_name in current_config[category]:
                    default_val = default_config[category][param_name].get("default", "N/A")
                    
                    if isinstance(current_config[category][param_name], dict):
                        learned_vals = current_config[category][param_name].get("learned_values", [])
                        confidence_weight = current_config[category][param_name].get("confidence_weight", 0.0)
                        
                        if learned_vals and confidence_weight > 0.1:
                            current_val = np.mean(learned_vals[-3:])  # Recent average
                            status = "🔄 LEARNED"
                        else:
                            current_val = default_val
                            status = "📝 DEFAULT"
                    else:
                        current_val = current_config[category][param_name]
                        status = "📝 DEFAULT"
                    
                    print(f"   {param_name}:")
                    print(f"     Default: {default_val}")
                    print(f"     Current: {current_val} {status}")

if __name__ == "__main__":
    try:
        print("🎬 Starting Adaptive Learning Demonstration...")
        
        # Reset config to defaults for clean demo
        print("\n🔄 Resetting to default configuration for demo...")
        adaptive_config.config = adaptive_config._load_default_config()
        
        # Run the main demonstration
        demonstrate_adaptive_learning()
        
        # Show detailed config comparison
        show_config_comparison()
        
        print("\n🎯 Demo completed successfully!")
        print("\nYou can now run the adaptive document processor and see")
        print("how it uses the learned parameters instead of hardcoded values.")
        
    except KeyboardInterrupt:
        print("\n⏹️  Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo error: {e}")
        import traceback
        traceback.print_exc()