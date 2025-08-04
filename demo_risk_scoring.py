#!/usr/bin/env python3
"""
Demo script to test the new risk-based quality scoring system
"""

import json
from adaptive_quality_checker import AdaptiveDocumentQualityChecker

def demo_risk_scoring_system():
    """Demonstrate the risk scoring functionality"""
    print("🎯 Risk-Based Quality Scoring Demo")
    print("=" * 50)
    
    # Initialize the quality checker
    quality_checker = AdaptiveDocumentQualityChecker()
    
    # Test different document types
    document_types = ["general", "form", "certificate", "receipt", "legal"]
    
    for doc_type in document_types:
        print(f"\n📄 Document Type: {doc_type.upper()}")
        print("-" * 30)
        
        # Get demo results for this document type
        results = quality_checker.demo_risk_scoring(document_type=doc_type)
        
        for scenario, result in results.items():
            print(f"  {scenario.capitalize():10} | "
                  f"Risk: {result['risk_score']:5.3f} | "
                  f"Decision: {result['decision']:6} | "
                  f"Confidence: {result['confidence']:5.3f}")
            if result['reasons']:
                print(f"             Reasons: {', '.join(result['reasons'])}")
        
    # Test specific scenarios with custom metrics
    print(f"\n\n🧪 Custom Test Scenarios")
    print("=" * 50)
    
    # Test case 1: Moderate blur but good edges
    custom_metrics_1 = {
        "blur_confidence": 0.5,  # Moderate blur
        "edge_cut_flags": 0,     # No edge cuts
        "text_density_violations": 0,
        "brightness_issue": False,
        "skew_angle": 3.0
    }
    
    risk_score, reasons, decision = quality_checker.compute_quality_risk_score(
        custom_metrics_1, document_type="general"
    )
    
    print(f"Test 1 - Moderate Blur:")
    print(f"  Risk Score: {risk_score:.3f}")
    print(f"  Decision: {decision}")
    print(f"  Reasons: {reasons}")
    print(f"  Confidence: {1.0 - risk_score:.3f}")
    
    # Test case 2: Good blur but multiple edge issues
    custom_metrics_2 = {
        "blur_confidence": 0.85,  # Good blur
        "edge_cut_flags": 3,      # Multiple edge cuts
        "text_density_violations": 2,  # Text density issues
        "brightness_issue": False,
        "skew_angle": 2.0
    }
    
    risk_score, reasons, decision = quality_checker.compute_quality_risk_score(
        custom_metrics_2, document_type="general"
    )
    
    print(f"\nTest 2 - Good Blur, Edge Issues:")
    print(f"  Risk Score: {risk_score:.3f}")
    print(f"  Decision: {decision}")
    print(f"  Reasons: {reasons}")
    print(f"  Confidence: {1.0 - risk_score:.3f}")
    
    # Test case 3: Perfect quality
    custom_metrics_3 = {
        "blur_confidence": 0.95,
        "edge_cut_flags": 0,
        "text_density_violations": 0,
        "brightness_issue": False,
        "skew_angle": 1.0
    }
    
    risk_score, reasons, decision = quality_checker.compute_quality_risk_score(
        custom_metrics_3, document_type="certificate"  # Higher standards
    )
    
    print(f"\nTest 3 - Perfect Quality (Certificate):")
    print(f"  Risk Score: {risk_score:.3f}")
    print(f"  Decision: {decision}")
    print(f"  Reasons: {reasons}")
    print(f"  Confidence: {1.0 - risk_score:.3f}")
    
    print(f"\n✅ Risk Scoring Demo Complete!")
    print(f"\n📊 Key Features Demonstrated:")
    print(f"  • Document-type specific thresholds")
    print(f"  • Weighted scoring across multiple quality dimensions")
    print(f"  • Clear risk-based decisions (accept/warn/reject)")
    print(f"  • Human-readable explanations")
    print(f"  • Adaptive weight learning capability")

if __name__ == "__main__":
    demo_risk_scoring_system()