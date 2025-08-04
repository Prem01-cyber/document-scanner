#!/usr/bin/env python3
"""
Hybrid Quality Assessment System Demo

This demo shows the integration of rule-based risk scoring with ML classification
for document quality assessment, comparing their predictions and showing agreement.
"""

import numpy as np
import json
from typing import Dict, List
import logging
from quality.adaptive_quality_checker import AdaptiveDocumentQualityChecker
from quality.quality_data_collector import quality_data_collector
from quality.train_quality_classifier import train_quality_classifier

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_test_scenarios() -> List[Dict]:
    """Generate diverse test scenarios for quality assessment"""
    scenarios = [
        {
            "name": "Perfect Document",
            "description": "High quality scan with no issues",
            "metrics": {
                "blur_confidence": 0.95,
                "edge_cut_flags": 0,
                "text_density_violations": 0,
                "brightness_issue": False,
                "skew_angle": 1.5,
                "document_area_ratio": 0.92
            },
            "expected": "accept"
        },
        {
            "name": "Blurry Scan",
            "description": "Poor quality due to camera shake",
            "metrics": {
                "blur_confidence": 0.25,
                "edge_cut_flags": 0,
                "text_density_violations": 0,
                "brightness_issue": False,
                "skew_angle": 2.0,
                "document_area_ratio": 0.88
            },
            "expected": "reject"
        },
        {
            "name": "Cut-off Edges",
            "description": "Document edges are cut off in scan",
            "metrics": {
                "blur_confidence": 0.80,
                "edge_cut_flags": 3,
                "text_density_violations": 2,
                "brightness_issue": False,
                "skew_angle": 4.0,
                "document_area_ratio": 0.70
            },
            "expected": "reject"
        },
        {
            "name": "Overexposed",
            "description": "Too bright, hard to read",
            "metrics": {
                "blur_confidence": 0.75,
                "edge_cut_flags": 0,
                "text_density_violations": 0,
                "brightness_issue": True,
                "skew_angle": 3.0,
                "document_area_ratio": 0.85
            },
            "expected": "warn"
        },
        {
            "name": "Skewed Document",
            "description": "Document is rotated significantly",
            "metrics": {
                "blur_confidence": 0.85,
                "edge_cut_flags": 0,
                "text_density_violations": 0,
                "brightness_issue": False,
                "skew_angle": 18.0,
                "document_area_ratio": 0.90
            },
            "expected": "reject"
        },
        {
            "name": "Marginal Quality",
            "description": "Borderline case with multiple minor issues",
            "metrics": {
                "blur_confidence": 0.65,
                "edge_cut_flags": 1,
                "text_density_violations": 1,
                "brightness_issue": False,
                "skew_angle": 8.0,
                "document_area_ratio": 0.75
            },
            "expected": "warn"
        },
        {
            "name": "Small Document",
            "description": "Document too small in frame",
            "metrics": {
                "blur_confidence": 0.80,
                "edge_cut_flags": 0,
                "text_density_violations": 0,
                "brightness_issue": False,
                "skew_angle": 2.0,
                "document_area_ratio": 0.25
            },
            "expected": "reject"
        }
    ]
    
    return scenarios

def demo_hybrid_system():
    """Demonstrate the hybrid quality assessment system"""
    print("🤖 Hybrid Quality Assessment System Demo")
    print("=" * 60)
    print("Comparing Rule-based Risk Scoring vs ML Classification")
    print()
    
    # Initialize quality checker
    try:
        quality_checker = AdaptiveDocumentQualityChecker()
    except Exception as e:
        print(f"❌ Failed to initialize quality checker: {e}")
        return
    
    # Check if ML model is available
    ml_available = quality_checker.ml_model_data is not None
    print(f"📊 Rule-based scoring: ✅ Available")
    print(f"🧠 ML classification: {'✅ Available' if ml_available else '❌ Not trained yet'}")
    print()
    
    if not ml_available:
        print("ℹ️  Training ML model first...")
        try:
            trainer = QualityClassifierTrainer()
            df = trainer.generate_synthetic_training_data(500)
            results = trainer.train_models(df)
            trainer.save_best_model(results)
            print("✅ ML model trained and saved!")
            
            # Reinitialize quality checker to load the model
            quality_checker = AdaptiveDocumentQualityChecker()
            ml_available = quality_checker.ml_model_data is not None
        except Exception as e:
            print(f"❌ Failed to train ML model: {e}")
    
    # Run test scenarios
    scenarios = generate_test_scenarios()
    
    print("\\n🧪 Testing Scenarios")
    print("=" * 60)
    
    results_summary = {
        "total_tests": len(scenarios),
        "rule_correct": 0,
        "ml_correct": 0,
        "agreements": 0,
        "disagreements": []
    }
    
    for scenario in scenarios:
        print(f"\\n📋 {scenario['name']}")
        print(f"   {scenario['description']}")
        print(f"   Expected: {scenario['expected']}")
        print("-" * 40)
        
        # Get predictions from both methods
        comparison = quality_checker.compare_prediction_methods(
            scenario['metrics'], document_type="general"
        )
        
        rule_result = comparison['rule_based']
        ml_result = comparison['ml_based']
        agreement = comparison['comparison']
        
        # Display results
        print(f"   Rule-based: {rule_result['decision']:8} (risk: {rule_result['risk_score']:.3f})")
        
        if ml_available and ml_result['status'] == 'success':
            ml_decision = "reject" if ml_result['needs_rescan'] else ("warn" if ml_result['probability'] > 0.3 else "accept")
            print(f"   ML-based:   {ml_decision:8} (prob: {ml_result['probability']:.3f})")
            print(f"   Agreement:  {'✅ Yes' if agreement['agreement'] else '❌ No'}")
            
            # Track accuracy
            if rule_result['decision'] == scenario['expected']:
                results_summary['rule_correct'] += 1
            if ml_decision == scenario['expected']:
                results_summary['ml_correct'] += 1
            if agreement['agreement']:
                results_summary['agreements'] += 1
            else:
                results_summary['disagreements'].append({
                    'scenario': scenario['name'],
                    'rule': rule_result['decision'],
                    'ml': ml_decision,
                    'expected': scenario['expected']
                })
        else:
            print(f"   ML-based:   {'❌ Unavailable' if not ml_available else '❌ Error'}")
            if rule_result['decision'] == scenario['expected']:
                results_summary['rule_correct'] += 1
    
    # Summary statistics
    print("\\n\\n📊 Performance Summary")
    print("=" * 60)
    
    rule_accuracy = results_summary['rule_correct'] / results_summary['total_tests']
    print(f"Rule-based accuracy: {rule_accuracy:.1%} ({results_summary['rule_correct']}/{results_summary['total_tests']})")
    
    if ml_available:
        ml_accuracy = results_summary['ml_correct'] / results_summary['total_tests']
        agreement_rate = results_summary['agreements'] / results_summary['total_tests']
        
        print(f"ML accuracy:         {ml_accuracy:.1%} ({results_summary['ml_correct']}/{results_summary['total_tests']})")
        print(f"Agreement rate:      {agreement_rate:.1%} ({results_summary['agreements']}/{results_summary['total_tests']})")
        
        if results_summary['disagreements']:
            print(f"\\nDisagreements ({len(results_summary['disagreements'])}):")
            for disagreement in results_summary['disagreements']:
                print(f"  • {disagreement['scenario']}: Rule={disagreement['rule']}, ML={disagreement['ml']}, Expected={disagreement['expected']}")
    
    print()

def demo_feature_importance():
    """Show feature importance from ML model"""
    print("\\n🔍 Feature Importance Analysis")
    print("=" * 60)
    
    try:
        quality_checker = AdaptiveDocumentQualityChecker()
        
        if not quality_checker.ml_model_data:
            print("❌ ML model not available for feature analysis")
            return
        
        model_data = quality_checker.ml_model_data
        model_type = model_data.get('model_type', 'unknown')
        
        print(f"Model type: {model_type}")
        print(f"Performance: ROC AUC = {model_data.get('performance', {}).get('roc_auc', 'unknown'):.3f}")
        print()
        
        # Get feature importance
        model = model_data['model']
        feature_names = model_data['feature_columns']
        
        if hasattr(model, 'feature_importances_'):
            # Random Forest
            importances = model.feature_importances_
            print("Feature Importance (Random Forest):")
        elif hasattr(model, 'coef_'):
            # Logistic Regression
            importances = np.abs(model.coef_[0])
            print("Feature Importance (Logistic Regression - Absolute Coefficients):")
        else:
            print("❌ Cannot extract feature importance from this model type")
            return
        
        # Sort features by importance
        feature_importance = list(zip(feature_names, importances))
        feature_importance.sort(key=lambda x: x[1], reverse=True)
        
        for feature, importance in feature_importance:
            bar_length = int(importance * 50)  # Scale for visualization
            bar = "█" * bar_length
            print(f"  {feature:25} │{bar:<50}│ {importance:.3f}")
        
        print("\\n💡 Interpretation:")
        print("   Higher values indicate features that are more important")
        print("   for predicting whether a document needs rescanning.")
        
    except Exception as e:
        print(f"❌ Error analyzing feature importance: {e}")

def demo_data_collection():
    """Demonstrate data collection for training"""  
    print("\\n📊 Data Collection Demo")
    print("=" * 60)
    
    try:
        collector = QualityDataCollector("demo_quality_data.csv")
        
        # Simulate some quality assessments
        print("Simulating quality assessments...")
        
        # Generate some example data
        examples = [
            {
                "metrics": {"blur_confidence": 0.85, "edge_cut_flags": 0, "text_density_violations": 0, 
                           "brightness_issue": False, "skew_angle": 2.0, "document_area_ratio": 0.90},
                "rule_result": {"risk_decision": "accept", "quality_risk_score": 0.15},
                "actual": False,
                "feedback": "Good quality"
            },
            {
                "metrics": {"blur_confidence": 0.30, "edge_cut_flags": 2, "text_density_violations": 1,
                           "brightness_issue": True, "skew_angle": 15.0, "document_area_ratio": 0.60},
                "rule_result": {"risk_decision": "reject", "quality_risk_score": 0.85},
                "actual": True,
                "feedback": "Multiple issues detected"
            }
        ]
        
        for i, example in enumerate(examples):
            session_id = f"demo_session_{i+1}"
            
            collector.log_quality_assessment(
                quality_metrics=example["metrics"],
                rule_based_result=example["rule_result"],
                document_type="demo",
                user_feedback=example["feedback"],
                actual_needs_rescan=example["actual"],
                session_id=session_id
            )
        
        # Show summary
        summary = collector.get_training_data_summary()
        print("\\nTraining Data Summary:")
        for key, value in summary.items():
            if key != "data_file":
                print(f"  {key}: {value}")
        
        print(f"\\n✅ Demo data saved to: {collector.data_file}")
        
    except Exception as e:
        print(f"❌ Data collection demo failed: {e}")

def main():
    """Run the complete hybrid system demo"""
    demo_hybrid_system()
    demo_feature_importance()
    demo_data_collection()
    
    print("\\n\\n🎯 Key Benefits of Hybrid System:")
    print("=" * 60)
    print("✅ Rule-based: Transparent, interpretable, no training needed")
    print("✅ ML-based: Data-driven, learns from feedback, handles complex patterns")
    print("✅ Hybrid: Best of both worlds, fallback redundancy")
    print("✅ Data collection: Continuous improvement through real usage")
    print("✅ Comparison: Identifies when methods disagree for human review")
    
    print("\\n🚀 Next Steps:")
    print("1. Deploy system with data collection enabled")
    print("2. Collect real user feedback on quality assessments") 
    print("3. Retrain ML model with labeled data using train_quality_classifier.py")
    print("4. Monitor rule vs ML agreement rates")
    print("5. Optimize thresholds based on performance metrics")

if __name__ == "__main__":
    main()