#!/usr/bin/env python3
"""
Quality Data Collector for ML Training

This module provides utilities to collect and manage training data
for the ML quality classifier from real document processing sessions.
"""

import csv
import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Any
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class QualityDataCollector:
    """Collect and manage quality assessment data for ML training"""
    
    def __init__(self, data_file: str = "quality_training_data.csv"):
        self.data_file = data_file
        self.fieldnames = [
            "timestamp",
            "document_type", 
            "blur_confidence",
            "edge_cut_flags",
            "text_density_violations", 
            "brightness_issue",
            "skew_angle",
            "document_area_ratio",
            "rule_based_decision",
            "rule_based_score",
            "ml_prediction",
            "ml_probability",
            "user_feedback",
            "actual_needs_rescan",
            "session_id"
        ]
        self._ensure_data_file_exists()
    
    def _ensure_data_file_exists(self):
        """Create CSV file with headers if it doesn't exist"""
        if not os.path.exists(self.data_file):
            with open(self.data_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writeheader()
            logger.info(f"Created new training data file: {self.data_file}")
    
    def log_quality_assessment(self, 
                             quality_metrics: Dict,
                             rule_based_result: Dict,
                             ml_result: Dict = None,
                             document_type: str = "general",
                             user_feedback: str = None,
                             actual_needs_rescan: bool = None,
                             session_id: str = None) -> None:
        """Log a quality assessment for training data
        
        Args:
            quality_metrics: Raw quality metrics from quality checker
            rule_based_result: Rule-based assessment results
            ml_result: ML assessment results (optional)
            document_type: Type of document processed
            user_feedback: User feedback about quality
            actual_needs_rescan: Ground truth label
            session_id: Unique session identifier
        """
        try:
            # Prepare data row
            row_data = {
                "timestamp": datetime.now().isoformat(),
                "document_type": document_type,
                "blur_confidence": quality_metrics.get("blur_confidence", 0.0),
                "edge_cut_flags": quality_metrics.get("edge_cut_flags", 0),
                "text_density_violations": quality_metrics.get("text_density_violations", 0),
                "brightness_issue": int(quality_metrics.get("brightness_issue", False)),
                "skew_angle": quality_metrics.get("skew_angle", 0.0),
                "document_area_ratio": quality_metrics.get("document_area_ratio", 0.85),
                "rule_based_decision": rule_based_result.get("risk_decision", "unknown"),
                "rule_based_score": rule_based_result.get("quality_risk_score", 0.0),
                "ml_prediction": ml_result.get("ml_prediction") if ml_result else None,
                "ml_probability": ml_result.get("ml_rescan_probability") if ml_result else None,
                "user_feedback": user_feedback,
                "actual_needs_rescan": int(actual_needs_rescan) if actual_needs_rescan is not None else None,
                "session_id": session_id or f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            }
            
            # Append to CSV
            with open(self.data_file, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writerow(row_data)
            
            logger.debug(f"Logged quality assessment: {session_id}")
            
        except Exception as e:
            logger.error(f"Failed to log quality assessment: {e}")
    
    def update_ground_truth(self, session_id: str, actual_needs_rescan: bool, user_feedback: str = None) -> None:
        """Update ground truth label for a specific session
        
        Args:
            session_id: Session to update
            actual_needs_rescan: True if document actually needed rescan
            user_feedback: Additional user feedback
        """
        try:
            # Read existing data
            rows = []
            with open(self.data_file, 'r') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            
            # Update matching session
            updated = False
            for row in rows:
                if row['session_id'] == session_id:
                    row['actual_needs_rescan'] = int(actual_needs_rescan)
                    if user_feedback:
                        row['user_feedback'] = user_feedback
                    updated = True
                    break
            
            if updated:
                # Write back to file
                with open(self.data_file, 'w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                    writer.writeheader()
                    writer.writerows(rows)
                logger.info(f"Updated ground truth for session {session_id}")
            else:
                logger.warning(f"Session {session_id} not found for ground truth update")
                
        except Exception as e:
            logger.error(f"Failed to update ground truth: {e}")
    
    def get_training_data_summary(self) -> Dict:
        """Get summary statistics of collected training data"""
        try:
            rows = []
            with open(self.data_file, 'r') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            
            if not rows:
                return {"total_samples": 0}
            
            # Calculate statistics
            total_samples = len(rows)
            labeled_samples = sum(1 for row in rows if row['actual_needs_rescan'] not in ['', None])
            
            # Document type distribution
            doc_types = {}
            for row in rows:
                doc_type = row['document_type']
                doc_types[doc_type] = doc_types.get(doc_type, 0) + 1
            
            # Rule vs ML agreement (where both available)
            agreements = 0
            comparisons = 0
            for row in rows:
                if row['rule_based_decision'] and row['ml_prediction']:
                    comparisons += 1
                    rule_needs_rescan = row['rule_based_decision'] == 'reject'
                    ml_needs_rescan = int(row['ml_prediction']) == 1
                    if rule_needs_rescan == ml_needs_rescan:
                        agreements += 1
                        
            agreement_rate = agreements / comparisons if comparisons > 0 else None
            
            # Quality metrics distributions
            blur_scores = [float(row['blur_confidence']) for row in rows if row['blur_confidence']]
            edge_cuts = [int(row['edge_cut_flags']) for row in rows if row['edge_cut_flags']]
            
            return {
                "total_samples": total_samples,
                "labeled_samples": labeled_samples,
                "labeling_progress": f"{labeled_samples}/{total_samples} ({labeled_samples/total_samples*100:.1f}%)" if total_samples > 0 else "0%",
                "document_types": doc_types,
                "rule_ml_agreement_rate": agreement_rate,
                "rule_ml_comparisons": comparisons,
                "avg_blur_confidence": sum(blur_scores) / len(blur_scores) if blur_scores else 0,
                "avg_edge_cuts": sum(edge_cuts) / len(edge_cuts) if edge_cuts else 0,
                "data_file": self.data_file
            }
            
        except Exception as e:
            logger.error(f"Failed to get training data summary: {e}")
            return {"error": str(e)}
    
    def export_for_training(self, output_file: str = None, min_samples: int = 100) -> str:
        """Export labeled data for ML training
        
        Args:
            output_file: Output filename (default: quality_labeled_data.csv)
            min_samples: Minimum samples needed for export
            
        Returns:
            Path to exported file or error message
        """
        output_file = output_file or "quality_labeled_data.csv"
        
        try:
            # Read and filter labeled data
            labeled_rows = []
            with open(self.data_file, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row['actual_needs_rescan'] not in ['', None]:
                        labeled_rows.append(row)
            
            if len(labeled_rows) < min_samples:
                return f"Insufficient labeled data: {len(labeled_rows)} < {min_samples} required"
            
            # Export for training (simplified format)
            training_fieldnames = [
                "blur_confidence", "edge_cut_flags", "text_density_violations",
                "brightness_issue", "skew_angle", "document_area_ratio", 
                "needs_rescan"
            ]
            
            with open(output_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=training_fieldnames)
                writer.writeheader()
                
                for row in labeled_rows:
                    training_row = {
                        "blur_confidence": float(row['blur_confidence']),
                        "edge_cut_flags": int(row['edge_cut_flags']),
                        "text_density_violations": int(row['text_density_violations']),
                        "brightness_issue": int(row['brightness_issue']),
                        "skew_angle": float(row['skew_angle']),
                        "document_area_ratio": float(row['document_area_ratio']),
                        "needs_rescan": int(row['actual_needs_rescan'])
                    }
                    writer.writerow(training_row)
            
            logger.info(f"Exported {len(labeled_rows)} labeled samples to {output_file}")
            return output_file
            
        except Exception as e:
            logger.error(f"Failed to export training data: {e}")
            return f"Export failed: {e}"

# Global instance for easy use
quality_data_collector = QualityDataCollector()

def log_assessment(quality_metrics: Dict, rule_result: Dict, ml_result: Dict = None, **kwargs):
    """Convenience function to log quality assessment"""
    quality_data_collector.log_quality_assessment(
        quality_metrics, rule_result, ml_result, **kwargs
    )

def update_label(session_id: str, needs_rescan: bool, feedback: str = None):
    """Convenience function to update ground truth labels"""
    quality_data_collector.update_ground_truth(session_id, needs_rescan, feedback)

def get_summary():
    """Convenience function to get data summary"""
    return quality_data_collector.get_training_data_summary()

if __name__ == "__main__":
    # Demo usage
    collector = QualityDataCollector()
    
    # Simulate some data collection
    sample_metrics = {
        "blur_confidence": 0.75,
        "edge_cut_flags": 1,
        "text_density_violations": 0,
        "brightness_issue": False,
        "skew_angle": 3.2,
        "document_area_ratio": 0.87
    }
    
    sample_rule_result = {
        "risk_decision": "accept",
        "quality_risk_score": 0.35
    }
    
    collector.log_quality_assessment(
        sample_metrics, 
        sample_rule_result,
        document_type="form",
        session_id="demo_session_1"
    )
    
    # Show summary
    summary = collector.get_training_data_summary()
    print("Training Data Summary:")
    print(json.dumps(summary, indent=2))