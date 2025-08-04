# 🧠 Hybrid Quality Assessment System

## Overview

The Enhanced Document Scanner now features a **hybrid quality assessment system** that combines rule-based computer vision analysis with machine learning classification to provide superior document quality evaluation.

## 🎯 Key Features

### **Rule-Based Risk Scoring**
- **Blur Detection**: Laplacian variance analysis
- **Edge Cut Detection**: Margin-based boundary analysis  
- **Text Density Analysis**: OCR-based text positioning
- **Brightness Assessment**: Histogram analysis
- **Skew Detection**: Hough line analysis
- **Document Size Validation**: Contour area analysis

### **ML Classification**
- **Logistic Regression** and **Random Forest** models
- **Automatic feature extraction** from quality metrics
- **Probabilistic predictions** with confidence scores
- **Continuous learning** from user feedback
- **Agreement tracking** with rule-based system

### **Hybrid Decision Making**
- **Dual assessment**: Both rule-based and ML predictions
- **Agreement monitoring**: Track when methods disagree
- **Intelligent fallback**: System works even if ML unavailable
- **Risk categorization**: Accept/Warn/Reject decisions
- **User guidance**: Actionable recommendations

## 🚀 Quick Start

### 1. Launch Enhanced UI
```bash
python enhanced_gradio_app.py
```
Open http://localhost:7861 in your browser.

### 2. Train ML Model (Optional)
```bash
python train_quality_classifier.py
```
This creates an initial ML model with synthetic data.

### 3. Collect Real Data
- Process documents through the UI
- Provide quality feedback in the "Quality Assessment" tab
- System automatically collects training data

### 4. Retrain with Real Data
Once you have 100+ labeled samples:
```bash
python train_quality_classifier.py
```
The model will automatically use your collected data.

## 📊 Enhanced UI Features

### **Quality Assessment Tab**
- **Detailed quality report** with metrics breakdown
- **Rule-based vs ML comparison** side-by-side
- **Risk factor analysis** with explanations
- **Cut-off detection results** with visual indicators
- **User feedback collection** for ML training

### **Quality Feedback System**
- Rate document quality (Good/Poor)
- Provide additional comments
- Automatic session tracking
- Contribution to ML model improvement

### **Training Data Tab**
- View collection progress
- Monitor rule-ML agreement rates
- Track labeling statistics
- Training readiness indicators

## 🔧 Technical Details

### **Quality Metrics**
The system extracts these 6 key features:
- `blur_confidence` (0.0-1.0): Normalized Laplacian variance
- `edge_cut_flags` (0-4): Number of document edges cut off
- `text_density_violations` (0-4): Edges with crowded text
- `brightness_issue` (0/1): Over/under-exposed flag
- `skew_angle` (0.0+): Absolute rotation angle
- `document_area_ratio` (0.0-1.0): Document coverage ratio

### **Decision Thresholds**
- **Accept**: Risk score < 0.4 (good quality)
- **Warn**: Risk score 0.4-0.65 (marginal quality)  
- **Reject**: Risk score ≥ 0.65 (poor quality)

Document-type specific adjustments:
- **Certificate/Legal**: More strict (0.55/0.35 thresholds)
- **Receipt/Note**: More lenient (0.75/0.5 thresholds)

### **ML Model Performance**
With sufficient training data, expect:
- **Rule-based accuracy**: ~85-90%
- **ML accuracy**: ~80-95%
- **Agreement rate**: ~75-85%

## 📈 Data Collection & Training

### **Automatic Data Collection**
Every document processing session logs:
- Quality metrics extracted from image
- Rule-based risk assessment results
- ML predictions (if model available)
- Processing metadata and timestamps

### **User Feedback Integration**
- Session-based feedback tracking
- Ground truth label collection
- Incremental model improvement
- Privacy-focused local storage

### **Training Pipeline**
1. **Data Collection**: Gradio UI automatically logs assessments
2. **Labeling**: Users provide feedback on quality decisions
3. **Export**: Training data exported to CSV format
4. **Training**: ML models trained on labeled data
5. **Deployment**: Best model automatically selected and saved
6. **Monitoring**: Performance tracked via agreement rates

## 🎛️ Configuration Options

### **Adaptive Thresholds**
Located in `adaptive_config`:
- `quality_risk_weights`: Per-document-type ML weights
- `cut_edge_margin_pct`: Document-type specific margins
- `quality_thresholds`: Various rule-based parameters

### **Model Selection**
The training pipeline automatically selects between:
- **Logistic Regression**: Fast, interpretable, good for linear relationships
- **Random Forest**: Handles non-linear patterns, feature interactions

Selection based on cross-validated ROC AUC scores.

## 🔍 Monitoring & Debugging

### **Quality Assessment Output**
```json
{
  "quality_risk_assessment": {
    "quality_risk_score": 0.32,
    "risk_decision": "accept",
    "risk_reasons": []
  },
  "ml_assessment": {
    "ml_available": true,
    "ml_prediction": 0,
    "ml_rescan_probability": 0.285,
    "rule_vs_ml_agreement": true
  },
  "cut_off_analysis": {
    "edge_cut_issues": ["left_edge_cut"],
    "text_density_violations": [],
    "density_ratios": {"left": 0.18, "right": 0.04}
  }
}
```

### **System Status Monitoring**
- ML model availability and performance
- Rule-ML agreement rates over time
- Training data collection progress
- Processing statistics and error rates

## 🛠️ Troubleshooting

### **ML Model Not Available**
- Run `python train_quality_classifier.py` to create initial model
- Check for `quality_rescan_model.pkl` in project directory
- Verify scikit-learn installation: `pip install scikit-learn`

### **Poor Agreement Between Rule-Based and ML**
- Collect more diverse training data
- Provide consistent feedback on quality assessments
- Consider document-type specific model training
- Review feature importance in training output

### **Quality Assessment Errors**
- Check image format and size requirements
- Verify OpenCV installation for computer vision features
- Review logs for specific error messages
- Ensure sufficient system memory for image processing

## 📚 Additional Resources

- **Demo Scripts**: `demo_hybrid_quality_system.py` for testing
- **Training Pipeline**: `train_quality_classifier.py` for ML model training
- **Data Collection**: `quality_data_collector.py` for training data management
- **Risk Scoring Demo**: `demo_risk_scoring.py` for understanding risk calculation

## 🎯 Best Practices

1. **Provide Consistent Feedback**: Help the ML model learn by giving accurate quality assessments
2. **Collect Diverse Data**: Process various document types to improve generalization
3. **Monitor Agreement**: Keep an eye on rule-ML agreement rates in the Training Data tab
4. **Retrain Periodically**: Update the ML model as you collect more labeled data
5. **Use Document Types**: Specify accurate document types for better threshold adaptation

The hybrid quality assessment system provides a robust, adaptive solution that improves over time while maintaining the reliability of rule-based computer vision analysis.