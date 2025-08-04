#!/usr/bin/env python3
"""
ML Classifier Training for Document Quality Assessment

This script trains a lightweight ML classifier to predict whether a document
needs rescanning based on quality features extracted from the quality checker.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple, Any
import logging
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QualityClassifierTrainer:
    """Train and evaluate ML classifiers for document quality assessment"""
    
    def __init__(self, data_path: str = "quality_labeled_data.csv"):
        self.data_path = data_path
        self.feature_columns = [
            "blur_confidence", 
            "edge_cut_flags", 
            "text_density_violations",
            "brightness_issue", 
            "skew_angle", 
            "document_area_ratio"
        ]
        self.scaler = StandardScaler()
        self.models = {}
        
    def generate_synthetic_training_data(self, n_samples: int = 1000) -> pd.DataFrame:
        """Generate synthetic training data for initial model development"""
        logger.info(f"Generating {n_samples} synthetic training samples")
        
        np.random.seed(42)
        data = []
        
        for _ in range(n_samples):
            # Generate realistic feature combinations
            blur_confidence = np.random.beta(2, 1)  # Skewed toward higher values
            edge_cut_flags = np.random.poisson(0.5)  # Most docs have 0-1 edge cuts
            edge_cut_flags = min(edge_cut_flags, 4)  # Cap at 4
            
            text_density_violations = np.random.poisson(0.3)
            text_density_violations = min(text_density_violations, 4)
            
            brightness_issue = np.random.choice([0, 1], p=[0.8, 0.2])  # 20% brightness issues
            skew_angle = np.abs(np.random.normal(0, 5))  # Most docs well-aligned
            document_area_ratio = np.random.beta(8, 2)  # Skewed toward higher ratios
            
            # Determine label based on realistic rules
            needs_rescan = 0
            
            # Strong indicators for rescan needed
            if blur_confidence < 0.4:  # Very blurry
                needs_rescan = 1
            elif edge_cut_flags >= 3:  # Multiple edges cut
                needs_rescan = 1
            elif text_density_violations >= 3:  # Text crowding issues
                needs_rescan = 1
            elif skew_angle > 20:  # Very skewed
                needs_rescan = 1
            elif document_area_ratio < 0.3:  # Document too small
                needs_rescan = 1
                
            # Moderate indicators (probabilistic)
            elif (blur_confidence < 0.6 and edge_cut_flags >= 1 and 
                  np.random.random() < 0.7):  # Combined moderate issues
                needs_rescan = 1
            elif (text_density_violations >= 2 and brightness_issue and 
                  np.random.random() < 0.6):
                needs_rescan = 1
            elif (skew_angle > 15 and blur_confidence < 0.7 and 
                  np.random.random() < 0.5):
                needs_rescan = 1
            
            data.append({
                "blur_confidence": blur_confidence,
                "edge_cut_flags": edge_cut_flags,
                "text_density_violations": text_density_violations,
                "brightness_issue": brightness_issue,
                "skew_angle": skew_angle,
                "document_area_ratio": document_area_ratio,
                "needs_rescan": needs_rescan
            })
        
        df = pd.DataFrame(data)
        
        # Add some noise to make it more realistic
        rescan_rate = df['needs_rescan'].mean()
        logger.info(f"Generated data with {rescan_rate:.1%} rescan rate")
        
        return df
    
    def load_or_generate_data(self) -> pd.DataFrame:
        """Load existing data or generate synthetic data"""
        if os.path.exists(self.data_path):
            logger.info(f"Loading existing data from {self.data_path}")
            df = pd.read_csv(self.data_path)
        else:
            logger.info(f"No existing data found. Generating synthetic data.")
            df = self.generate_synthetic_training_data()
            df.to_csv(self.data_path, index=False)
            logger.info(f"Saved synthetic data to {self.data_path}")
        
        return df
    
    def train_models(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Train multiple ML models and compare performance"""
        logger.info("Training ML models for quality classification")
        
        # Prepare features and labels
        X = df[self.feature_columns]
        y = df["needs_rescan"]
        
        logger.info(f"Training on {len(df)} samples with {y.mean():.1%} positive class rate")
        
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features for logistic regression
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        results = {}
        
        # 1. Logistic Regression
        logger.info("Training Logistic Regression...")
        lr_model = LogisticRegression(random_state=42, max_iter=1000)
        lr_model.fit(X_train_scaled, y_train)
        
        lr_pred = lr_model.predict(X_test_scaled)
        lr_prob = lr_model.predict_proba(X_test_scaled)[:, 1]
        
        results['logistic_regression'] = {
            'model': lr_model,
            'predictions': lr_pred,
            'probabilities': lr_prob,
            'test_score': lr_model.score(X_test_scaled, y_test),
            'roc_auc': roc_auc_score(y_test, lr_prob),
            'feature_importance': dict(zip(self.feature_columns, lr_model.coef_[0]))
        }
        
        # 2. Random Forest
        logger.info("Training Random Forest...")
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)
        
        rf_pred = rf_model.predict(X_test)
        rf_prob = rf_model.predict_proba(X_test)[:, 1]
        
        results['random_forest'] = {
            'model': rf_model,
            'predictions': rf_pred,
            'probabilities': rf_prob,
            'test_score': rf_model.score(X_test, y_test),
            'roc_auc': roc_auc_score(y_test, rf_prob),
            'feature_importance': dict(zip(self.feature_columns, rf_model.feature_importances_))
        }
        
        # Cross-validation scores
        for model_name in ['logistic_regression', 'random_forest']:
            model = results[model_name]['model']
            X_cv = X_train_scaled if model_name == 'logistic_regression' else X_train
            cv_scores = cross_val_score(model, X_cv, y_train, cv=5, scoring='roc_auc')
            results[model_name]['cv_roc_auc'] = cv_scores.mean()
            results[model_name]['cv_roc_auc_std'] = cv_scores.std()
        
        # Store test data for evaluation
        self.X_test = X_test
        self.y_test = y_test
        self.X_test_scaled = X_test_scaled
        
        return results
    
    def evaluate_models(self, results: Dict[str, Any]) -> None:
        """Print detailed evaluation of trained models"""
        logger.info("\\n" + "="*60)
        logger.info("MODEL EVALUATION RESULTS")
        logger.info("="*60)
        
        for model_name, result in results.items():
            logger.info(f"\\n{model_name.upper().replace('_', ' ')}")
            logger.info("-" * 40)
            logger.info(f"Test Accuracy: {result['test_score']:.3f}")
            logger.info(f"ROC AUC: {result['roc_auc']:.3f}")
            logger.info(f"CV ROC AUC: {result['cv_roc_auc']:.3f} (+/- {result['cv_roc_auc_std']*2:.3f})")
            
            # Classification report
            print(f"\\nClassification Report for {model_name}:")
            print(classification_report(self.y_test, result['predictions']))
            
            # Feature importance
            logger.info("\\nFeature Importance:")
            for feature, importance in sorted(result['feature_importance'].items(), 
                                            key=lambda x: abs(x[1]), reverse=True):
                logger.info(f"  {feature:25}: {importance:8.3f}")
    
    def save_best_model(self, results: Dict[str, Any], output_path: str = "quality_rescan_model.pkl") -> str:
        """Save the best performing model"""
        # Choose best model based on ROC AUC
        best_model_name = max(results.keys(), key=lambda k: results[k]['roc_auc'])
        best_model = results[best_model_name]['model']
        
        # Save model and scaler
        model_data = {
            'model': best_model,
            'scaler': self.scaler if best_model_name == 'logistic_regression' else None,
            'feature_columns': self.feature_columns,
            'model_type': best_model_name,
            'performance': {
                'test_accuracy': results[best_model_name]['test_score'],
                'roc_auc': results[best_model_name]['roc_auc'],
                'cv_roc_auc': results[best_model_name]['cv_roc_auc']
            }
        }
        
        joblib.dump(model_data, output_path)
        logger.info(f"\\nSaved best model ({best_model_name}) to {output_path}")
        logger.info(f"ROC AUC: {results[best_model_name]['roc_auc']:.3f}")
        
        return best_model_name
    
    def plot_feature_importance(self, results: Dict[str, Any], save_path: str = "feature_importance.png"):
        """Plot feature importance comparison"""
        try:
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))
            
            for idx, (model_name, result) in enumerate(results.items()):
                ax = axes[idx]
                
                features = list(result['feature_importance'].keys())
                importances = [abs(x) for x in result['feature_importance'].values()]
                
                # Sort by importance
                sorted_idx = sorted(range(len(importances)), key=lambda i: importances[i], reverse=True)
                features = [features[i] for i in sorted_idx]
                importances = [importances[i] for i in sorted_idx]
                
                bars = ax.barh(features, importances)
                ax.set_title(f"{model_name.replace('_', ' ').title()} Feature Importance")
                ax.set_xlabel("Absolute Importance")
                
                # Color bars
                colors = plt.cm.viridis(np.linspace(0, 1, len(bars)))
                for bar, color in zip(bars, colors):
                    bar.set_color(color)
            
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Feature importance plot saved to {save_path}")
            
        except ImportError:
            logger.warning("Matplotlib not available, skipping feature importance plot")

def main():
    """Main training pipeline"""
    logger.info("Starting Quality Classifier Training Pipeline")
    
    # Initialize trainer
    trainer = QualityClassifierTrainer()
    
    # Load or generate data
    df = trainer.load_or_generate_data()
    
    # Train models
    results = trainer.train_models(df)
    
    # Evaluate models
    trainer.evaluate_models(results)
    
    # Plot feature importance
    trainer.plot_feature_importance(results)
    
    # Save best model
    best_model = trainer.save_best_model(results)
    
    logger.info(f"\\n✅ Training completed successfully!")
    logger.info(f"Best model: {best_model}")
    logger.info(f"Model saved as: quality_rescan_model.pkl")
    logger.info(f"To use in production, load with: joblib.load('quality_rescan_model.pkl')")

if __name__ == "__main__":
    main()