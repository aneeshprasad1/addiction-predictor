#!/usr/bin/env python3
"""
Evaluation Script for EEG CNN Model

This script provides comprehensive evaluation of the trained model including:
- Cross-validation
- Statistical significance testing
- Detailed performance analysis
- Model interpretability
"""

import os
import sys
import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import logging
import wandb
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, roc_auc_score,
    confusion_matrix, classification_report, roc_curve, precision_recall_curve
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import pandas as pd

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from models.cnn_model import create_model, count_parameters


def load_config_with_proper_types(config_path: Path) -> Dict[str, Any]:
    """
    Load YAML config with proper type conversion for scientific notation and numbers.
    
    Args:
        config_path: Path to YAML config file
        
    Returns:
        Config dictionary with proper types
    """
    def convert_numeric(loader, node):
        """Convert numeric strings to appropriate types."""
        value = loader.construct_scalar(node)
        if isinstance(value, str):
            # Try to convert to int first, then float
            try:
                if '.' in value or 'e' in value.lower() or 'E' in value:
                    return float(value)
                else:
                    return int(value)
            except ValueError:
                return value
        return value
    
    # Create custom loader
    class ProperTypeLoader(yaml.SafeLoader):
        pass
    
    # Register constructors
    ProperTypeLoader.add_constructor('tag:yaml.org,2002:str', convert_numeric)
    
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=ProperTypeLoader)
    
    return config


class EEGEvaluator:
    """Comprehensive evaluation pipeline for EEG CNN model."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.evaluation_config = config['evaluation']
        self.training_config = config['training']
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Setup device
        if self.training_config['device'] == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(self.training_config['device'])
        
        # Initialize wandb
        self._setup_wandb()
        
    def _setup_wandb(self):
        """Initialize Weights & Biases logging."""
        wandb.init(
            project=self.config['wandb']['project'],
            entity=self.config['wandb']['entity'],
            tags=self.config['wandb']['tags'] + ['evaluation'],
            name="model_evaluation"
        )
    
    def load_data(self, data_path: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Load preprocessed data."""
        data = np.load(data_path)
        features = torch.FloatTensor(data['features'])
        labels = torch.LongTensor(data['labels'])
        
        self.logger.info(f"Loaded data: {features.shape}, labels: {labels.shape}")
        return features, labels
    
    def load_trained_model(self, model_path: str) -> nn.Module:
        """Load trained model from checkpoint."""
        model = create_model(self.config)
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.to(self.device)
        model.eval()
        
        self.logger.info(f"Loaded model from {model_path}")
        return model
    
    def cross_validation_evaluation(self, features: torch.Tensor, labels: torch.Tensor) -> Dict[str, List[float]]:
        """
        Perform k-fold cross-validation.
        
        Args:
            features: Feature tensor
            labels: Label tensor
            
        Returns:
            Dictionary of cross-validation results
        """
        n_folds = self.evaluation_config['cross_validation_folds']
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=self.training_config['random_seed'])
        
        cv_results = {
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1': [],
            'auc': []
        }
        
        self.logger.info(f"Starting {n_folds}-fold cross-validation...")
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(features, labels)):
            self.logger.info(f"Fold {fold + 1}/{n_folds}")
            
            # Split data
            train_features, val_features = features[train_idx], features[val_idx]
            train_labels, val_labels = labels[train_idx], labels[val_idx]
            
            # Create data loaders
            train_dataset = TensorDataset(train_features, train_labels)
            val_dataset = TensorDataset(val_features, val_labels)
            
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.training_config['batch_size'],
                shuffle=True
            )
            
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.training_config['batch_size'],
                shuffle=False
            )
            
            # Train model for this fold
            model = create_model(self.config)
            model.to(self.device)
            
            # Train the model (simplified training for CV)
            self._train_model_fold(model, train_loader, val_loader)
            
            # Evaluate
            fold_metrics = self._evaluate_model_fold(model, val_loader)
            
            # Store results
            for metric, value in fold_metrics.items():
                cv_results[metric].append(value)
            
            self.logger.info(f"Fold {fold + 1} - Accuracy: {fold_metrics['accuracy']:.4f}")
        
        # Calculate statistics
        cv_stats = {}
        for metric, values in cv_results.items():
            cv_stats[f'{metric}_mean'] = np.mean(values)
            cv_stats[f'{metric}_std'] = np.std(values)
            cv_stats[f'{metric}_values'] = values
        
        self.logger.info("Cross-validation results:")
        for metric in ['accuracy', 'precision', 'recall', 'f1', 'auc']:
            mean_val = cv_stats[f'{metric}_mean']
            std_val = cv_stats[f'{metric}_std']
            self.logger.info(f"  {metric.capitalize()}: {mean_val:.4f} ± {std_val:.4f}")
        
        return cv_stats
    
    def _train_model_fold(self, model: nn.Module, train_loader: DataLoader, val_loader: DataLoader):
        """Train model for a single fold (simplified version)."""
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.training_config['learning_rate'],
            weight_decay=self.training_config['weight_decay']
        )
        criterion = nn.CrossEntropyLoss()
        
        # Train for a few epochs
        for epoch in range(10):  # Reduced epochs for CV
            model.train()
            for batch_features, batch_labels in train_loader:
                batch_features = batch_features.to(self.device)
                batch_labels = batch_labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(batch_features)
                loss = criterion(outputs, batch_labels)
                loss.backward()
                optimizer.step()
    
    def _evaluate_model_fold(self, model: nn.Module, val_loader: DataLoader) -> Dict[str, float]:
        """Evaluate model for a single fold."""
        model.eval()
        all_predictions = []
        all_labels = []
        all_probabilities = []
        
        with torch.no_grad():
            for batch_features, batch_labels in val_loader:
                batch_features = batch_features.to(self.device)
                batch_labels = batch_labels.to(self.device)
                
                outputs = model(batch_features)
                probabilities = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs.data, 1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(batch_labels.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        return self._calculate_metrics(
            np.array(all_predictions),
            np.array(all_labels),
            np.array(all_probabilities)
        )
    
    def _calculate_metrics(self, predictions: np.ndarray, true_labels: np.ndarray, 
                          probabilities: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Calculate comprehensive evaluation metrics."""
        accuracy = accuracy_score(true_labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels, predictions, average='binary'
        )
        
        # Calculate ROC AUC
        auc = 0.0
        if probabilities is not None:
            auc = roc_auc_score(true_labels, probabilities[:, 1])
        else:
            auc = roc_auc_score(true_labels, predictions)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc
        }
    
    def statistical_significance_test(self, cv_results: Dict[str, List[float]]) -> Dict[str, float]:
        """
        Perform statistical significance tests.
        
        Args:
            cv_results: Cross-validation results
            
        Returns:
            Dictionary of p-values and confidence intervals
        """
        # For binary classification, we can test if accuracy is significantly better than chance (0.5)
        accuracies = cv_results['accuracy_values']
        
        # One-sample t-test against chance level
        t_stat, p_value = stats.ttest_1samp(accuracies, 0.5)
        
        # Calculate confidence interval
        ci_lower, ci_upper = stats.t.interval(
            confidence=0.95,
            df=len(accuracies) - 1,
            loc=np.mean(accuracies),
            scale=stats.sem(accuracies)
        )
        
        # Effect size (Cohen's d)
        effect_size = (np.mean(accuracies) - 0.5) / np.std(accuracies)
        
        results = {
            't_statistic': t_stat,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'confidence_interval_lower': ci_lower,
            'confidence_interval_upper': ci_upper,
            'effect_size': effect_size
        }
        
        self.logger.info("Statistical significance test results:")
        self.logger.info(f"  t-statistic: {t_stat:.4f}")
        self.logger.info(f"  p-value: {p_value:.4f}")
        self.logger.info(f"  Significant: {p_value < 0.05}")
        self.logger.info(f"  95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
        self.logger.info(f"  Effect size: {effect_size:.4f}")
        
        return results
    
    def detailed_performance_analysis(self, model: nn.Module, test_loader: DataLoader) -> Dict[str, Any]:
        """
        Perform detailed performance analysis.
        
        Args:
            model: Trained model
            test_loader: Test data loader
            
        Returns:
            Dictionary of detailed analysis results
        """
        model.eval()
        all_predictions = []
        all_labels = []
        all_probabilities = []
        
        with torch.no_grad():
            for batch_features, batch_labels in tqdm(test_loader, desc="Evaluating"):
                batch_features = batch_features.to(self.device)
                batch_labels = batch_labels.to(self.device)
                
                outputs = model(batch_features)
                probabilities = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs.data, 1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(batch_labels.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        predictions = np.array(all_predictions)
        true_labels = np.array(all_labels)
        probabilities = np.array(all_probabilities)
        
        # Calculate metrics
        metrics = self._calculate_metrics(predictions, true_labels, probabilities)
        
        # Detailed classification report
        class_report = classification_report(
            true_labels, predictions,
            target_names=['Control', 'Internet Addicted'],
            output_dict=True
        )
        
        # Confusion matrix
        cm = confusion_matrix(true_labels, predictions)
        
        # ROC curve
        fpr, tpr, roc_thresholds = roc_curve(true_labels, probabilities[:, 1])
        roc_auc = roc_auc_score(true_labels, probabilities[:, 1])
        
        # Precision-Recall curve
        precision, recall, pr_thresholds = precision_recall_curve(true_labels, probabilities[:, 1])
        
        # Create visualizations
        self._create_evaluation_plots(
            cm, fpr, tpr, precision, recall, roc_auc,
            predictions, true_labels, probabilities
        )
        
        results = {
            'metrics': metrics,
            'classification_report': class_report,
            'confusion_matrix': cm,
            'roc_curve': {'fpr': fpr, 'tpr': tpr, 'auc': roc_auc},
            'pr_curve': {'precision': precision, 'recall': recall},
            'predictions': predictions,
            'true_labels': true_labels,
            'probabilities': probabilities
        }
        
        return results
    
    def _create_evaluation_plots(self, cm: np.ndarray, fpr: np.ndarray, tpr: np.ndarray,
                                precision: np.ndarray, recall: np.ndarray, roc_auc: float,
                                predictions: np.ndarray, true_labels: np.ndarray,
                                probabilities: np.ndarray):
        """Create and log evaluation plots."""
        
        # Confusion Matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Control', 'Internet Addicted'],
                   yticklabels=['Control', 'Internet Addicted'])
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        wandb.log({"confusion_matrix_detailed": wandb.Image(plt)})
        plt.close()
        
        # ROC Curve
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        wandb.log({"roc_curve_detailed": wandb.Image(plt)})
        plt.close()
        
        # Precision-Recall Curve
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, label='Precision-Recall Curve')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend()
        wandb.log({"pr_curve": wandb.Image(plt)})
        plt.close()
        
        # Probability Distribution
        plt.figure(figsize=(10, 6))
        plt.hist(probabilities[true_labels == 0, 1], alpha=0.5, label='Control', bins=20)
        plt.hist(probabilities[true_labels == 1, 1], alpha=0.5, label='Internet Addicted', bins=20)
        plt.xlabel('Predicted Probability (Internet Addicted)')
        plt.ylabel('Frequency')
        plt.title('Probability Distribution by Class')
        plt.legend()
        wandb.log({"probability_distribution": wandb.Image(plt)})
        plt.close()
    
    def model_interpretability_analysis(self, model: nn.Module, test_loader: DataLoader) -> Dict[str, Any]:
        """
        Perform model interpretability analysis.
        
        Args:
            model: Trained model
            test_loader: Test data loader
            
        Returns:
            Dictionary of interpretability results
        """
        # This is a simplified interpretability analysis
        # In practice, you might want to use techniques like:
        # - Grad-CAM for CNN visualization
        # - SHAP values for feature importance
        # - Attention weights for transformer models
        
        model.eval()
        
        # Get feature importance from the first convolutional layer
        first_conv = model.conv_layers[0]
        weights = first_conv.weight.data.cpu().numpy()
        
        # Calculate channel importance (simplified)
        channel_importance = np.mean(np.abs(weights), axis=(0, 2, 3))
        
        # Create channel importance plot
        plt.figure(figsize=(12, 6))
        plt.bar(range(len(channel_importance)), channel_importance)
        plt.xlabel('Channel Index')
        plt.ylabel('Average Weight Magnitude')
        plt.title('Channel Importance (First Conv Layer)')
        plt.xticks(range(0, len(channel_importance), 5))
        wandb.log({"channel_importance": wandb.Image(plt)})
        plt.close()
        
        # Top channels
        top_channels = np.argsort(channel_importance)[-10:]
        
        results = {
            'channel_importance': channel_importance,
            'top_channels': top_channels,
            'top_channel_importance': channel_importance[top_channels]
        }
        
        self.logger.info("Top 10 most important channels:")
        for i, channel_idx in enumerate(reversed(top_channels)):
            self.logger.info(f"  {i+1}. Channel {channel_idx}: {channel_importance[channel_idx]:.4f}")
        
        return results
    
    def save_evaluation_results(self, results: Dict[str, Any], output_path: str):
        """Save evaluation results to disk."""
        import pickle
        
        with open(output_path, 'wb') as f:
            pickle.dump(results, f)
        
        self.logger.info(f"Saved evaluation results to {output_path}")


def main():
    """Main evaluation function."""
    # Load configuration
    config_path = Path(__file__).parent.parent / "configs" / "default_config.yaml"
    
    config = load_config_with_proper_types(config_path)
    
    # Initialize evaluator
    evaluator = EEGEvaluator(config)
    
    # Load data
    data_path = Path(config['data']['processed_data_path']) / "processed_data.npz"
    
    if not data_path.exists():
        print(f"Processed data not found at {data_path}")
        print("Please run the preprocessing script first.")
        return
    
    features, labels = evaluator.load_data(str(data_path))
    
    # Load trained model
    model_path = "artifacts/models/best_model.pth"
    if not Path(model_path).exists():
        print(f"Trained model not found at {model_path}")
        print("Please run the training script first.")
        return
    
    model = evaluator.load_trained_model(model_path)
    
    # Perform cross-validation
    print("Performing cross-validation...")
    cv_results = evaluator.cross_validation_evaluation(features, labels)
    
    # Statistical significance test
    print("Performing statistical significance test...")
    stats_results = evaluator.statistical_significance_test(cv_results)
    
    # Create test data loader for detailed analysis
    test_size = int(len(features) * config['training']['test_split'])
    val_size = int(len(features) * config['training']['validation_split'])
    train_size = len(features) - test_size - val_size
    
    # Use stratified sampling to ensure class balance
    train_val_features, test_features, train_val_labels, test_labels = train_test_split(
        features, labels, 
        test_size=test_size, 
        stratify=labels,
        random_state=config['training']['random_seed']
    )
    
    # Create test data loader
    test_dataset = TensorDataset(test_features, test_labels)
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False
    )
    
    # Detailed performance analysis
    print("Performing detailed performance analysis...")
    performance_results = evaluator.detailed_performance_analysis(model, test_loader)
    
    # Model interpretability analysis
    print("Performing model interpretability analysis...")
    interpretability_results = evaluator.model_interpretability_analysis(model, test_loader)
    
    # Combine all results
    all_results = {
        'cross_validation': cv_results,
        'statistical_significance': stats_results,
        'performance_analysis': performance_results,
        'interpretability': interpretability_results,
        'model_info': {
            'parameters': count_parameters(model),
            'architecture': config['model']['architecture']
        }
    }
    
    # Save results
    results_save_path = Path("artifacts/results/evaluation_results.pkl")
    results_save_path.parent.mkdir(parents=True, exist_ok=True)
    evaluator.save_evaluation_results(all_results, str(results_save_path))
    
    # Print summary
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    print(f"Model parameters: {all_results['model_info']['parameters']:,}")
    print(f"Cross-validation accuracy: {cv_results['accuracy_mean']:.4f} ± {cv_results['accuracy_std']:.4f}")
    print(f"Statistical significance: p = {stats_results['p_value']:.4f}")
    print(f"Test accuracy: {performance_results['metrics']['accuracy']:.4f}")
    print(f"Test AUC: {performance_results['metrics']['auc']:.4f}")
    print("="*60)
    
    # Close wandb
    wandb.finish()


if __name__ == "__main__":
    main() 