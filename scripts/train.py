#!/usr/bin/env python3
"""
Training Script for EEG CNN Model

This script implements the training pipeline for the 5-layer CNN model
with Weights & Biases integration for experiment tracking.
"""

import os
import sys
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import logging
import wandb
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from models.cnn_model import create_model, count_parameters


class EarlyStopping:
    """Early stopping utility to prevent overfitting."""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0, restore_best_weights: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, val_loss: float, model: nn.Module) -> bool:
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_weights = model.state_dict().copy()
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_weights = model.state_dict().copy()
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            if self.restore_best_weights:
                model.load_state_dict(self.best_weights)
            return True
        return False


class EEGTrainer:
    """Training pipeline for EEG CNN model."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.training_config = config['training']
        self.model_config = config['model']
        self.wandb_config = config['wandb']
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Set random seeds
        torch.manual_seed(self.training_config['random_seed'])
        np.random.seed(self.training_config['random_seed'])
        
        # Setup device
        if self.training_config['device'] == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(self.training_config['device'])
        
        self.logger.info(f"Using device: {self.device}")
        
        # Initialize model
        self.model = create_model(config)
        self.model.to(self.device)
        
        # Setup optimizer and loss function
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.training_config['learning_rate'],
            weight_decay=self.training_config['weight_decay']
        )
        
        self.criterion = nn.CrossEntropyLoss()
        
        # Setup early stopping
        self.early_stopping = EarlyStopping(
            patience=self.training_config['early_stopping_patience']
        )
        
        # Initialize wandb
        self._setup_wandb()
        
    def _setup_wandb(self):
        """Initialize Weights & Biases logging."""
        if self.wandb_config['entity'] is None:
            # Try to get entity from environment or use None
            entity = os.getenv('WANDB_ENTITY')
        else:
            entity = self.wandb_config['entity']
        
        wandb.init(
            project=self.wandb_config['project'],
            entity=entity,
            tags=self.wandb_config['tags'],
            config={
                'model': self.model_config,
                'training': self.training_config,
                'preprocessing': self.config['preprocessing']
            }
        )
        
        # Log model architecture
        wandb.watch(self.model, log="all")
        
    def load_data(self, data_path: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Load preprocessed data.
        
        Args:
            data_path: Path to processed data file
            
        Returns:
            Tuple of (features, labels)
        """
        data = np.load(data_path)
        features = torch.FloatTensor(data['features'])
        labels = torch.LongTensor(data['labels'])
        
        self.logger.info(f"Loaded data: {features.shape}, labels: {labels.shape}")
        return features, labels
    
    def create_data_loaders(self, features: torch.Tensor, labels: torch.Tensor) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Create train, validation, and test data loaders.
        
        Args:
            features: Feature tensor
            labels: Label tensor
            
        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        # Calculate split sizes
        total_size = len(features)
        test_size = int(total_size * self.training_config['test_split'])
        val_size = int(total_size * self.training_config['validation_split'])
        train_size = total_size - test_size - val_size
        
        # Split data
        train_features, val_features, test_features = random_split(
            features, [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(self.training_config['random_seed'])
        )
        
        train_labels, val_labels, test_labels = random_split(
            labels, [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(self.training_config['random_seed'])
        )
        
        # Create datasets
        train_dataset = TensorDataset(train_features.dataset[train_features.indices], 
                                    train_labels.dataset[train_labels.indices])
        val_dataset = TensorDataset(val_features.dataset[val_features.indices], 
                                  val_labels.dataset[val_labels.indices])
        test_dataset = TensorDataset(test_features.dataset[test_features.indices], 
                                   test_labels.dataset[test_labels.indices])
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.training_config['batch_size'],
            shuffle=True,
            num_workers=0  # Set to 0 for debugging, increase for production
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.training_config['batch_size'],
            shuffle=False,
            num_workers=0
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.training_config['batch_size'],
            shuffle=False,
            num_workers=0
        )
        
        self.logger.info(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
        
        return train_loader, val_loader, test_loader
    
    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            
        Returns:
            Tuple of (loss, accuracy)
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_features, batch_labels in tqdm(train_loader, desc="Training"):
            batch_features = batch_features.to(self.device)
            batch_labels = batch_labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(batch_features)
            loss = self.criterion(outputs, batch_labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += batch_labels.size(0)
            correct += (predicted == batch_labels).sum().item()
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100 * correct / total
        
        return avg_loss, accuracy
    
    def validate_epoch(self, val_loader: DataLoader) -> Tuple[float, float, np.ndarray, np.ndarray]:
        """
        Validate for one epoch.
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Tuple of (loss, accuracy, predictions, true_labels)
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch_features, batch_labels in tqdm(val_loader, desc="Validation"):
                batch_features = batch_features.to(self.device)
                batch_labels = batch_labels.to(self.device)
                
                # Forward pass
                outputs = self.model(batch_features)
                loss = self.criterion(outputs, batch_labels)
                
                # Statistics
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += batch_labels.size(0)
                correct += (predicted == batch_labels).sum().item()
                
                # Store predictions and labels
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(batch_labels.cpu().numpy())
        
        avg_loss = total_loss / len(val_loader)
        accuracy = 100 * correct / total
        
        return avg_loss, accuracy, np.array(all_predictions), np.array(all_labels)
    
    def calculate_metrics(self, predictions: np.ndarray, true_labels: np.ndarray) -> Dict[str, float]:
        """
        Calculate evaluation metrics.
        
        Args:
            predictions: Predicted labels
            true_labels: True labels
            
        Returns:
            Dictionary of metrics
        """
        accuracy = accuracy_score(true_labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels, predictions, average='binary'
        )
        
        # Calculate AUC (requires probability scores)
        # For now, we'll use the binary predictions
        # In practice, you'd want to use softmax probabilities
        auc = roc_auc_score(true_labels, predictions)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc
        }
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader) -> Dict[str, Any]:
        """
        Main training loop.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            
        Returns:
            Training history
        """
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'val_metrics': []
        }
        
        best_val_acc = 0.0
        
        for epoch in range(self.training_config['num_epochs']):
            self.logger.info(f"Epoch {epoch+1}/{self.training_config['num_epochs']}")
            
            # Train
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # Validate
            val_loss, val_acc, val_preds, val_labels = self.validate_epoch(val_loader)
            
            # Calculate metrics
            val_metrics = self.calculate_metrics(val_preds, val_labels)
            
            # Log to wandb
            wandb.log({
                'epoch': epoch,
                'train_loss': train_loss,
                'train_accuracy': train_acc,
                'val_loss': val_loss,
                'val_accuracy': val_acc,
                **{f'val_{k}': v for k, v in val_metrics.items()}
            })
            
            # Update history
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            history['val_metrics'].append(val_metrics)
            
            # Log progress
            self.logger.info(
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%"
            )
            
            # Early stopping
            if self.early_stopping(val_loss, self.model):
                self.logger.info("Early stopping triggered")
                break
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(self.model.state_dict(), 'best_model.pth')
                wandb.save('best_model.pth')
        
        return history
    
    def evaluate(self, test_loader: DataLoader) -> Dict[str, float]:
        """
        Evaluate model on test set.
        
        Args:
            test_loader: Test data loader
            
        Returns:
            Test metrics
        """
        self.model.eval()
        all_predictions = []
        all_labels = []
        all_probabilities = []
        
        with torch.no_grad():
            for batch_features, batch_labels in tqdm(test_loader, desc="Testing"):
                batch_features = batch_features.to(self.device)
                batch_labels = batch_labels.to(self.device)
                
                # Forward pass
                outputs = self.model(batch_features)
                probabilities = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs.data, 1)
                
                # Store results
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(batch_labels.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        # Calculate metrics
        test_metrics = self.calculate_metrics(
            np.array(all_predictions), 
            np.array(all_labels)
        )
        
        # Log to wandb
        wandb.log({'test_metrics': test_metrics})
        
        # Create confusion matrix
        self._plot_confusion_matrix(
            np.array(all_predictions), 
            np.array(all_labels)
        )
        
        # Create ROC curve
        self._plot_roc_curve(
            np.array(all_probabilities)[:, 1],  # Probability of positive class
            np.array(all_labels)
        )
        
        return test_metrics
    
    def _plot_confusion_matrix(self, predictions: np.ndarray, true_labels: np.ndarray):
        """Plot and log confusion matrix."""
        from sklearn.metrics import confusion_matrix
        
        cm = confusion_matrix(true_labels, predictions)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        wandb.log({"confusion_matrix": wandb.Image(plt)})
        plt.close()
    
    def _plot_roc_curve(self, probabilities: np.ndarray, true_labels: np.ndarray):
        """Plot and log ROC curve."""
        from sklearn.metrics import roc_curve
        
        fpr, tpr, _ = roc_curve(true_labels, probabilities)
        auc = roc_auc_score(true_labels, probabilities)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        
        wandb.log({"roc_curve": wandb.Image(plt)})
        plt.close()


def main():
    """Main training function."""
    # Load configuration
    config_path = Path(__file__).parent.parent / "configs" / "default_config.yaml"
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize trainer
    trainer = EEGTrainer(config)
    
    # Load data
    data_path = Path(config['data']['processed_data_path']) / "processed_data.npz"
    
    if not data_path.exists():
        print(f"Processed data not found at {data_path}")
        print("Please run the preprocessing script first.")
        return
    
    features, labels = trainer.load_data(str(data_path))
    
    # Create data loaders
    train_loader, val_loader, test_loader = trainer.create_data_loaders(features, labels)
    
    # Train model
    print("Starting training...")
    history = trainer.train(train_loader, val_loader)
    
    # Evaluate model
    print("Evaluating model...")
    test_metrics = trainer.evaluate(test_loader)
    
    # Print results
    print("\n" + "="*50)
    print("TRAINING COMPLETED")
    print("="*50)
    print(f"Model parameters: {count_parameters(trainer.model):,}")
    print(f"Best validation accuracy: {max(history['val_acc']):.2f}%")
    print("\nTest Results:")
    for metric, value in test_metrics.items():
        print(f"  {metric.capitalize()}: {value:.4f}")
    print("="*50)
    
    # Close wandb
    wandb.finish()


if __name__ == "__main__":
    main() 