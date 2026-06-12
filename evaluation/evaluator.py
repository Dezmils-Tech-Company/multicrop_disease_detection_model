"""Model evaluation on test set."""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Tuple
import json
import numpy as np
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report

from training.metrics import MetricsCalculator, compute_confusion_matrix


class Evaluator:
    """Evaluate model on test set."""
    
    def __init__(
        self,
        model: nn.Module,
        test_loader: DataLoader,
        class_mapping: Dict[int, str],
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Args:
            model: Trained model
            test_loader: Test data loader
            class_mapping: Index to class name mapping
            device: Device for evaluation
        """
        self.model = model.to(device)
        self.test_loader = test_loader
        self.class_mapping = class_mapping
        self.device = device
        self.model.eval()
    
    def evaluate(self) -> Dict:
        """
        Evaluate model on test set.
        
        Returns:
            Dict with evaluation metrics
        """
        num_classes = len(self.class_mapping)
        metrics = MetricsCalculator(num_classes)
        all_predictions = []
        all_targets = []
        
        print("Evaluating on test set...")
        
        with torch.no_grad():
            for images, labels, class_names in self.test_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                
                metrics.update(outputs, labels)
                
                # Store for confusion matrix
                _, preds = outputs.max(1)
                all_predictions.extend(preds.cpu().numpy())
                all_targets.extend(labels.cpu().numpy())
        
        # Get metrics
        overall_metrics = metrics.get_metrics()
        per_class_metrics = metrics.get_per_class_metrics()
        
        # Confusion matrix
        cm = confusion_matrix(all_targets, all_predictions, labels=list(range(num_classes)))
        
        # Classification report
        class_names_list = [self.class_mapping[i] for i in range(num_classes)]
        report = classification_report(
            all_targets,
            all_predictions,
            target_names=class_names_list,
            output_dict=True,
            zero_division=0
        )
        
        results = {
            "overall_metrics": overall_metrics,
            "per_class_metrics": per_class_metrics,
            "confusion_matrix": cm.tolist(),
            "classification_report": report,
        }
        
        return results
    
    def save_results(self, results: Dict, output_dir: str = "logs"):
        """Save evaluation results to files."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save overall results
        with open(output_path / "evaluation_results.json", "w") as f:
            json.dump(results, f, indent=2)
        
        # Save confusion matrix
        cm = np.array(results["confusion_matrix"])
        np.save(output_path / "confusion_matrix.npy", cm)
        
        print(f"Results saved to {output_path}")
