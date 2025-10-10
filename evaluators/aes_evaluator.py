"""
Automated Essay Scoring (AES) Evaluator
Implements evaluation metrics for essay scoring including QWK and MSE
"""

import torch
import numpy as np
from typing import Dict, Any, Optional
from sklearn.metrics import cohen_kappa_score, mean_squared_error
import pydantic


class AESEvaluatorConfig(pydantic.BaseModel):
    """Configuration for AES Evaluator"""

    name: str = "aes"
    min_score: int = 0
    max_score: int = 12
    score_bins: int = 11
    metric_types: list = ["qwk", "mse", "accuracy"]


class AESEvaluator:
    """
    Evaluator for Automated Essay Scoring tasks
    Computes Quadratic Weighted Kappa (QWK), MSE, and accuracy
    """

    def __init__(self, config: AESEvaluatorConfig):
        self.config = config
        self.reset()

    def reset(self):
        """Reset accumulated predictions and labels"""
        self.predictions = []
        self.labels = []
        self.essay_ids = []

    def denormalize_score(self, normalized_bin: int) -> int:
        """Convert normalized bin back to original score"""
        # Bin to normalized value
        if self.config.score_bins <= 1:
            normalized = 0.5
        else:
            normalized = normalized_bin / (self.config.score_bins - 1)

        # Normalized to original score
        score = (
            normalized * (self.config.max_score - self.config.min_score)
            + self.config.min_score
        )
        return int(round(score))

    def add_batch(
        self,
        predictions: torch.Tensor,
        labels: torch.Tensor,
        essay_ids: Optional[torch.Tensor] = None,
    ):
        """
        Add a batch of predictions and labels

        Args:
            predictions: Model predictions (logits or class indices) [batch_size, seq_len, num_classes]
            labels: Ground truth labels [batch_size, seq_len]
            essay_ids: Optional essay identifiers [batch_size]
        """
        # Handle predictions - take the last non-ignored position
        if predictions.dim() == 3:  # [batch, seq, classes]
            # Get predictions at last position
            pred_at_last = predictions[:, -1, :]  # [batch, classes]
            pred_classes = torch.argmax(pred_at_last, dim=-1)  # [batch]
        elif predictions.dim() == 2:  # [batch, seq]
            pred_classes = predictions[:, -1]
        else:  # [batch]
            pred_classes = predictions

        # Get labels at last position (where score is stored)
        if labels.dim() == 2:
            label_classes = labels[:, -1]
        else:
            label_classes = labels

        # Filter out ignored labels
        valid_mask = label_classes != -100
        valid_preds = pred_classes[valid_mask]
        valid_labels = label_classes[valid_mask]

        # Convert to CPU numpy
        valid_preds_np = valid_preds.cpu().numpy()
        valid_labels_np = valid_labels.cpu().numpy()

        # Denormalize scores
        pred_scores = [self.denormalize_score(int(p)) for p in valid_preds_np]
        label_scores = [self.denormalize_score(int(l)) for l in valid_labels_np]

        self.predictions.extend(pred_scores)
        self.labels.extend(label_scores)

        if essay_ids is not None:
            valid_ids = essay_ids[valid_mask]
            self.essay_ids.extend(valid_ids.cpu().numpy().tolist())

    def compute_qwk(self, predictions: np.ndarray, labels: np.ndarray) -> float:
        """
        Compute Quadratic Weighted Kappa

        Args:
            predictions: Predicted scores
            labels: True scores

        Returns:
            QWK score
        """
        if len(predictions) == 0 or len(labels) == 0:
            return 0.0

        try:
            qwk = cohen_kappa_score(
                labels,
                predictions,
                weights="quadratic",
                labels=list(range(self.config.min_score, self.config.max_score + 1)),
            )
            return float(qwk)
        except Exception as e:
            print(f"Error computing QWK: {e}")
            return 0.0

    def compute_mse(self, predictions: np.ndarray, labels: np.ndarray) -> float:
        """Compute Mean Squared Error"""
        if len(predictions) == 0 or len(labels) == 0:
            return float("inf")

        return float(mean_squared_error(labels, predictions))

    def compute_rmse(self, predictions: np.ndarray, labels: np.ndarray) -> float:
        """Compute Root Mean Squared Error"""
        mse = self.compute_mse(predictions, labels)
        return float(np.sqrt(mse))

    def compute_accuracy(self, predictions: np.ndarray, labels: np.ndarray) -> float:
        """Compute exact match accuracy"""
        if len(predictions) == 0 or len(labels) == 0:
            return 0.0

        return float(np.mean(predictions == labels))

    def compute_adjacent_accuracy(
        self, predictions: np.ndarray, labels: np.ndarray
    ) -> float:
        """
        Compute accuracy allowing for ±1 score difference
        Often used in essay scoring evaluation
        """
        if len(predictions) == 0 or len(labels) == 0:
            return 0.0

        return float(np.mean(np.abs(predictions - labels) <= 1))

    def compute_metrics(self) -> Dict[str, float]:
        """
        Compute all evaluation metrics

        Returns:
            Dictionary of metric names to values
        """
        if len(self.predictions) == 0 or len(self.labels) == 0:
            return {
                "qwk": 0.0,
                "mse": float("inf"),
                "rmse": float("inf"),
                "accuracy": 0.0,
                "adjacent_accuracy": 0.0,
                "num_samples": 0,
            }

        predictions = np.array(self.predictions)
        labels = np.array(self.labels)

        metrics = {
            "qwk": self.compute_qwk(predictions, labels),
            "mse": self.compute_mse(predictions, labels),
            "rmse": self.compute_rmse(predictions, labels),
            "accuracy": self.compute_accuracy(predictions, labels),
            "adjacent_accuracy": self.compute_adjacent_accuracy(predictions, labels),
            "num_samples": len(predictions),
        }

        return metrics

    def get_predictions(self) -> tuple:
        """Return accumulated predictions and labels"""
        return np.array(self.predictions), np.array(self.labels)

    def __call__(
        self,
        model: torch.nn.Module,
        dataloader,
        device: torch.device,
        max_batches: Optional[int] = None,
    ) -> Dict[str, float]:
        """
        Evaluate model on a dataloader

        Args:
            model: The model to evaluate
            dataloader: DataLoader providing batches
            device: Device to run evaluation on
            max_batches: Maximum number of batches to evaluate (None = all)

        Returns:
            Dictionary of metrics
        """
        model.eval()
        self.reset()

        with torch.no_grad():
            for batch_idx, (set_name, batch, global_batch_size) in enumerate(
                dataloader
            ):
                if max_batches is not None and batch_idx >= max_batches:
                    break

                # Move batch to device
                inputs = batch["inputs"].to(device)
                labels = batch["labels"].to(device)
                puzzle_identifiers = batch.get("puzzle_identifiers", None)

                # Forward pass
                outputs = model(inputs, labels)

                # Get predictions (logits)
                if isinstance(outputs, dict):
                    predictions = outputs.get("logits", outputs.get("predictions"))
                else:
                    predictions = outputs

                # Add to evaluator
                self.add_batch(predictions, labels, puzzle_identifiers)

        # Compute final metrics
        metrics = self.compute_metrics()

        model.train()
        return metrics


def create_evaluator(config: Dict[str, Any]) -> AESEvaluator:
    """Factory function to create AES evaluator"""
    evaluator_config = AESEvaluatorConfig(**config)
    return AESEvaluator(evaluator_config)
