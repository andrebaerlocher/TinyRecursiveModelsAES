"""
Evaluation script for trained AES models
"""

import os
import sys
import json
import argparse
from typing import Dict, Any

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

from puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig
from torch.utils.data import DataLoader
from evaluators.aes_evaluator import AESEvaluator, AESEvaluatorConfig


def get_device():
    """Get the best available device"""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


class SimpleRecursiveModel(nn.Module):
    """
    Simplified Tiny Recursive Model for Essay Scoring
    (Must match the architecture in train_aes_m1.py)
    """

    def __init__(
        self,
        vocab_size: int,
        seq_len: int,
        num_classes: int,
        d_model: int = 128,
        d_hidden: int = 256,
        n_heads: int = 4,
        n_layers: int = 2,
        h_cycles: int = 2,
        l_cycles: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.num_classes = num_classes
        self.d_model = d_model
        self.h_cycles = h_cycles
        self.l_cycles = l_cycles

        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, d_model)

        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)

        # Encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_hidden,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Latent state
        self.latent_init = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # Recursive reasoning layers
        self.latent_update = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_hidden,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )

        # Answer update layers
        self.answer_init = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.answer_update = nn.Linear(d_model * 2, d_model)

        # Output head
        self.output_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, num_classes),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs: torch.Tensor, labels: torch.Tensor = None):
        """Forward pass with recursive reasoning"""
        batch_size = inputs.shape[0]

        # Embed tokens
        x = self.token_embedding(inputs)
        x = x + self.pos_encoding
        x = self.dropout(x)

        # Encode input
        encoded = self.encoder(x)

        # Initialize latent state and answer
        latent = self.latent_init.expand(batch_size, -1, -1)
        answer = self.answer_init.expand(batch_size, -1, -1)

        # Recursive reasoning: H-cycles
        for h in range(self.h_cycles):
            # L-cycles: Update latent state
            for l in range(self.l_cycles):
                reasoning_input = torch.cat([encoded, answer, latent], dim=1)
                latent_new = self.latent_update(reasoning_input)
                latent = latent_new[:, -1:, :]

            # Update answer based on latent state
            answer_input = torch.cat([answer, latent], dim=-1)
            answer = self.answer_update(answer_input)
            answer = torch.tanh(answer)

        # Generate final prediction from answer
        logits = self.output_head(answer.squeeze(1))

        # Expand logits to match sequence length format
        logits_expanded = logits.unsqueeze(1).expand(-1, self.seq_len, -1)

        return {"logits": logits_expanded, "final_logits": logits}


def load_model(checkpoint_path: str, device: torch.device) -> tuple:
    """Load model from checkpoint"""
    print(f"Loading checkpoint from {checkpoint_path}...")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint["config"]

    # Get model architecture from config
    with open(os.path.join(config["data_path"], "train", "dataset.json"), "r") as f:
        dataset_info = json.load(f)

    vocab_size = dataset_info["vocab_size"]
    seq_len = dataset_info["seq_len"]
    score_bins = dataset_info["score_bins"]

    # Create model
    model = SimpleRecursiveModel(
        vocab_size=vocab_size,
        seq_len=seq_len,
        num_classes=score_bins,
        d_model=config.get("d_model", 128),
        d_hidden=config.get("d_hidden", 256),
        n_heads=config.get("n_heads", 4),
        n_layers=config.get("n_layers", 2),
        h_cycles=config.get("h_cycles", 2),
        l_cycles=config.get("l_cycles", 3),
        dropout=config.get("dropout", 0.1),
    )

    # Load weights
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    print(f"✓ Model loaded successfully")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Training step: {checkpoint.get('step', 'N/A')}")
    print(f"  Best QWK: {checkpoint.get('best_qwk', 'N/A'):.4f}")

    return model, config, dataset_info


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    evaluator: AESEvaluator,
    device: torch.device,
    save_predictions: bool = False,
    output_file: str = None,
) -> Dict[str, Any]:
    """Evaluate model on dataset"""
    model.eval()
    evaluator.reset()

    all_predictions = []
    all_labels = []
    all_essay_ids = []

    with torch.no_grad():
        for set_name, batch, global_batch_size in tqdm(dataloader, desc="Evaluating"):
            # Move to device
            inputs = batch["inputs"].to(device)
            labels = batch["labels"].to(device)
            puzzle_identifiers = batch.get("puzzle_identifiers", None)

            # Forward pass
            outputs = model(inputs, labels)
            predictions = outputs["logits"]

            # Add to evaluator
            evaluator.add_batch(predictions, labels, puzzle_identifiers)

            # Save for detailed analysis if requested
            if save_predictions:
                pred_at_last = predictions[:, -1, :]
                pred_classes = torch.argmax(pred_at_last, dim=-1)
                label_classes = labels[:, -1]

                valid_mask = label_classes != -100

                all_predictions.extend(pred_classes[valid_mask].cpu().numpy().tolist())
                all_labels.extend(label_classes[valid_mask].cpu().numpy().tolist())
                if puzzle_identifiers is not None:
                    all_essay_ids.extend(
                        puzzle_identifiers[valid_mask].cpu().numpy().tolist()
                    )

    # Compute metrics
    metrics = evaluator.compute_metrics()

    # Save predictions if requested
    if save_predictions and output_file:
        output_data = {
            "predictions": all_predictions,
            "labels": all_labels,
            "essay_ids": all_essay_ids,
            "metrics": metrics,
        }

        # Convert predictions and labels to original scores
        pred_scores = [evaluator.denormalize_score(p) for p in all_predictions]
        label_scores = [evaluator.denormalize_score(l) for l in all_labels]

        output_data["pred_scores"] = pred_scores
        output_data["label_scores"] = label_scores

        os.makedirs(
            os.path.dirname(output_file) if os.path.dirname(output_file) else ".",
            exist_ok=True,
        )
        with open(output_file, "w") as f:
            json.dump(output_data, f, indent=2)

        print(f"\n✓ Predictions saved to {output_file}")

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained AES model")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="Path to dataset directory",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "test"],
        help="Dataset split to evaluate",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for evaluation",
    )
    parser.add_argument(
        "--save-predictions",
        action="store_true",
        help="Save predictions to file",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="predictions.json",
        help="Output file for predictions",
    )

    args = parser.parse_args()

    # Get device
    device = get_device()
    print(f"Using device: {device}")

    # Load model
    model, config, dataset_info = load_model(args.checkpoint, device)

    # Create dataloader
    print(f"\nLoading {args.split} dataset...")
    dataset = PuzzleDataset(
        PuzzleDatasetConfig(
            seed=42,
            dataset_paths=[args.data_path],
            global_batch_size=args.batch_size,
            test_set_mode=True,
            epochs_per_iter=1,
            rank=0,
            num_replicas=1,
        ),
        split=args.split,
    )

    dataloader = DataLoader(dataset, batch_size=None, num_workers=0, pin_memory=False)

    print(f"✓ Dataset loaded")
    print(f"  Total samples: {dataset.metadata.total_puzzles}")

    # Create evaluator
    min_score = dataset_info.get("min_score", 0)
    max_score = dataset_info.get("max_score", 12)
    score_bins = dataset_info.get("score_bins", 11)

    evaluator = AESEvaluator(
        AESEvaluatorConfig(
            name="aes",
            min_score=min_score,
            max_score=max_score,
            score_bins=score_bins,
        )
    )

    # Evaluate
    print(f"\nEvaluating on {args.split} set...")
    metrics = evaluate_model(
        model=model,
        dataloader=dataloader,
        evaluator=evaluator,
        device=device,
        save_predictions=args.save_predictions,
        output_file=args.output_file if args.save_predictions else None,
    )

    # Print results
    print("\n" + "=" * 50)
    print("Evaluation Results")
    print("=" * 50)
    print(f"Dataset: {args.data_path}")
    print(f"Split: {args.split}")
    print(f"Score range: {min_score}-{max_score}")
    print(f"\nMetrics:")
    print(f"  QWK:                {metrics['qwk']:.4f}")
    print(f"  MSE:                {metrics['mse']:.4f}")
    print(f"  RMSE:               {metrics['rmse']:.4f}")
    print(f"  Accuracy:           {metrics['accuracy']:.4f}")
    print(f"  Adjacent Accuracy:  {metrics['adjacent_accuracy']:.4f}")
    print(f"  Samples:            {metrics['num_samples']}")
    print("=" * 50)

    # Interpretation
    print("\nInterpretation:")
    qwk = metrics["qwk"]
    if qwk < 0.40:
        print("  QWK < 0.40: Poor agreement")
    elif qwk < 0.60:
        print("  QWK 0.40-0.60: Moderate agreement")
    elif qwk < 0.80:
        print("  QWK 0.60-0.80: Substantial agreement")
    else:
        print("  QWK > 0.80: Almost perfect agreement")


if __name__ == "__main__":
    main()
