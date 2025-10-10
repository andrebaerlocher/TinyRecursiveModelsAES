"""
Training script for Automated Essay Scoring (AES) using Tiny Recursive Models
Optimized for MacBook Pro M1 with 16GB RAM
"""

import os
import sys
import json
import math
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import wandb

from puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig
from evaluators.aes_evaluator import AESEvaluator, AESEvaluatorConfig
from models.ema import EMAHelper


def get_device():
    """Get the best available device for M1 Mac"""
    if torch.backends.mps.is_available():
        print("Using MPS (Metal Performance Shaders) backend for M1 Mac")
        return torch.device("mps")
    else:
        print("MPS not available, using CPU")
        return torch.device("cpu")


def set_seed(seed: int):
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class SimpleRecursiveModel(nn.Module):
    """
    Simplified Tiny Recursive Model for Essay Scoring
    Optimized for M1 Mac with reduced memory footprint
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

    def forward(self, inputs: torch.Tensor, labels: Optional[torch.Tensor] = None):
        """
        Forward pass with recursive reasoning

        Args:
            inputs: [batch_size, seq_len]
            labels: [batch_size, seq_len] (optional)

        Returns:
            Dictionary with logits and loss
        """
        batch_size = inputs.shape[0]

        # Embed tokens
        x = self.token_embedding(inputs)  # [B, L, D]
        x = x + self.pos_encoding  # Add positional encoding
        x = self.dropout(x)

        # Encode input
        encoded = self.encoder(x)  # [B, L, D]

        # Initialize latent state and answer
        latent = self.latent_init.expand(batch_size, -1, -1)  # [B, 1, D]
        answer = self.answer_init.expand(batch_size, -1, -1)  # [B, 1, D]

        # Recursive reasoning: H-cycles
        for h in range(self.h_cycles):
            # L-cycles: Update latent state
            for l in range(self.l_cycles):
                # Concatenate encoded input, current answer, and latent
                reasoning_input = torch.cat(
                    [encoded, answer, latent], dim=1
                )  # [B, L+2, D]

                # Update latent state
                latent_new = self.latent_update(reasoning_input)  # [B, L+2, D]
                latent = latent_new[:, -1:, :]  # Take last position [B, 1, D]

            # Update answer based on latent state
            answer_input = torch.cat([answer, latent], dim=-1)  # [B, 1, 2*D]
            answer = self.answer_update(answer_input)  # [B, 1, D]
            answer = torch.tanh(answer)  # Normalize

        # Generate final prediction from answer
        logits = self.output_head(answer.squeeze(1))  # [B, num_classes]

        # Expand logits to match sequence length format
        logits_expanded = logits.unsqueeze(1).expand(
            -1, self.seq_len, -1
        )  # [B, L, num_classes]

        result = {"logits": logits_expanded, "final_logits": logits}

        # Compute loss if labels provided
        if labels is not None:
            # Get labels at last position (where score is stored)
            labels_last = labels[:, -1]  # [B]
            valid_mask = labels_last != -100

            if valid_mask.sum() > 0:
                loss = nn.functional.cross_entropy(
                    logits[valid_mask], labels_last[valid_mask]
                )
                result["loss"] = loss
            else:
                result["loss"] = torch.tensor(0.0, device=logits.device)

        return result


class AESTrainer:
    """Trainer for Automated Essay Scoring"""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        test_loader: DataLoader,
        evaluator: AESEvaluator,
        device: torch.device,
        config: Dict[str, Any],
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.evaluator = evaluator
        self.device = device
        self.config = config

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.get("lr", 3e-4),
            weight_decay=config.get("weight_decay", 0.1),
            betas=(config.get("beta1", 0.9), config.get("beta2", 0.999)),
        )

        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.get("epochs", 10000),
            eta_min=config.get("lr", 3e-4) * config.get("lr_min_ratio", 0.1),
        )

        # EMA
        self.use_ema = config.get("ema", True)
        if self.use_ema:
            self.ema_helper = EMAHelper(mu=config.get("ema_rate", 0.999), device=device)
            self.ema_helper.register(self.model)

        # Training state
        self.step = 0
        self.best_qwk = -1.0

        # Wandb logging
        self.use_wandb = config.get("use_wandb", False)
        if self.use_wandb:
            wandb.init(
                project=config.get("project_name", "TinyRecursiveModels-AES"),
                name=config.get("run_name", None),
                config=config,
            )

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        progress_bar = tqdm(self.train_loader, desc=f"Training")

        for set_name, batch, global_batch_size in progress_bar:
            # Move to device
            inputs = batch["inputs"].to(self.device)
            labels = batch["labels"].to(self.device)

            # Forward pass
            outputs = self.model(inputs, labels)
            loss = outputs["loss"]

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            # Update EMA
            if self.use_ema:
                self.ema_helper.update(self.model)

            # Track metrics
            total_loss += loss.item()
            num_batches += 1
            self.step += 1

            # Update progress bar
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

            # Log to wandb
            if self.use_wandb and self.step % 10 == 0:
                wandb.log({"train/loss": loss.item(), "train/step": self.step})

        # Update scheduler
        self.scheduler.step()

        avg_loss = total_loss / max(num_batches, 1)
        return {"loss": avg_loss}

    @torch.no_grad()
    def evaluate(self, use_ema: bool = False) -> Dict[str, float]:
        """Evaluate on test set"""
        # Switch to EMA weights if requested
        if use_ema and self.use_ema:
            self.ema_helper.ema(self.model)

        self.model.eval()
        self.evaluator.reset()

        for set_name, batch, global_batch_size in tqdm(
            self.test_loader, desc="Evaluating"
        ):
            # Move to device
            inputs = batch["inputs"].to(self.device)
            labels = batch["labels"].to(self.device)
            puzzle_identifiers = batch.get("puzzle_identifiers", None)

            # Forward pass
            outputs = self.model(inputs, labels)
            predictions = outputs["logits"]

            # Add to evaluator
            self.evaluator.add_batch(predictions, labels, puzzle_identifiers)

        # Compute metrics
        metrics = self.evaluator.compute_metrics()

        # Restore original weights if using EMA
        if use_ema and self.use_ema:
            self.ema_helper.restore(self.model)

        self.model.train()
        return metrics

    def train(self, num_epochs: int):
        """Main training loop"""
        print(f"Starting training for {num_epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")

        eval_interval = self.config.get("eval_interval", 500)
        min_eval_interval = self.config.get("min_eval_interval", 0)

        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")

            # Train
            train_metrics = self.train_epoch()
            print(f"Train Loss: {train_metrics['loss']:.4f}")

            # Evaluate
            if epoch >= min_eval_interval and (epoch + 1) % eval_interval == 0:
                print("Evaluating...")
                eval_metrics = self.evaluate(use_ema=self.use_ema)

                print(f"Evaluation Metrics:")
                print(f"  QWK: {eval_metrics['qwk']:.4f}")
                print(f"  MSE: {eval_metrics['mse']:.4f}")
                print(f"  RMSE: {eval_metrics['rmse']:.4f}")
                print(f"  Accuracy: {eval_metrics['accuracy']:.4f}")
                print(f"  Adjacent Accuracy: {eval_metrics['adjacent_accuracy']:.4f}")

                # Log to wandb
                if self.use_wandb:
                    wandb.log(
                        {
                            f"eval/{k}": v
                            for k, v in eval_metrics.items()
                            if k != "num_samples"
                        }
                    )

                # Save best model
                if eval_metrics["qwk"] > self.best_qwk:
                    self.best_qwk = eval_metrics["qwk"]
                    self.save_checkpoint("best_model.pt")
                    print(f"Saved new best model (QWK: {self.best_qwk:.4f})")

            # Save periodic checkpoint
            if (epoch + 1) % 1000 == 0:
                self.save_checkpoint(f"checkpoint_epoch_{epoch+1}.pt")

    def save_checkpoint(self, filename: str):
        """Save model checkpoint"""
        checkpoint_dir = self.config.get("checkpoint_path", "checkpoints/aes")
        os.makedirs(checkpoint_dir, exist_ok=True)

        checkpoint_path = os.path.join(checkpoint_dir, filename)

        checkpoint = {
            "step": self.step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_qwk": self.best_qwk,
            "config": self.config,
        }

        if self.use_ema:
            checkpoint["ema_state_dict"] = self.ema_helper.state_dict()

        torch.save(checkpoint, checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Train Tiny Recursive Model for AES on M1 Mac"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="Path to processed dataset directory",
    )
    parser.add_argument(
        "--batch-size", type=int, default=16, help="Batch size (default: 16)"
    )
    parser.add_argument(
        "--epochs", type=int, default=10000, help="Number of epochs (default: 10000)"
    )
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument(
        "--d-model", type=int, default=128, help="Model embedding dimension"
    )
    parser.add_argument(
        "--d-hidden", type=int, default=256, help="Hidden layer dimension"
    )
    parser.add_argument(
        "--n-heads", type=int, default=4, help="Number of attention heads"
    )
    parser.add_argument(
        "--n-layers", type=int, default=2, help="Number of encoder layers"
    )
    parser.add_argument(
        "--h-cycles", type=int, default=2, help="Number of high-level reasoning cycles"
    )
    parser.add_argument(
        "--l-cycles", type=int, default=3, help="Number of low-level reasoning cycles"
    )
    parser.add_argument(
        "--eval-interval", type=int, default=500, help="Evaluation interval (epochs)"
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default="checkpoints/aes",
        help="Checkpoint directory",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--use-wandb", action="store_true", help="Use Weights & Biases logging"
    )
    parser.add_argument("--project-name", type=str, default="TinyRecursiveModels-AES")
    parser.add_argument("--run-name", type=str, default=None, help="Run name for wandb")

    args = parser.parse_args()

    # Set seed
    set_seed(args.seed)

    # Get device
    device = get_device()

    # Create config
    config = {
        "data_path": args.data_path,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "lr": args.lr,
        "lr_min_ratio": 0.1,
        "weight_decay": 0.1,
        "beta1": 0.9,
        "beta2": 0.999,
        "d_model": args.d_model,
        "d_hidden": args.d_hidden,
        "n_heads": args.n_heads,
        "n_layers": args.n_layers,
        "h_cycles": args.h_cycles,
        "l_cycles": args.l_cycles,
        "dropout": 0.1,
        "ema": True,
        "ema_rate": 0.999,
        "eval_interval": args.eval_interval,
        "min_eval_interval": 0,
        "checkpoint_path": args.checkpoint_path,
        "seed": args.seed,
        "use_wandb": args.use_wandb,
        "project_name": args.project_name,
        "run_name": args.run_name,
    }

    # Create dataloaders
    print("Loading datasets...")
    train_dataset = PuzzleDataset(
        PuzzleDatasetConfig(
            seed=args.seed,
            dataset_paths=[args.data_path],
            global_batch_size=args.batch_size,
            test_set_mode=False,
            epochs_per_iter=1,
            rank=0,
            num_replicas=1,
        ),
        split="train",
    )

    test_dataset = PuzzleDataset(
        PuzzleDatasetConfig(
            seed=args.seed,
            dataset_paths=[args.data_path],
            global_batch_size=args.batch_size,
            test_set_mode=True,
            epochs_per_iter=1,
            rank=0,
            num_replicas=1,
        ),
        split="test",
    )

    train_loader = DataLoader(
        train_dataset, batch_size=None, num_workers=0, pin_memory=False
    )
    test_loader = DataLoader(
        test_dataset, batch_size=None, num_workers=0, pin_memory=False
    )

    # Get metadata
    metadata = train_dataset.metadata
    print(f"Dataset metadata:")
    print(f"  Vocabulary size: {metadata.vocab_size}")
    print(f"  Sequence length: {metadata.seq_len}")
    print(f"  Total puzzles: {metadata.total_puzzles}")

    # Load dataset metadata for score info
    with open(os.path.join(args.data_path, "train", "dataset.json"), "r") as f:
        dataset_info = json.load(f)

    min_score = dataset_info.get("min_score", 0)
    max_score = dataset_info.get("max_score", 12)
    score_bins = dataset_info.get("score_bins", 11)

    print(f"Score range: {min_score} - {max_score} (bins: {score_bins})")

    # Create model
    print("Creating model...")
    model = SimpleRecursiveModel(
        vocab_size=metadata.vocab_size,
        seq_len=metadata.seq_len,
        num_classes=score_bins,
        d_model=args.d_model,
        d_hidden=args.d_hidden,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        h_cycles=args.h_cycles,
        l_cycles=args.l_cycles,
        dropout=config["dropout"],
    )

    # Create evaluator
    evaluator = AESEvaluator(
        AESEvaluatorConfig(
            name="aes",
            min_score=min_score,
            max_score=max_score,
            score_bins=score_bins,
        )
    )

    # Create trainer
    trainer = AESTrainer(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        evaluator=evaluator,
        device=device,
        config=config,
    )

    # Train
    trainer.train(num_epochs=args.epochs)

    print("\nTraining complete!")
    print(f"Best QWK: {trainer.best_qwk:.4f}")


if __name__ == "__main__":
    main()
