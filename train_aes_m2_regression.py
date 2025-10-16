"""Training script for Automated Essay Scoring (AES) using Tiny Recursive Models (Regression)
Optimized for Mac Studio M2 Ultra with 192GB RAM"""

import os
import sys
import json
import math
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
import argparse
import time
import subprocess
import sys
import copy
from datetime import datetime, timedelta

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import wandb

from puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig
from evaluators.aes_evaluator import AESEvaluator, AESEvaluatorConfig
from models.ema import EMAHelper
from models.recursive_reasoning.trm_regression import (
    TinyRecursiveReasoningModel_ACTV1_Regression,
)


def get_device():
    """Get the best available device for M2 Mac"""
    if torch.backends.mps.is_available():
        print("Using MPS (Metal Performance Shaders) backend for M2 Mac")
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


class MSELossWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.loss_fn = nn.MSELoss()

    def initial_carry(self, batch):
        return self.model.initial_carry(batch)

    def forward(self, carry, batch, return_keys=[]):
        carry, outputs = self.model(carry, batch)

        # Ensure labels are float for MSE loss
        labels = batch["labels"].float()

        loss = self.loss_fn(outputs["prediction"].squeeze(), labels.squeeze()[:, 0])

        # The rest of the returned values are for compatibility with the trainer
        metrics = {}
        all_finish = carry.halted.all()

        return carry, loss, metrics, outputs, all_finish


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
        self.model_config = config["model_config"]

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
            self.ema_helper = EMAHelper(mu=config.get("ema_rate", 0.999))
            self.ema_helper.register(self.model)

        # Training state
        self.step = 0
        self.best_qwk = -1.0
        self.carry = None
        self.early_stopping_counter = 0
        self.early_stopping_patience = config.get("early_stopping_patience", 5)

        # Wandb logging
        self.use_wandb = config.get("use_wandb", False)
        if self.use_wandb:
            wandb.init(
                project=config.get("project_name", "TRM-AES-Regression"),
                name=config.get("run_name", None),
                config=config,
                entity=config.get(
                    "entity_name", "andre-baerlocher-lehrmittelverlag-st-gallen"
                ),
            )
            wandb.watch(self.model, log_freq=10)

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        progress_bar = tqdm(self.train_loader, desc=f"Training")

        for set_name, batch, global_batch_size in progress_bar:
            # Move to device
            batch = {k: v.to(self.device) for k, v in batch.items()}

            # Init carry if it is None
            if self.carry is None:
                with torch.device(self.device):
                    self.carry = self.model.initial_carry(batch)

            # Forward pass
            self.carry, loss, metrics, _, _ = self.model(
                carry=self.carry, batch=batch, return_keys=[]
            )
            loss = loss / self.model_config["batch_size"]

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
                wandb.log({"train/loss": loss.item()}, step=self.step)

        # Update scheduler
        self.scheduler.step()

        avg_loss = total_loss / max(num_batches, 1)
        return {"loss": avg_loss}

    @torch.no_grad()
    def evaluate(self, use_ema: bool = False) -> Dict[str, float]:
        """Evaluate on test set"""
        original_state_dict = None
        if use_ema and self.use_ema:
            original_state_dict = copy.deepcopy(self.model.state_dict())
            self.ema_helper.ema(self.model)

        self.model.eval()
        
        all_preds = []
        all_labels = []

        eval_carry = None
        for set_name, batch, global_batch_size in tqdm(self.test_loader, desc="Evaluating"):
            batch = {k: v.to(self.device) for k, v in batch.items()}

            if eval_carry is None:
                with torch.device(self.device):
                    eval_carry = self.model.initial_carry(batch)

            while True:
                eval_carry, _, _, preds, all_finish = self.model(
                    carry=eval_carry, batch=batch, return_keys=[]
                )
                if all_finish:
                    break
            
            predictions = preds["prediction"].squeeze()
            labels = batch["labels"].squeeze()[:, 0]

            all_preds.append(predictions.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)

        # Round and clip predictions for classification metrics
        pred_scores = np.round(all_preds).astype(int)
        pred_scores = np.clip(pred_scores, self.evaluator.config.min_score, self.evaluator.config.max_score)
        label_scores = all_labels.astype(int)

        # --- DEBUGGING --- 
        print("\n--- Evaluation Debug Info ---")
        print(f"Unique Predicted Scores: {np.unique(pred_scores)}")
        print(f"Min/Max Predicted: {pred_scores.min()}, {pred_scores.max()}")
        print(f"Unique Label Scores: {np.unique(label_scores)}")
        print(f"Min/Max Label: {label_scores.min()}, {label_scores.max()}")
        print("---------------------------\n")
        # --- END DEBUGGING ---

        # Use the evaluator's compute methods directly
        metrics = {
            "qwk": self.evaluator.compute_qwk(pred_scores, label_scores),
            "mse": self.evaluator.compute_mse(all_preds, all_labels),
            "rmse": self.evaluator.compute_rmse(all_preds, all_labels),
            "accuracy": self.evaluator.compute_accuracy(pred_scores, label_scores),
            "adjacent_accuracy": self.evaluator.compute_adjacent_accuracy(pred_scores, label_scores),
            "num_samples": len(pred_scores),
        }

        # Restore original weights if using EMA
        if original_state_dict is not None:
            self.model.load_state_dict(original_state_dict)

        self.model.train()
        return metrics

    def train(self, num_epochs: int):
        """Main training loop"""
        print(f"Starting training for {num_epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")

        eval_interval = self.config.get("eval_interval", 500)
        min_eval_interval = self.config.get("min_eval_interval", 0)

        start_time = time.time()

        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")

            # Train
            train_metrics = self.train_epoch()
            print(f"Train Loss: {train_metrics['loss']:.4f}")

            # --- Time estimation logic ---
            elapsed_time = time.time() - start_time
            avg_epoch_time = elapsed_time / (epoch + 1)
            remaining_epochs = num_epochs - (epoch + 1)
            remaining_time = remaining_epochs * avg_epoch_time
            finish_time = datetime.now() + timedelta(seconds=remaining_time)

            print(f"Average epoch time: {avg_epoch_time:.2f}s")
            print(f"Estimated remaining time: {str(timedelta(seconds=int(remaining_time)))}")
            print(f"Estimated finish time: {finish_time.strftime('%Y-%m-%d %H:%M:%S')}")
            # --- End of time estimation logic ---

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
                    self.save_checkpoint("best_model_m2_regression.pt")
                    print(f"Saved new best model (QWK: {self.best_qwk:.4f})")
                    self.early_stopping_counter = 0
                else:
                    self.early_stopping_counter += 1
                    print(
                        f"QWK did not improve. Early stopping counter: {self.early_stopping_counter}/{self.early_stopping_patience}"
                    )

                if self.early_stopping_counter >= self.early_stopping_patience:
                    print("Early stopping triggered.")
                    break

            # Save periodic checkpoint
            if (epoch + 1) % 5 == 0:
                self.save_checkpoint(f"checkpoint_epoch_{epoch+1}_m2_regression.pt")

    def save_checkpoint(self, filename: str):
        """Save model checkpoint and trigger evaluation"""
        checkpoint_dir = self.config.get(
            "checkpoint_path", "checkpoints/aes_m2_regression"
        )
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

        # --- Automatically run evaluation on the new checkpoint ---
        print(f"--- Running evaluation for checkpoint: {filename} ---")
        command = [
            sys.executable,
            "evaluate_aes.py",
            "--checkpoint",
            checkpoint_path,
            "--batch-size",
            str(self.config.get("batch_size", 16)),
        ]
        
        data_paths = self.config.get("data_path", [])
        if isinstance(data_paths, list):
            for path in data_paths:
                command.extend(["--data-path", path])
        else:
            command.extend(["--data-path", data_paths])

        try:
            result = subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
                encoding='utf-8'
            )
            print(result.stdout)
            if result.stderr:
                print("--- Evaluation Errors ---")
                print(result.stderr)
        except subprocess.CalledProcessError as e:
            print(f"--- Evaluation for {filename} failed! ---")
            print("--- STDOUT ---")
            print(e.stdout)
            print("--- STDERR ---")
            print(e.stderr)
        print(f"--- End of evaluation for checkpoint: {filename} ---")


def main():
    parser = argparse.ArgumentParser(
        description="Train Tiny Recursive Model for AES (Regression) on M2 Mac"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        nargs="+",
        required=True,
        help="Path(s) to processed dataset directory",
    )
    parser.add_argument(
        "--batch-size", type=int, default=8, help="Batch size (default: 8)"
    )
    parser.add_argument(
        "--epochs", type=int, default=100, help="Number of epochs (default: 100)"
    )
    parser.add_argument("--lr", type=float, default=3e-5, help="Learning rate")
    parser.add_argument(
        "--hidden_size", type=int, default=512, help="Model embedding dimension"
    )
    parser.add_argument(
        "--expansion", type=float, default=4, help="Hidden layer expansion"
    )
    parser.add_argument(
        "--num_heads", type=int, default=8, help="Number of attention heads"
    )
    parser.add_argument("--L_layers", type=int, default=2, help="Number of L layers")
    parser.add_argument(
        "--H_cycles", type=int, default=3, help="Number of high-level reasoning cycles"
    )
    parser.add_argument(
        "--L_cycles", type=int, default=4, help="Number of low-level reasoning cycles"
    )
    parser.add_argument(
        "--eval-interval", type=int, default=500, help="Evaluation interval (epochs)"
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default="checkpoints/aes_m2_regression",
        help="Checkpoint directory",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--use-wandb", action="store_true", help="Use Weights & Biases logging"
    )
    parser.add_argument("--project-name", type=str, default="TRM-AES")
    parser.add_argument("--run-name", type=str, default=None, help="Run name for wandb")
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=5,
        help="Patience for early stopping",
    )
    parser.add_argument(
        "--halt-threshold",
        type=float,
        default=0.5,
        help="ACT halt threshold for regression",
    )

    args = parser.parse_args()

    # Set seed
    set_seed(args.seed)

    # Get device
    device = get_device()

    # Create dataloaders
    print("Loading datasets...")
    train_dataset = PuzzleDataset(
        PuzzleDatasetConfig(
            seed=args.seed,
            dataset_paths=args.data_path,
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
            dataset_paths=args.data_path,
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
    print(f"  Number of puzzle identifiers: {metadata.num_puzzle_identifiers}")

    # Load dataset metadata for score info (use first path if multiple provided)
    first_data_path = (
        args.data_path[0] if isinstance(args.data_path, list) else args.data_path
    )
    with open(os.path.join(first_data_path, "train", "dataset.json"), "r") as f:
        dataset_info = json.load(f)

    min_score = dataset_info.get("min_score", 0)
    max_score = dataset_info.get("max_score", 12)
    score_bins = dataset_info.get("score_bins", 11)

    print(f"Score range: {min_score} - {max_score} (bins: {score_bins})")

    # Create model
    print("Creating model...")

    model_config = {
        "batch_size": args.batch_size,
        "seq_len": metadata.seq_len,
        "puzzle_emb_ndim": 128,
        "num_puzzle_identifiers": metadata.num_puzzle_identifiers,
        "vocab_size": metadata.vocab_size,
        "H_cycles": args.H_cycles,
        "L_cycles": args.L_cycles,
        "H_layers": 1,  # Not used in TRM
        "L_layers": args.L_layers,
        "hidden_size": args.hidden_size,
        "expansion": args.expansion,
        "num_heads": args.num_heads,
        "pos_encodings": "rope",
        "halt_max_steps": 10,
        "halt_exploration_prob": 0.1,
        "halt_threshold": args.halt_threshold,
        "forward_dtype": "float32",
    }

    model = TinyRecursiveReasoningModel_ACTV1_Regression(model_config)
    model = MSELossWrapper(model)

    # Create evaluator
    evaluator = AESEvaluator(
        AESEvaluatorConfig(
            name="aes",
            min_score=min_score,
            max_score=max_score,
            score_bins=score_bins,
        )
    )

    # Create config for trainer
    config = {
        "data_path": args.data_path,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "lr": args.lr,
        "lr_min_ratio": 0.1,
        "weight_decay": 0.1,
        "beta1": 0.9,
        "beta2": 0.999,
        "ema": True,
        "ema_rate": 0.999,
        "eval_interval": args.eval_interval,
        "min_eval_interval": 0,
        "checkpoint_path": args.checkpoint_path,
        "seed": args.seed,
        "use_wandb": args.use_wandb,
        "project_name": args.project_name,
        "run_name": args.run_name,
        "early_stopping_patience": args.early_stopping_patience,
        "entity_name": "andre-baerlocher-lehrmittelverlag-st-gallen",
        "model_config": model_config,
    }

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
