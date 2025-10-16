"""
Training script for a Baseline AES Regression Model.
This script is a simplified version of the main training script, with H and L cycles set to 1
to create a strong, non-recursive baseline for comparison.
"""

import os
import sys
import json
import argparse
import copy
import time
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
from models.recursive_reasoning.trm_regression import TinyRecursiveReasoningModel_ACTV1_Regression

# Note: This script is designed for single-device training (e.g., Apple Silicon)

def get_device():
    """Get the best available device."""
    if torch.backends.mps.is_available():
        print("Using MPS (Metal Performance Shaders) backend")
        return torch.device("mps")
    elif torch.cuda.is_available():
        print("Using CUDA backend")
        return torch.device("cuda")
    else:
        print("Using CPU")
        return torch.device("cpu")

def set_seed(seed: int):
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
        labels = batch["labels"].float()
        loss = self.loss_fn(outputs["prediction"].squeeze(), labels.squeeze()[:, 0])
        metrics = {}
        all_finish = carry.halted.all()
        return carry, loss, metrics, outputs, all_finish

class AESTrainer:
    def __init__(self, model: nn.Module, train_loader: DataLoader, test_loader: DataLoader, evaluator: AESEvaluator, device: torch.device, config: dict):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.evaluator = evaluator
        self.device = device
        self.config = config
        self.model_config = config["model_config"]

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config.get("lr", 1e-5), weight_decay=config.get("weight_decay", 0.1), betas=(0.9, 0.999))
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=config.get("epochs", 100), eta_min=config.get("lr", 1e-5) * 0.1)
        
        self.use_ema = config.get("ema", True)
        if self.use_ema:
            self.ema_helper = EMAHelper(mu=config.get("ema_rate", 0.999))
            self.ema_helper.register(self.model)

        self.step = 0
        self.best_qwk = -1.0
        self.carry = None
        self.early_stopping_counter = 0

        if config.get("use_wandb", False):
            wandb.init(project=config.get("project_name", "TRM-AES-Baseline"), name=config.get("run_name"), config=config)
            wandb.watch(self.model, log_freq=100)

    def train_epoch(self, epoch: int) -> dict:
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        progress_bar = tqdm(self.train_loader, desc=f"Training Epoch {epoch+1}")

        for _, batch, _ in progress_bar:
            batch = {k: v.to(self.device) for k, v in batch.items()}

            if self.carry is None:
                with torch.device(self.device):
                    self.carry = self.model.initial_carry(batch)

            self.carry, loss, _, _, _ = self.model(carry=self.carry, batch=batch)
            
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            if self.use_ema:
                self.ema_helper.update(self.model)

            total_loss += loss.item()
            num_batches += 1
            self.step += 1

            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
            if self.config.get("use_wandb", False) and self.step % 10 == 0:
                wandb.log({"train/loss": loss.item(), "train/step": self.step, "train/lr": self.optimizer.param_groups[0]['lr']})

        self.scheduler.step()
        return {"loss": total_loss / max(num_batches, 1)}

    @torch.no_grad()
    def evaluate(self) -> dict:
        model_to_eval = self.model
        original_state_dict = None
        if self.use_ema:
            original_state_dict = copy.deepcopy(model_to_eval.state_dict())
            self.ema_helper.ema(model_to_eval)

        model_to_eval.eval()
        all_preds, all_labels = [], []
        eval_carry = None

        for _, batch, _ in tqdm(self.test_loader, desc="Evaluating"):
            batch = {k: v.to(self.device) for k, v in batch.items()}
            if eval_carry is None:
                with torch.device(self.device):
                    eval_carry = model_to_eval.initial_carry(batch)
            
            while True:
                eval_carry, _, _, preds, all_finish = model_to_eval(carry=eval_carry, batch=batch)
                if all_finish:
                    break
            
            all_preds.append(preds["prediction"].squeeze().cpu().numpy())
            all_labels.append(batch["labels"].squeeze()[:, 0].cpu().numpy())

        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        pred_scores = np.clip(np.round(all_preds), self.evaluator.config.min_score, self.evaluator.config.max_score).astype(int)
        label_scores = all_labels.astype(int)

        metrics = {
            "qwk": self.evaluator.compute_qwk(pred_scores, label_scores),
            "mse": self.evaluator.compute_mse(all_preds, all_labels),
            "rmse": self.evaluator.compute_rmse(all_preds, all_labels),
            "accuracy": self.evaluator.compute_accuracy(pred_scores, label_scores),
            "adjacent_accuracy": self.evaluator.compute_adjacent_accuracy(pred_scores, label_scores),
            "num_samples": len(pred_scores),
        }

        if original_state_dict is not None:
            model_to_eval.load_state_dict(original_state_dict)

        return metrics

    def train(self, start_epoch: int, num_epochs: int):
        print(f"Starting training from epoch {start_epoch + 1} to {num_epochs}...")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")

        for epoch in range(start_epoch, num_epochs):
            self.train_epoch(epoch)

            if (epoch + 1) % self.config.get("eval_interval", 5) == 0:
                eval_metrics = self.evaluate()
                print("\nEvaluation Metrics:")
                for k, v in eval_metrics.items():
                    print(f"  {k.upper()}: {v:.4f}")
                
                if self.config.get("use_wandb", False):
                    wandb.log({f"eval/{k}": v for k, v in eval_metrics.items() if k != "num_samples"}, step=self.step)

                if eval_metrics["qwk"] > self.best_qwk:
                    self.best_qwk = eval_metrics["qwk"]
                    self.save_checkpoint("best_model_baseline_regression.pt")
                    print(f"Saved new best model (QWK: {self.best_qwk:.4f})")
                    self.early_stopping_counter = 0
                else:
                    self.early_stopping_counter += 1
                    print(f"QWK did not improve. Early stopping counter: {self.early_stopping_counter}/{self.config.get('early_stopping_patience', 5)}")

                if self.early_stopping_counter >= self.config.get('early_stopping_patience', 5):
                    print("Early stopping triggered.")
                    break

            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(f"checkpoint_epoch_{epoch+1}_baseline_regression.pt")

    def save_checkpoint(self, filename: str):
        checkpoint_dir = self.config.get("checkpoint_path", "checkpoints/aes_baseline_regression")
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, filename)

        checkpoint = {
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
    parser = argparse.ArgumentParser(description="Train a Baseline TRM for AES")
    parser.add_argument("--data-path", type=str, nargs='+', required=True, help="Path(s) to dataset directory")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--seq-len", type=int, default=1024, help="Model sequence length")
    parser.add_argument("--hidden_size", type=int, default=768, help="Model embedding dimension")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--eval-interval", type=int, default=5, help="Evaluation interval in epochs")
    parser.add_argument("--early-stopping-patience", type=int, default=10, help="Patience for early stopping")
    parser.add_argument("--use-wandb", action="store_true", help="Use Weights & Biases logging")
    parser.add_argument("--project-name", type=str, default="TRM-AES-Baseline")
    parser.add_argument("--run-name", type=str, default=None, help="Run name for wandb")
    parser.add_argument("--resume-from-checkpoint", type=str, default=None, help="Path to a checkpoint to resume training from.")

    args = parser.parse_args()

    set_seed(42)
    device = get_device()

    print("Loading datasets...")
    train_dataset = PuzzleDataset(PuzzleDatasetConfig(seed=42, dataset_paths=args.data_path, global_batch_size=args.batch_size, test_set_mode=False, epochs_per_iter=1, rank=0, num_replicas=1), split="train")
    test_dataset = PuzzleDataset(PuzzleDatasetConfig(seed=42, dataset_paths=args.data_path, global_batch_size=args.batch_size, test_set_mode=True, epochs_per_iter=1, rank=0, num_replicas=1), split="test")
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=0)

    metadata = train_dataset.metadata
    print(f"Dataset metadata: Vocab size: {metadata.vocab_size}, Seq len: {metadata.seq_len}")

    with open(os.path.join(args.data_path[0], "train", "dataset.json"), "r") as f:
        dataset_info = json.load(f)

    print("Creating model...")
    model_config = {
        "batch_size": args.batch_size, "seq_len": args.seq_len, "puzzle_emb_ndim": args.hidden_size,
        "num_puzzle_identifiers": metadata.num_puzzle_identifiers, "vocab_size": metadata.vocab_size,
        "H_cycles": 1, "L_cycles": 1, # Simplified non-recursive baseline
        "H_layers": 1, "L_layers": 2, # L_layers is still used
        "hidden_size": args.hidden_size, "expansion": 4, "num_heads": 12,
        "pos_encodings": "rope", "dropout": args.dropout, "halt_max_steps": 1, # No ACT
        "halt_exploration_prob": 0.0, "halt_threshold": 0.5, "forward_dtype": "bfloat16" if device.type == 'cuda' else 'float32',
    }
    base_model = TinyRecursiveReasoningModel_ACTV1_Regression(model_config)
    model = MSELossWrapper(base_model)

    evaluator = AESEvaluator(AESEvaluatorConfig(name="aes", min_score=dataset_info.get("min_score", 0), max_score=dataset_info.get("max_score", 12), score_bins=dataset_info.get("score_bins", 11)))

    config = vars(args)
    config["model_config"] = model_config

    trainer = AESTrainer(model=model, train_loader=train_loader, test_loader=test_loader, evaluator=evaluator, device=device, config=config)

    start_epoch = 0
    if args.resume_from_checkpoint:
        if not os.path.exists(args.resume_from_checkpoint):
            print(f"WARNING: Checkpoint file not found, starting from scratch: {args.resume_from_checkpoint}")
        else:
            print(f"Resuming training from checkpoint: {args.resume_from_checkpoint}")
            checkpoint = torch.load(args.resume_from_checkpoint, map_location=device)
            trainer.model.load_state_dict(checkpoint['model_state_dict'])
            trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            trainer.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = trainer.scheduler.last_epoch
            trainer.step = checkpoint.get('step', 0)
            trainer.best_qwk = checkpoint.get('best_qwk', -1.0)
            if trainer.use_ema and 'ema_state_dict' in checkpoint:
                trainer.ema_helper.load_state_dict(checkpoint['ema_state_dict'])
            print(f"Resuming from epoch {start_epoch + 1}")

    trainer.train(start_epoch=start_epoch, num_epochs=args.epochs)

if __name__ == "__main__":
    main()
