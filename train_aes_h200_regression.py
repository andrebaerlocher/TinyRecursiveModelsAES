"""
Training script for Automated Essay Scoring (AES) using Tiny Recursive Models (Regression)
Optimized for Multi-GPU CUDA systems (e.g., 2x H100/H200).

Usage:
  torchrun --nproc_per_node=2 train_aes_h200_regression.py --data-path <path> [other_args...] 
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
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast

import numpy as np
from tqdm import tqdm
import wandb

from puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig
from evaluators.aes_evaluator import AESEvaluator, AESEvaluatorConfig
from models.ema import EMAHelper
from models.recursive_reasoning.trm_regression import TinyRecursiveReasoningModel_ACTV1_Regression

# --- DDP Setup ---
def setup_ddp():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank, dist.get_world_size()

def cleanup_ddp():
    dist.destroy_process_group()

def is_main_process():
    return dist.get_rank() == 0

# -------------------

def get_device(local_rank):
    """Get the CUDA device for the current process."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. This script requires a CUDA-enabled environment.")
    print(f"Process {dist.get_rank()}: Using CUDA device {local_rank}")
    return torch.device(f"cuda:{local_rank}")

def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
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
    def __init__(self, model: nn.Module, train_loader: DataLoader, test_loader: DataLoader, evaluator: AESEvaluator, device: torch.device, config: dict, local_rank: int):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.evaluator = evaluator
        self.device = device
        self.config = config
        self.model_config = config["model_config"]
        self.local_rank = local_rank

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config.get("lr", 1e-5), weight_decay=config.get("weight_decay", 0.1), betas=(0.9, 0.999))
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=config.get("epochs", 100), eta_min=config.get("lr", 1e-5) * 0.1)
        self.scaler = GradScaler()

        self.use_ema = config.get("ema", True)
        if self.use_ema:
            self.ema_helper = EMAHelper(mu=config.get("ema_rate", 0.999))
            self.ema_helper.register(self.model.module) # Register the underlying model

        self.step = 0
        self.best_qwk = -1.0
        self.carry = None
        self.early_stopping_counter = 0

        if is_main_process() and config.get("use_wandb", False):
            wandb.init(project=config.get("project_name", "TRM-AES-H200"), name=config.get("run_name"), config=config)
            wandb.watch(self.model, log_freq=100)

    def train_epoch(self, epoch: int) -> dict:
        self.model.train()
        # The IterableDataset handles shuffling and distribution, no sampler.set_epoch needed
        total_loss = 0.0
        num_batches = 0

        progress_bar = tqdm(self.train_loader, desc=f"Training Epoch {epoch+1}", disable=not is_main_process())

        for _, batch, _ in progress_bar:
            batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}

            if self.carry is None:
                with torch.device(self.device):
                    self.carry = self.model.module.initial_carry(batch)

            with autocast(device_type='cuda', dtype=torch.bfloat16):
                self.carry, loss, _, _, _ = self.model(carry=self.carry, batch=batch)
                dist.all_reduce(loss, op=dist.ReduceOp.AVG)

            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            if self.use_ema:
                self.ema_helper.update(self.model.module)

            total_loss += loss.item()
            num_batches += 1
            self.step += 1

            if is_main_process():
                progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
                if self.config.get("use_wandb", False) and self.step % 10 == 0:
                    wandb.log({"train/loss": loss.item(), "train/step": self.step, "train/lr": self.optimizer.param_groups[0]['lr']})

        self.scheduler.step()
        avg_loss = total_loss / max(num_batches, 1)
        return {"loss": avg_loss}

    @torch.no_grad()
    def evaluate(self) -> dict:
        if not is_main_process():
            return {}

        model_to_eval = self.model.module
        original_state_dict = None
        if self.use_ema:
            original_state_dict = copy.deepcopy(model_to_eval.state_dict())
            self.ema_helper.ema(model_to_eval)

        model_to_eval.eval()
        all_preds, all_labels = [], []
        eval_carry = None

        for _, batch, _ in tqdm(self.test_loader, desc="Evaluating"):
            batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}
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
        if is_main_process():
            print(f"Starting training from epoch {start_epoch + 1} to {num_epochs} on {dist.get_world_size()} GPUs...")
            print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")

        for epoch in range(start_epoch, num_epochs):
            self.train_epoch(epoch)

            if (epoch + 1) % self.config.get("eval_interval", 5) == 0:
                eval_metrics = self.evaluate()
                dist.barrier() # Wait for main process to finish evaluation

                if is_main_process():
                    # Broadcast metrics to other processes if needed, for now just log and save on main
                    print("\nEvaluation Metrics:")
                    for k, v in eval_metrics.items():
                        print(f"  {k.upper()}: {v:.4f}")
                    
                    if self.config.get("use_wandb", False):
                        wandb.log({f"eval/{k}": v for k, v in eval_metrics.items() if k != "num_samples"}, step=self.step)

                    if eval_metrics["qwk"] > self.best_qwk:
                        self.best_qwk = eval_metrics["qwk"]
                        self.save_checkpoint("best_model_h200_regression.pt")
                        print(f"Saved new best model (QWK: {self.best_qwk:.4f})")
                        self.early_stopping_counter = 0
                    else:
                        self.early_stopping_counter += 1
                        print(f"QWK did not improve. Early stopping counter: {self.early_stopping_counter}/{self.config.get('early_stopping_patience', 5)}")

                    if self.early_stopping_counter >= self.config.get('early_stopping_patience', 5):
                        print("Early stopping triggered.")
                        # Need to signal other processes to stop
                        stop_tensor = torch.tensor([1], device=self.device)
                    else:
                        stop_tensor = torch.tensor([0], device=self.device)
                else:
                    stop_tensor = torch.tensor([0], device=self.device)
                
                dist.broadcast(stop_tensor, 0)
                if stop_tensor.item() == 1:
                    break # All processes break

            if is_main_process() and (epoch + 1) % 10 == 0:
                self.save_checkpoint(f"checkpoint_epoch_{epoch+1}_h200_regression.pt")

    def save_checkpoint(self, filename: str):
        if not is_main_process():
            return
        
        checkpoint_dir = self.config.get("checkpoint_path", "checkpoints/aes_h200_regression")
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, filename)

        checkpoint = {
            "model_state_dict": self.model.module.state_dict(),
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
    parser = argparse.ArgumentParser(description="Train TRM for AES on Multi-GPU CUDA system")
    parser.add_argument("--data-path", type=str, nargs='+', required=True, help="Path(s) to dataset directory")
    parser.add_argument("--batch-size", type=int, default=128, help="Per-GPU batch size")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--seq-len", type=int, default=1024, help="Model sequence length")
    parser.add_argument("--hidden_size", type=int, default=768, help="Model embedding dimension")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers. NOTE: Must be 0 for current IterableDataset.")
    parser.add_argument("--eval-interval", type=int, default=5, help="Evaluation interval in epochs")
    parser.add_argument("--early-stopping-patience", type=int, default=10, help="Patience for early stopping (in evaluations)")
    parser.add_argument("--expansion", type=float, default=4, help="Hidden layer expansion")
    parser.add_argument("--num_heads", type=int, default=12, help="Number of attention heads (must be compatible with hidden_size=768)")
    parser.add_argument("--L_layers", type=int, default=2, help="Number of L layers")
    parser.add_argument("--H_cycles", type=int, default=3, help="Number of high-level reasoning cycles")
    parser.add_argument("--L_cycles", type=int, default=4, help="Number of low-level reasoning cycles")
    parser.add_argument("--use-wandb", action="store_true", help="Use Weights & Biases logging")
    parser.add_argument("--project-name", type=str, default="TRM-AES-H200")
    parser.add_argument("--run-name", type=str, default=None, help="Run name for wandb")
    parser.add_argument("--resume-from-checkpoint", type=str, default=None, help="Path to a checkpoint to resume training from.")

    args = parser.parse_args()

    local_rank, world_size = setup_ddp()
    set_seed(42)
    device = get_device(local_rank)

    if is_main_process(): print("Loading datasets...")
    train_dataset = PuzzleDataset(PuzzleDatasetConfig(seed=42, dataset_paths=args.data_path, global_batch_size=args.batch_size * world_size, test_set_mode=False, epochs_per_iter=1, rank=local_rank, num_replicas=world_size), split="train")
    test_dataset = PuzzleDataset(PuzzleDatasetConfig(seed=42, dataset_paths=args.data_path, global_batch_size=args.batch_size * world_size, test_set_mode=True, epochs_per_iter=1, rank=local_rank, num_replicas=world_size), split="test")
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers)

    metadata = train_dataset.metadata
    if is_main_process():
        print(f"Dataset metadata: Vocab size: {metadata.vocab_size}, Seq len: {metadata.seq_len}")

    if is_main_process(): print("Creating model...")
    model_config = {
        "batch_size": args.batch_size, "seq_len": args.seq_len,
        "vocab_size": metadata.vocab_size,
        "H_cycles": args.H_cycles, "L_cycles": args.L_cycles, "H_layers": 1, "L_layers": args.L_layers,
        "hidden_size": args.hidden_size, "expansion": args.expansion, "num_heads": args.num_heads,
        "pos_encodings": "rope", "dropout": args.dropout, "halt_max_steps": 10,
        "halt_exploration_prob": 0.1, "halt_threshold": 0.5, "forward_dtype": "bfloat16",
    }
    base_model = TinyRecursiveReasoningModel_ACTV1_Regression(model_config)
    model = MSELossWrapper(base_model).to(device)
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=True) # find_unused_parameters can be helpful

    evaluator = None
    if is_main_process():
        with open(os.path.join(args.data_path[0], "train", "dataset.json"), "r") as f:
            dataset_info = json.load(f)
        evaluator = AESEvaluator(AESEvaluatorConfig(name="aes", min_score=dataset_info.get("min_score", 0), max_score=dataset_info.get("max_score", 12), score_bins=dataset_info.get("score_bins", 11)))

    config = vars(args)
    config["model_config"] = model_config

    trainer = AESTrainer(model=model, train_loader=train_loader, test_loader=test_loader, evaluator=evaluator, device=device, config=config, local_rank=local_rank)

    start_epoch = 0
    if args.resume_from_checkpoint:
        if not os.path.exists(args.resume_from_checkpoint):
            print(f"WARNING: Checkpoint file not found, starting from scratch: {args.resume_from_checkpoint}")
        else:
            print(f"Resuming training from checkpoint: {args.resume_from_checkpoint}")
            checkpoint = torch.load(args.resume_from_checkpoint, map_location=device)
            
            trainer.model.module.load_state_dict(checkpoint['model_state_dict'])
            trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            trainer.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            start_epoch = trainer.scheduler.last_epoch
            trainer.step = checkpoint.get('step', 0)
            trainer.best_qwk = checkpoint.get('best_qwk', -1.0)
            
            if trainer.use_ema and 'ema_state_dict' in checkpoint:
                trainer.ema_helper.load_state_dict(checkpoint['ema_state_dict'])
            
            print(f"Resuming from epoch {start_epoch + 1}")

    # Train
    trainer.train(start_epoch=start_epoch, num_epochs=args.epochs)
    
    cleanup_ddp()

if __name__ == "__main__":
    main()