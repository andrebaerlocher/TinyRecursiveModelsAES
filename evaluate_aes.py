"""
Evaluation script for trained AES models (Regression version)
"""

import os
import sys
import json
import argparse
import time
from typing import Dict, Any

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

from puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig
from torch.utils.data import DataLoader
from evaluators.aes_evaluator import AESEvaluator, AESEvaluatorConfig
from models.recursive_reasoning.trm_regression import (
    TinyRecursiveReasoningModel_ACTV1_Regression,
)
from train_aes_m2_regression import MSELossWrapper, get_device, set_seed


def load_model(checkpoint_path: str, device: torch.device) -> tuple:
    """Load model from checkpoint"""
    print(f"Loading checkpoint from {checkpoint_path}...")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint["config"]
    model_config = config["model_config"]

    # Create model
    model = TinyRecursiveReasoningModel_ACTV1_Regression(model_config)
    model = MSELossWrapper(model)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    print(f"✓ Model loaded successfully")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Training step: {checkpoint.get('step', 'N/A')}")
    print(f"  Best QWK: {checkpoint.get('best_qwk', 'N/A'):.4f}")

    # Get dataset info from config
    first_data_path = (
        config["data_path"][0]
        if isinstance(config["data_path"], list)
        else config["data_path"]
    )
    # Correct the path to dataset.json
    if not os.path.exists(os.path.join(first_data_path, "train", "dataset.json")):
        # If not in train, check root of data_path
        if os.path.exists(os.path.join(first_data_path, "dataset.json")):
            with open(os.path.join(first_data_path, "dataset.json"), "r") as f:
                dataset_info = json.load(f)
        else:
            raise FileNotFoundError(
                "dataset.json not found in train directory or data_path root."
            )
    else:
        with open(os.path.join(first_data_path, "train", "dataset.json"), "r") as f:
            dataset_info = json.load(f)

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

    all_preds = []
    all_labels = []
    all_essay_ids = []

    eval_carry = None
    start_time = time.time()
    with torch.no_grad():
        for set_name, batch, global_batch_size in tqdm(dataloader, desc="Evaluating"):
            batch = {k: v.to(device) for k, v in batch.items()}

            if eval_carry is None:
                with torch.device(device):
                    eval_carry = model.initial_carry(batch)

            while True:
                eval_carry, _, _, preds, all_finish = model(
                    carry=eval_carry, batch=batch, return_keys=[]
                )
                if all_finish:
                    break

            predictions = preds["prediction"].squeeze()
            labels = batch["labels"].squeeze()[:, -1]

            all_preds.append(predictions.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            if "puzzle_identifiers" in batch:
                all_essay_ids.append(batch["puzzle_identifiers"].cpu().numpy())

    end_time = time.time()
    duration = end_time - start_time

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    if all_essay_ids:
        all_essay_ids = np.concatenate(all_essay_ids)

    pred_scores = np.round(all_preds).astype(int)
    label_scores = all_labels.astype(int)

    num_samples = len(pred_scores)
    metrics = {
        "qwk": evaluator.compute_qwk(pred_scores, label_scores),
        "mse": evaluator.compute_mse(all_preds, all_labels),
        "rmse": evaluator.compute_rmse(all_preds, all_labels),
        "accuracy": evaluator.compute_accuracy(pred_scores, label_scores),
        "adjacent_accuracy": evaluator.compute_adjacent_accuracy(
            pred_scores, label_scores
        ),
        "num_samples": num_samples,
        "total_inference_time": duration,
        "time_per_sample": duration / num_samples if num_samples > 0 else 0,
    }

    if save_predictions and output_file:
        output_data = {
            "predictions": pred_scores.tolist(),
            "labels": label_scores.tolist(),
            "essay_ids": all_essay_ids.tolist() if all_essay_ids.any() else [],
            "metrics": metrics,
        }

        os.makedirs(
            os.path.dirname(output_file) if os.path.dirname(output_file) else ".",
            exist_ok=True,
        )
        with open(output_file, "w") as f:
            json.dump(output_data, f, indent=2)

        print(f"\n✓ Predictions saved to {output_file}")

    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate trained AES model (Regression)"
    )
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
        nargs="+",
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
        default="predictions_regression.json",
        help="Output file for predictions",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")

    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    print(f"Using device: {device}")

    model, config, dataset_info = load_model(args.checkpoint, device)

    print(f"\nLoading {args.split} dataset...")
    # Use data_path from the loaded config to ensure consistency
    data_path_from_config = config.get("data_path", args.data_path)
    dataset = PuzzleDataset(
        PuzzleDatasetConfig(
            seed=args.seed,
            dataset_paths=data_path_from_config,
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

    print(f"\nEvaluating on {args.split} set...")
    metrics = evaluate_model(
        model=model,
        dataloader=dataloader,
        evaluator=evaluator,
        device=device,
        save_predictions=args.save_predictions,
        output_file=args.output_file if args.save_predictions else None,
    )

    print("\n" + "=" * 50)
    print("Evaluation Results")
    print("=" * 50)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Dataset: {', '.join(data_path_from_config)}")
    print(f"Split: {args.split}")
    print(f"Score range: {min_score}-{max_score}")
    print(f"\nMetrics:")
    print(f"  QWK:                {metrics['qwk']:.4f}")
    print(f"  MSE:                {metrics['mse']:.4f}")
    print(f"  RMSE:               {metrics['rmse']:.4f}")
    print(f"  Accuracy:           {metrics['accuracy']:.4f}")
    print(f"  Adjacent Accuracy:  {metrics['adjacent_accuracy']:.4f}")
    print(f"  Samples:            {metrics['num_samples']}")
    print(f"\nInference Time:")
    print(f"  Total time:         {metrics['total_inference_time']:.2f} seconds")
    print(f"  Time per sample:    {metrics['time_per_sample'] * 1000:.2f} ms")
    print("=" * 50)

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
