"""
Example Usage Script for TinyRecursiveModels-AES
Demonstrates the complete workflow from dataset preparation to evaluation
"""

import os
import sys
import json
from pathlib import Path


def print_section(title):
    """Print a section header"""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60 + "\n")


def check_environment():
    """Check if the environment is set up correctly"""
    print_section("1. Environment Check")

    try:
        import torch

        print(f"✓ PyTorch installed: {torch.__version__}")

        if torch.backends.mps.is_available():
            print("✓ MPS (Metal Performance Shaders) available")
            device = "mps"
        elif torch.cuda.is_available():
            print("✓ CUDA available")
            device = "cuda"
        else:
            print("⚠ Using CPU (slower)")
            device = "cpu"

        print(f"  Device: {device}")

    except ImportError:
        print("✗ PyTorch not installed")
        return False

    try:
        from datasets import load_dataset

        print("✓ HuggingFace datasets installed")
    except ImportError:
        print("✗ HuggingFace datasets not installed")
        return False

    try:
        from sklearn.metrics import cohen_kappa_score

        print("✓ scikit-learn installed")
    except ImportError:
        print("✗ scikit-learn not installed")
        return False

    return True


def prepare_dataset_example():
    """Example: Prepare ASAPPP dataset"""
    print_section("2. Dataset Preparation")

    print("To prepare the ASAPPP dataset, run:")
    print()
    print("  # For Prompts 1-2 (Score range: 2-12)")
    print("  python dataset/build_asappp_dataset.py \\")
    print("    --prompt-set 1-2 \\")
    print("    --output-dir data/asappp \\")
    print("    --num-aug 1")
    print()
    print("  # For Prompts 3-6 (Score range: 0-4)")
    print("  python dataset/build_asappp_dataset.py \\")
    print("    --prompt-set 3-6 \\")
    print("    --output-dir data/asappp \\")
    print("    --num-aug 1")
    print()
    print("  # For Prompt 7 (Score range: 2-24)")
    print("  python dataset/build_asappp_dataset.py \\")
    print("    --prompt-set 7 \\")
    print("    --output-dir data/asappp \\")
    print("    --num-aug 1")
    print()
    print("  # Or build all at once")
    print("  python dataset/build_asappp_dataset.py \\")
    print("    --prompt-set all \\")
    print("    --output-dir data/asappp \\")
    print("    --num-aug 1")
    print()

    # Check if any datasets exist
    datasets = [
        "data/asappp_prompts_1-2",
        "data/asappp_prompts_3-6",
        "data/asappp_prompts_7",
    ]

    existing = [d for d in datasets if os.path.exists(d)]
    if existing:
        print(f"Found {len(existing)} existing dataset(s):")
        for d in existing:
            print(f"  ✓ {d}")
    else:
        print("No datasets found. Run the commands above to prepare datasets.")


def training_example():
    """Example: Train a model"""
    print_section("3. Training a Model")

    print("Basic training command:")
    print()
    print("  python train_aes_m1.py \\")
    print("    --data-path data/asappp_prompts_1-2 \\")
    print("    --batch-size 16 \\")
    print("    --epochs 5000 \\")
    print("    --lr 3e-4 \\")
    print("    --eval-interval 250 \\")
    print("    --checkpoint-path checkpoints/prompts_1-2 \\")
    print("    --seed 42")
    print()

    print("With Weights & Biases logging:")
    print()
    print("  python train_aes_m1.py \\")
    print("    --data-path data/asappp_prompts_1-2 \\")
    print("    --batch-size 16 \\")
    print("    --epochs 5000 \\")
    print("    --use-wandb \\")
    print("    --project-name TinyRecursiveModels-AES \\")
    print("    --run-name experiment_1")
    print()

    print("Memory-constrained settings (if you encounter OOM):")
    print()
    print("  python train_aes_m1.py \\")
    print("    --data-path data/asappp_prompts_1-2 \\")
    print("    --batch-size 8 \\")
    print("    --d-model 96 \\")
    print("    --d-hidden 192 \\")
    print("    --n-heads 3 \\")
    print("    --h-cycles 1 \\")
    print("    --l-cycles 2")
    print()

    # Check for existing checkpoints
    checkpoint_dirs = [
        "checkpoints/prompts_1-2",
        "checkpoints/prompts_3-6",
        "checkpoints/prompts_7",
    ]

    for ckpt_dir in checkpoint_dirs:
        if os.path.exists(ckpt_dir):
            best_model = os.path.join(ckpt_dir, "best_model.pt")
            if os.path.exists(best_model):
                print(f"✓ Found trained model: {best_model}")

                # Try to read checkpoint info
                try:
                    import torch

                    checkpoint = torch.load(best_model, map_location="cpu")
                    print(f"  - Best QWK: {checkpoint.get('best_qwk', 'N/A'):.4f}")
                    print(f"  - Step: {checkpoint.get('step', 'N/A')}")
                except:
                    pass


def evaluation_example():
    """Example: Evaluate a trained model"""
    print_section("4. Evaluating a Model")

    print("Evaluate on test set:")
    print()
    print("  python evaluate_aes.py \\")
    print("    --checkpoint checkpoints/prompts_1-2/best_model.pt \\")
    print("    --data-path data/asappp_prompts_1-2 \\")
    print("    --split test")
    print()

    print("Evaluate and save predictions:")
    print()
    print("  python evaluate_aes.py \\")
    print("    --checkpoint checkpoints/prompts_1-2/best_model.pt \\")
    print("    --data-path data/asappp_prompts_1-2 \\")
    print("    --split test \\")
    print("    --save-predictions \\")
    print("    --output-file results/predictions.json")
    print()

    print("Evaluate on training set (check for overfitting):")
    print()
    print("  python evaluate_aes.py \\")
    print("    --checkpoint checkpoints/prompts_1-2/best_model.pt \\")
    print("    --data-path data/asappp_prompts_1-2 \\")
    print("    --split train")
    print()


def interpret_results():
    """Example: Interpreting evaluation results"""
    print_section("5. Interpreting Results")

    print("Quadratic Weighted Kappa (QWK) interpretation:")
    print("  < 0.40:     Poor agreement")
    print("  0.40-0.60:  Moderate agreement")
    print("  0.60-0.80:  Substantial agreement")
    print("  > 0.80:     Almost perfect agreement")
    print()

    print("Expected results after 5000 epochs:")
    print("  Prompts 1-2: QWK ~0.70-0.80")
    print("  Prompts 3-6: QWK ~0.65-0.75")
    print("  Prompt 7:    QWK ~0.65-0.75")
    print()

    print("Other important metrics:")
    print("  - RMSE: Lower is better (typical: 1-3 points)")
    print("  - Accuracy: Exact match rate (typical: 30-50%)")
    print("  - Adjacent Accuracy: ±1 score (typical: 80-95%)")
    print()


def hyperparameter_guide():
    """Guide to hyperparameter tuning"""
    print_section("6. Hyperparameter Tuning")

    print("Key hyperparameters and their effects:")
    print()
    print("Model Size:")
    print("  --d-model [64, 96, 128, 192]      # Embedding dimension")
    print("  --d-hidden [128, 192, 256, 384]   # Hidden layer dimension")
    print("  --n-heads [2, 3, 4, 6]            # Attention heads")
    print("  --n-layers [1, 2, 3]              # Encoder layers")
    print()

    print("Recursive Reasoning:")
    print("  --h-cycles [1, 2, 3]              # High-level cycles")
    print("  --l-cycles [2, 3, 4, 5]           # Low-level cycles")
    print("  More cycles = more reasoning but slower/more memory")
    print()

    print("Training:")
    print("  --batch-size [8, 12, 16, 24]      # Batch size")
    print("  --lr [1e-4, 3e-4, 5e-4, 1e-3]     # Learning rate")
    print("  --epochs [3000, 5000, 10000]      # Training duration")
    print()

    print("Recommended starting point (M1 with 16GB):")
    print("  --d-model 128 --d-hidden 256 --n-heads 4 --n-layers 2")
    print("  --h-cycles 2 --l-cycles 3 --batch-size 16 --lr 3e-4")
    print()


def troubleshooting():
    """Troubleshooting guide"""
    print_section("7. Troubleshooting")

    print("Problem: Out of Memory (OOM)")
    print("Solution:")
    print("  1. Reduce batch size: --batch-size 8")
    print("  2. Reduce model size: --d-model 96 --d-hidden 192")
    print("  3. Reduce cycles: --h-cycles 1 --l-cycles 2")
    print("  4. Close other applications")
    print()

    print("Problem: MPS not available")
    print("Solution:")
    print("  1. Check PyTorch version (need 2.0+)")
    print("  2. Verify macOS version (need 12.3+)")
    print("  3. Check M1 Mac (Intel Macs don't support MPS)")
    print("  4. Code will fall back to CPU automatically")
    print()

    print("Problem: Training is slow")
    print("Solution:")
    print("  1. Ensure Mac is plugged in (performance throttles on battery)")
    print("  2. Close unnecessary applications")
    print("  3. Reduce model size if needed")
    print("  4. Check Activity Monitor for resource usage")
    print()

    print("Problem: Poor QWK scores")
    print("Solution:")
    print("  1. Train for more epochs")
    print("  2. Try different learning rates")
    print("  3. Increase model capacity")
    print("  4. Check for overfitting (compare train vs test)")
    print("  5. Verify dataset was prepared correctly")
    print()


def quick_start():
    """Quick start guide"""
    print_section("Quick Start (All in One)")

    print("Run the automated quick start script:")
    print()
    print("  ./quickstart.sh")
    print()
    print("This will:")
    print("  1. Set up virtual environment")
    print("  2. Install dependencies")
    print("  3. Download and prepare dataset")
    print("  4. Start training")
    print()
    print("Or follow the manual steps:")
    print()
    print("  # 1. Setup")
    print("  python3 -m venv venv")
    print("  source venv/bin/activate")
    print("  pip install -r requirements.txt")
    print("  huggingface-cli login")
    print()
    print("  # 2. Prepare dataset")
    print("  python dataset/build_asappp_dataset.py \\")
    print("    --prompt-set 1-2 --output-dir data/asappp")
    print()
    print("  # 3. Train")
    print("  python train_aes_m1.py \\")
    print("    --data-path data/asappp_prompts_1-2 \\")
    print("    --batch-size 16 --epochs 5000")
    print()
    print("  # 4. Evaluate")
    print("  python evaluate_aes.py \\")
    print("    --checkpoint checkpoints/prompts_1-2/best_model.pt \\")
    print("    --data-path data/asappp_prompts_1-2")
    print()


def main():
    """Main function"""
    print("\n" + "#" * 60)
    print("#  TinyRecursiveModels - AES Example Usage")
    print("#  Automated Essay Scoring on MacBook Pro M1")
    print("#" * 60)

    quick_start()

    if check_environment():
        prepare_dataset_example()
        training_example()
        evaluation_example()
        interpret_results()
        hyperparameter_guide()
        troubleshooting()
    else:
        print("\n⚠ Environment not properly set up.")
        print("Please install requirements: pip install -r requirements.txt")

    print_section("Additional Resources")
    print("Documentation:")
    print("  - README_AES.md: Complete guide")
    print("  - CHANGES.md: Technical details of adaptations")
    print("  - Original paper: https://arxiv.org/abs/2510.04871")
    print()
    print("Datasets:")
    print("  - ASAPPP on HuggingFace: https://huggingface.co/datasets/llm-aes")
    print("  - Original Kaggle: https://www.kaggle.com/c/asap-aes")
    print()
    print("Support:")
    print("  - Open an issue on GitHub")
    print("  - Check README_AES.md for troubleshooting")
    print()


if __name__ == "__main__":
    main()
