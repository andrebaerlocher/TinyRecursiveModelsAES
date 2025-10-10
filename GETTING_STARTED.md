# Getting Started with TinyRecursiveModels-AES

A quick 5-minute guide to get you up and running with Automated Essay Scoring on your M1 Mac.

## Prerequisites

- MacBook Pro with M1 chip (or M2/M3)
- 16GB RAM (minimum)
- macOS 12.3 or later
- Python 3.9 or later

## Option 1: Quick Start (Recommended)

Run the automated setup script:

```bash
./quickstart.sh
```

This will:
1. Create a virtual environment
2. Install all dependencies
3. Login to HuggingFace
4. Download and prepare the ASAPPP dataset
5. Start training

**That's it!** The script will guide you through the process interactively.

## Option 2: Manual Setup

### Step 1: Environment Setup (2 minutes)

```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip wheel setuptools
pip install -r requirements.txt

# Login to HuggingFace (required for dataset access)
huggingface-cli login
```

When prompted, enter your HuggingFace token. Get one at: https://huggingface.co/settings/tokens

### Step 2: Prepare Dataset (3-5 minutes)

Choose which prompt set to work with:

```bash
# Prompts 1-2 (recommended for beginners, score range: 2-12)
python dataset/build_asappp_dataset.py \
  --prompt-set 1-2 \
  --output-dir data/asappp \
  --num-aug 1
```

The dataset will be downloaded from HuggingFace and processed automatically.

### Step 3: Train Model (2-4 hours)

```bash
python train_aes_m1.py \
  --data-path data/asappp_prompts_1-2 \
  --batch-size 16 \
  --epochs 5000 \
  --eval-interval 250 \
  --checkpoint-path checkpoints/prompts_1-2
```

**Training Progress:**
- You'll see a progress bar showing training loss
- Evaluation happens every 250 epochs
- Best model is saved automatically
- Training takes ~2-4 hours on M1 Mac

**Expected Results:**
- QWK (Quadratic Weighted Kappa): ~0.70-0.80
- Adjacent Accuracy: ~85-95%

### Step 4: Evaluate Model (1 minute)

```bash
python evaluate_aes.py \
  --checkpoint checkpoints/prompts_1-2/best_model.pt \
  --data-path data/asappp_prompts_1-2 \
  --split test
```

You'll see metrics like:
- **QWK**: Primary metric (0.60-0.80 is good)
- **RMSE**: Root mean squared error
- **Accuracy**: Exact score match
- **Adjacent Accuracy**: Within ±1 point

## What's Next?

### Try Different Prompt Sets

**Prompts 3-6** (shorter essays, score range: 0-4):
```bash
python dataset/build_asappp_dataset.py --prompt-set 3-6 --output-dir data/asappp
python train_aes_m1.py --data-path data/asappp_prompts_3-6 --batch-size 16 --epochs 5000
```

**Prompt 7** (longer essays, score range: 2-24):
```bash
python dataset/build_asappp_dataset.py --prompt-set 7 --output-dir data/asappp
python train_aes_m1.py --data-path data/asappp_prompts_7 --batch-size 16 --epochs 5000
```

### Enable Logging with Weights & Biases

Track experiments and visualize training:

```bash
# Login to wandb
wandb login

# Train with logging
python train_aes_m1.py \
  --data-path data/asappp_prompts_1-2 \
  --batch-size 16 \
  --epochs 5000 \
  --use-wandb \
  --project-name TinyRecursiveModels-AES \
  --run-name my_first_experiment
```

### Tune Hyperparameters

Adjust model architecture for better results:

```bash
python train_aes_m1.py \
  --data-path data/asappp_prompts_1-2 \
  --batch-size 16 \
  --epochs 5000 \
  --d-model 192 \        # Larger embedding (default: 128)
  --d-hidden 384 \       # Larger hidden layer (default: 256)
  --n-heads 6 \          # More attention heads (default: 4)
  --h-cycles 3 \         # More reasoning cycles (default: 2)
  --l-cycles 4           # More latent updates (default: 3)
```

**Note:** Larger models need more memory. If you get OOM errors, reduce these values.

## Common Issues

### "MPS not available"
- Check PyTorch version: `python -c "import torch; print(torch.__version__)"`
- Need PyTorch 2.0+ for MPS support
- Code will automatically fall back to CPU

### Out of Memory
Reduce memory usage:
```bash
python train_aes_m1.py \
  --data-path data/asappp_prompts_1-2 \
  --batch-size 8 \       # Smaller batch
  --d-model 96 \         # Smaller model
  --d-hidden 192 \
  --n-heads 3 \
  --h-cycles 1 \
  --l-cycles 2
```

### Slow Training
- Ensure Mac is plugged in (performance throttles on battery)
- Close unnecessary applications
- Check Activity Monitor for memory pressure

### HuggingFace Login Issues
```bash
# Make sure you're logged in
huggingface-cli login

# Or set token as environment variable
export HF_TOKEN="your_token_here"
```

## Example Output

**During Training:**
```
Epoch 250/5000
Train Loss: 0.8234
Evaluating...
Evaluation Metrics:
  QWK: 0.7234
  MSE: 2.1234
  RMSE: 1.4574
  Accuracy: 0.4523
  Adjacent Accuracy: 0.8945
Saved new best model (QWK: 0.7234)
```

**After Evaluation:**
```
Evaluation Results
==================================================
Dataset: data/asappp_prompts_1-2
Split: test
Score range: 2-12

Metrics:
  QWK:                0.7456
  MSE:                1.8923
  RMSE:               1.3756
  Accuracy:           0.4823
  Adjacent Accuracy:  0.9123
  Samples:            384
==================================================

Interpretation:
  QWK 0.60-0.80: Substantial agreement
```

## Understanding the Metrics

- **QWK (Quadratic Weighted Kappa)**: Measures agreement with human raters
  - < 0.40: Poor
  - 0.40-0.60: Moderate
  - 0.60-0.80: Substantial ← Target
  - > 0.80: Almost perfect
  
- **RMSE**: Average error in score points (lower is better)
  - Good: < 2.0 points
  
- **Adjacent Accuracy**: Predictions within ±1 of true score
  - Good: > 85%

## Quick Reference Commands

```bash
# See all options for training
python train_aes_m1.py --help

# See all options for evaluation
python evaluate_aes.py --help

# See all options for dataset building
python dataset/build_asappp_dataset.py --help

# Run example usage guide
python example_usage.py
```

## File Locations

- **Datasets**: `data/asappp_prompts_*/`
- **Checkpoints**: `checkpoints/prompts_*/best_model.pt`
- **Logs**: Printed to console (or use `--use-wandb`)
- **Predictions**: Specify with `--output-file` in evaluate_aes.py

## Need More Help?

- **Full Documentation**: See [README_AES.md](README_AES.md)
- **Technical Details**: See [CHANGES.md](CHANGES.md)
- **Example Usage**: Run `python example_usage.py`
- **Troubleshooting**: Check README_AES.md section "Tips for M1 Mac"
- **Original Paper**: https://arxiv.org/abs/2510.04871

## Success Checklist

- [ ] Virtual environment created and activated
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] HuggingFace login completed (`huggingface-cli login`)
- [ ] Dataset prepared (one of prompts 1-2, 3-6, or 7)
- [ ] Training started and showing progress
- [ ] Model evaluated with QWK > 0.60

**Congratulations!** You've successfully trained a Tiny Recursive Model for Automated Essay Scoring! 🎉

---

**Time Investment:**
- Setup: 5-10 minutes
- Training: 2-4 hours
- Evaluation: 1-2 minutes

**Hardware Used:**
- 4-6GB RAM during training
- ~50-80% CPU utilization
- ~5-10GB disk space for datasets

**Next Steps:**
- Experiment with different prompt sets
- Tune hyperparameters for better results
- Compare multiple models
- Read the full documentation in README_AES.md