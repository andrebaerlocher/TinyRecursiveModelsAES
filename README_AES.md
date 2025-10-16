# Tiny Recursive Models for Automated Essay Scoring (AES)

This document describes an advanced adaptation of the Tiny Recursive Models (TRM) architecture for Automated Essay Scoring (AES), optimized for both Apple Silicon (M1/M2/M3/M4) and high-performance NVIDIA CUDA systems (H100/H200).

## Overview

This project adapts the recursive reasoning approach from [Tiny Recursive Models](https://arxiv.org/abs/2510.04871) to the task of automated essay scoring. The model has been significantly enhanced from the original character-level adaptation to a modern, tokenizer-based regression model that leverages pre-trained embeddings.

### Key Features

1.  **Advanced Model**: A regression-based model that predicts a continuous score, not just a classification bin.
2.  **Tokenizer-based Input**: Uses a `bert-base-uncased` tokenizer to process text as meaningful word/sub-word tokens, allowing the model to understand much more content.
3.  **Pre-trained Embeddings**: Initializes the model with embeddings from BERT, giving it a massive head-start in understanding language.
4.  **Multi-Platform Support**: Includes dedicated training scripts for both Apple Silicon (`train_aes_m2_regression.py`) and multi-GPU NVIDIA systems (`train_aes_h200_regression.py`).
5.  **Configurable Datasets**: The dataset builder script is highly configurable, allowing you to combine prompt sets, limit training examples, and adjust sequence length.

## Requirements

- Python 3.9+
- PyTorch 2.0+
- For Apple: macOS with Apple Silicon (M1/M2/M3/M4)
- For NVIDIA: A CUDA-enabled environment and NVIDIA drivers.
- `pip install -r requirements.txt`
- `huggingface-cli login`

## 1. Dataset Preparation

The dataset builder script is now highly configurable.

### Analyzing Essay Lengths
Before building a dataset, you can analyze the token distribution of the official ASAPPP essays to make informed decisions about your sequence length. This command will download the datasets and print a report without saving any files.

```bash
python dataset/build_asappp_dataset.py --prompt-set all --analyze-lengths
```

### Building a Combined Dataset
Based on our analysis, a good strategy is to combine prompt sets 1-2 and 7, while excluding the very long essays from set 3-6. The following command builds a combined dataset with a 1024 token limit.

```bash
# This will create a dataset in `data/asappp_combined` from prompts 1-2 and 7.
python dataset/build_asappp_dataset.py \
  --prompt-set all \
  --output-dir data/asappp_combined \
  --max-tokens 1024
```

### Building a Smaller Dataset for Experiments
You can limit the number of training essays to create smaller datasets for experiments, such as plotting a learning curve.

```bash
# Creates a training set with only 1000 unique essays
python dataset/build_asappp_dataset.py \
  --prompt-set all \
  --output-dir data/asappp_combined_1k \
  --max-tokens 1024 \
  --limit-train-essays 1000
```

## 2. Training

We now have two specialized training scripts depending on your hardware.

### A) On Apple Silicon (M1/M2/M3/M4)

Use the `train_aes_m2_regression.py` script. This is optimized for the MPS backend.

```bash
python train_aes_m2_regression.py \
  --data-path data/asappp_combined \
  --eval-interval 5 \
  --use-wandb \
  --project-name "AES-M2-Training"
```
*   The script will automatically use the recommended default hyperparameters (e.g., `hidden_size=768`, `lr=1e-5`, `dropout=0.1`).
*   The batch size is small (`--batch-size 8`) to accommodate the memory constraints of typical Mac devices.

### B) On Multi-GPU NVIDIA System (H100/H200)

Use the `train_aes_h200_regression.py` script. This script uses Distributed Data Parallel (DDP) and Automatic Mixed Precision (AMP) for maximum performance.

You must launch it with `torchrun`, specifying the number of GPUs you want to use.

```bash
# For a 2-GPU system
torchrun --nproc_per_node=2 train_aes_h200_regression.py \
  --data-path data/asappp_combined \
  --eval-interval 5 \
  --use-wandb \
  --project-name "AES-H200-Training"
```
*   This script uses a much larger per-GPU batch size (`--batch-size 128`) to leverage the available VRAM.

## 3. Evaluation

You can evaluate any saved checkpoint using the `evaluate_aes.py` script.

```bash
python evaluate_aes.py \
  --checkpoint checkpoints/aes_h200_regression/best_model_h200_regression.pt \
  --data-path data/asappp_combined
```

## Model Architecture (Current Version)

The current model is a sophisticated, 37M parameter regression model.

1.  **Input**: Essays and their prompts are combined and tokenized using a `bert-base-uncased` tokenizer up to a sequence length of 1024 tokens.
2.  **Embeddings**: The model uses a large token embedding layer (`~30k vocab * 768 hidden_size`) initialized with pre-trained weights from BERT.
3.  **Recursive Reasoning**: The core TRM architecture with H-cycles and L-cycles processes the token sequence to produce a final hidden state. Dropout is used for regularization.
4.  **Output**: A single regression head predicts a continuous score based on the model's final state.
5.  **Loss Function**: The model is trained using Mean Squared Error (MSE) loss.

## Evaluation Metrics

The primary metric for this task is **QWK (Quadratic Weighted Kappa)**, which measures the agreement between the model's predictions and the human raters. We also track RMSE, accuracy, and other metrics.

