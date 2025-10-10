# Tiny Recursive Models for Automated Essay Scoring (AES)

This is an adaptation of the Tiny Recursive Models (TRM) architecture for Automated Essay Scoring using the ASAPPP dataset. The code has been optimized to run on a MacBook Pro M1 with 16GB RAM.

## Overview

This project adapts the recursive reasoning approach from [Tiny Recursive Models](https://arxiv.org/abs/2510.04871) to the task of automated essay scoring. Instead of solving puzzles, the model learns to score essays through recursive refinement of its predictions.

### Key Adaptations

1. **Dataset**: Uses ASAPPP (Automated Student Assessment Prize Plus Project) dataset from Kaggle
2. **Task**: Changed from puzzle-solving to essay scoring (regression/classification)
3. **Platform**: Optimized for Apple M1 Silicon with MPS (Metal Performance Shaders) backend
4. **Memory**: Reduced model size and batch sizes to work within 16GB RAM constraints
5. **Evaluation**: Added AES-specific metrics (QWK, MSE, Adjacent Accuracy)

## Requirements

- macOS with Apple Silicon (M1/M2/M3)
- Python 3.9 or higher
- 16GB RAM (minimum)

## Installation

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip wheel setuptools

# Install PyTorch for M1 Mac
pip install torch torchvision torchaudio

# Install dependencies
pip install -r requirements.txt

# Login to HuggingFace (required for dataset access)
huggingface-cli login
```

## Dataset Preparation

The ASAPPP dataset consists of three different prompt sets with different scoring ranges:

### Prompts 1-2 (Score range: 2-12)
```bash
python dataset/build_asappp_dataset.py \
  --prompt-set 1-2 \
  --output-dir data/asappp \
  --num-aug 1
```

### Prompts 3-6 (Score range: 0-4)
```bash
python dataset/build_asappp_dataset.py \
  --prompt-set 3-6 \
  --output-dir data/asappp \
  --num-aug 1
```

### Prompt 7 (Score range: 2-24)
```bash
python dataset/build_asappp_dataset.py \
  --prompt-set 7 \
  --output-dir data/asappp \
  --num-aug 1
```

### Build All Datasets
```bash
python dataset/build_asappp_dataset.py \
  --prompt-set all \
  --output-dir data/asappp \
  --num-aug 1
```

**Note**: Building the datasets requires an active internet connection and HuggingFace authentication to download from the Hub.

## Training

### Quick Start (Prompts 1-2)

```bash
python train_aes_m1.py \
  --data-path data/asappp_prompts_1-2 \
  --batch-size 16 \
  --epochs 5000 \
  --lr 3e-4 \
  --d-model 128 \
  --d-hidden 256 \
  --n-heads 4 \
  --n-layers 2 \
  --h-cycles 2 \
  --l-cycles 3 \
  --eval-interval 250 \
  --checkpoint-path checkpoints/prompts_1-2 \
  --seed 42
```

### Training with Weights & Biases Logging

```bash
# First, login to wandb
wandb login

# Then train with logging
python train_aes_m1.py \
  --data-path data/asappp_prompts_1-2 \
  --batch-size 16 \
  --epochs 5000 \
  --use-wandb \
  --project-name TinyRecursiveModels-AES \
  --run-name prompts_1-2_experiment
```

### Training on Different Prompt Sets

**Prompts 3-6:**
```bash
python train_aes_m1.py \
  --data-path data/asappp_prompts_3-6 \
  --batch-size 16 \
  --epochs 5000 \
  --checkpoint-path checkpoints/prompts_3-6
```

**Prompt 7:**
```bash
python train_aes_m1.py \
  --data-path data/asappp_prompts_7 \
  --batch-size 16 \
  --epochs 5000 \
  --checkpoint-path checkpoints/prompts_7
```

### Memory-Constrained Training

If you encounter memory issues, reduce these parameters:

```bash
python train_aes_m1.py \
  --data-path data/asappp_prompts_1-2 \
  --batch-size 8 \        # Reduced batch size
  --d-model 96 \          # Smaller embedding dimension
  --d-hidden 192 \        # Smaller hidden dimension
  --n-heads 3 \           # Fewer attention heads
  --h-cycles 1 \          # Fewer high-level cycles
  --l-cycles 2            # Fewer low-level cycles
```

## Model Architecture

The model uses a recursive reasoning approach adapted for essay scoring:

1. **Input Encoding**: Essays are tokenized at the character level (max 512 chars)
2. **Recursive Reasoning**: 
   - H-cycles: High-level reasoning iterations (default: 2)
   - L-cycles: Low-level latent state updates per H-cycle (default: 3)
3. **Score Prediction**: Final answer state is decoded into a score bin
4. **Output**: Discretized score that is denormalized to the original scale

### Key Parameters

- `d_model`: Embedding dimension (default: 128)
- `d_hidden`: Hidden layer dimension (default: 256)
- `n_heads`: Number of attention heads (default: 4)
- `n_layers`: Number of encoder layers (default: 2)
- `h_cycles`: High-level reasoning cycles (default: 2)
- `l_cycles`: Low-level reasoning cycles (default: 3)

### Model Size

With default parameters:
- Total parameters: ~1-2M (much smaller than typical essay scoring models)
- Memory usage: ~4-6GB during training (fits comfortably in 16GB)
- Training time: ~2-4 hours for 5000 epochs on M1 Mac

## Evaluation Metrics

The model is evaluated using standard AES metrics:

1. **QWK (Quadratic Weighted Kappa)**: Primary metric, measures agreement with human raters
2. **MSE (Mean Squared Error)**: Measures prediction accuracy
3. **RMSE (Root Mean Squared Error)**: Square root of MSE
4. **Accuracy**: Exact match with ground truth scores
5. **Adjacent Accuracy**: Predictions within ±1 of ground truth (common in AES)

### Interpreting Results

- **QWK**: 
  - < 0.40: Poor agreement
  - 0.40-0.60: Moderate agreement
  - 0.60-0.80: Substantial agreement
  - > 0.80: Almost perfect agreement
  
- **Adjacent Accuracy**: Typically 80-95% for good AES systems

## Project Structure

```
TinyRecursiveModelsAES/
├── dataset/
│   ├── build_asappp_dataset.py    # ASAPPP dataset builder
│   ├── build_arc_dataset.py        # Original ARC dataset builder
│   └── common.py                   # Dataset utilities
├── evaluators/
│   └── aes_evaluator.py            # AES evaluation metrics
├── models/
│   ├── recursive_reasoning/        # TRM model implementations
│   ├── ema.py                      # Exponential Moving Average
│   └── layers.py                   # Model layers
├── config/
│   ├── cfg_aes.yaml               # AES training configuration
│   └── cfg_pretrain.yaml          # Original pretrain config
├── checkpoints/                    # Saved model checkpoints
├── data/                          # Processed datasets
├── train_aes_m1.py                # M1-optimized training script
├── pretrain.py                    # Original training script
├── puzzle_dataset.py              # Dataset loading utilities
├── requirements.txt               # Python dependencies
├── README.md                      # Original TRM README
└── README_AES.md                  # This file
```

## Tips for M1 Mac

### Optimizing Performance

1. **Use MPS backend**: The code automatically detects and uses MPS for M1 acceleration
2. **Monitor memory**: Use Activity Monitor to check memory usage
3. **Adjust batch size**: Reduce if you see memory pressure
4. **Close other apps**: Free up RAM for training
5. **Keep Mac plugged in**: Training is power-intensive

### Troubleshooting

**MPS not available:**
```python
# Check if MPS is available
import torch
print(torch.backends.mps.is_available())  # Should return True
print(torch.backends.mps.is_built())      # Should return True
```

**Out of memory:**
- Reduce `--batch-size`
- Reduce `--d-model` and `--d-hidden`
- Reduce `--h-cycles` and `--l-cycles`
- Close other applications

**Slow training:**
- Ensure Mac is plugged in (performance throttles on battery)
- Check Activity Monitor for CPU/Memory pressure
- Reduce `num_workers` in DataLoader (set to 0)

## Differences from Original TRM

| Aspect | Original TRM | AES Adaptation |
|--------|-------------|----------------|
| Task | Puzzle solving (ARC-AGI) | Essay scoring |
| Input | Grid patterns | Character-level text |
| Output | Grid transformations | Score bins |
| Vocab Size | ~10-20 | 256 (ASCII) |
| Sequence Length | Variable | Fixed (512) |
| Loss Function | Cross-entropy on pixels | Cross-entropy on score bins |
| Evaluation | Accuracy | QWK, MSE, Adjacent Acc |
| Augmentation | Dihedral transforms | Simple repetition |
| Hardware | Multi-GPU (CUDA) | Single M1 Mac (MPS) |
| Batch Size | 32-128 | 8-16 |
| Model Size | 7M params | 1-2M params |

## Expected Results

Based on the ASAPPP dataset benchmarks:

### Prompts 1-2
- Target QWK: 0.70-0.80
- Training time: ~2-3 hours
- Dataset size: ~1,800 training essays

### Prompts 3-6
- Target QWK: 0.65-0.75
- Training time: ~2-3 hours
- Dataset size: ~1,600 training essays

### Prompt 7
- Target QWK: 0.65-0.75
- Training time: ~2-3 hours
- Dataset size: ~1,600 training essays

*Note: These are approximate targets. Results may vary based on hyperparameters and random initialization.*

## Future Improvements

1. **Better Tokenization**: Use word-piece or BPE instead of character-level
2. **Larger Context**: Support longer essays (>512 characters)
3. **Trait Scoring**: Predict individual trait scores (content, organization, etc.)
4. **Cross-Prompt Learning**: Train on multiple prompts simultaneously
5. **Ensemble Methods**: Combine multiple models for better predictions
6. **Data Augmentation**: More sophisticated text augmentation techniques
7. **Pretrained Embeddings**: Initialize with pretrained character/word embeddings

## Citation

If you use this code, please cite the original TRM paper:

```bibtex
@misc{jolicoeurmartineau2025morerecursivereasoningtiny,
      title={Less is More: Recursive Reasoning with Tiny Networks}, 
      author={Alexia Jolicoeur-Martineau},
      year={2025},
      eprint={2510.04871},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2510.04871}, 
}
```

And the ASAPPP dataset:

```bibtex
@misc{asappp,
      title={ASAP Automated Essay Scoring},
      author={The Hewlett Foundation},
      year={2012},
      url={https://www.kaggle.com/c/asap-aes},
}
```

## License

This project inherits the license from the original Tiny Recursive Models repository.

## Acknowledgments

- Original TRM implementation by Alexia Jolicoeur-Martineau
- ASAPPP dataset provided by The Hewlett Foundation via Kaggle
- HuggingFace for hosting the datasets
- Apple for the excellent M1 hardware and MPS backend

## Support

For issues specific to:
- **AES adaptation**: Open an issue in this repository
- **Original TRM**: Refer to the [original TRM repository](https://github.com/AlexiaJM/TinyRecursiveModels)
- **ASAPPP dataset**: Check the [HuggingFace dataset page](https://huggingface.co/datasets/llm-aes)

## Contact

For questions or suggestions about this AES adaptation, please open an issue in the repository.