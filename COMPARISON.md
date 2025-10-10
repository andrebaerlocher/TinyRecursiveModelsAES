# Comparison: Original TRM vs AES Adaptation

Quick reference guide comparing the original Tiny Recursive Models implementation with the AES adaptation.

## At a Glance

| Feature | Original TRM | AES Adaptation |
|---------|-------------|----------------|
| **Task** | Puzzle solving (ARC-AGI) | Essay scoring (ASAPPP) |
| **Platform** | Multi-GPU (CUDA) | Single M1 Mac (MPS) |
| **Model Size** | 7M parameters | 1-2M parameters |
| **Memory** | 24GB+ | 4-6GB |
| **Training Time** | 2-3 days (4x H100) | 2-4 hours (M1) |
| **Batch Size** | 32-128 | 8-16 |
| **Primary Metric** | Accuracy | QWK |

## Detailed Comparison

### Hardware & Environment

| Aspect | Original TRM | AES Adaptation |
|--------|-------------|----------------|
| Hardware | NVIDIA H100/A100 GPUs | Apple M1/M2/M3 |
| Backend | CUDA 12.6 | MPS (Metal) / CPU |
| Memory | 40-80GB GPU RAM | 16GB unified RAM |
| Distribution | Multi-GPU (4+) | Single device |
| OS | Linux preferred | macOS required |
| Cost | ~$2-4/hour cloud | Free (local Mac) |

### Installation

| Step | Original TRM | AES Adaptation |
|------|-------------|----------------|
| Python | 3.10+ | 3.9+ |
| PyTorch | CUDA-enabled | Standard (MPS support) |
| Special packages | triton, CUDA tools | HuggingFace datasets |
| Setup time | 30-60 min | 5-10 min |
| Quick start | Manual | `./quickstart.sh` |

### Dataset

| Property | Original TRM | AES Adaptation |
|----------|-------------|----------------|
| **Name** | ARC-AGI | ASAPPP |
| **Source** | Kaggle | HuggingFace |
| **Size** | ~400 puzzles | ~1,600-1,800 essays |
| **Input Format** | 2D grids (10×10 to 30×30) | Text (up to 512 chars) |
| **Vocabulary** | 10-20 colors | 256 ASCII characters |
| **Output** | Grid transformation | Score (0-24 range) |
| **Augmentation** | 8 dihedral transforms | Simple repetition |
| **Splits** | Train/Eval/Concept | Train/Test |

### Model Architecture

| Component | Original TRM | AES Adaptation |
|-----------|-------------|----------------|
| **Embedding dim** | 256 | 128 |
| **Hidden dim** | 512 | 256 |
| **Attention heads** | 8 | 4 |
| **Encoder layers** | 2 | 2 |
| **H-cycles** | 3 | 2 |
| **L-cycles** | 4 | 3 |
| **Dropout** | 0.1 | 0.1 |
| **Pos encoding** | Learned | Learned |
| **Total params** | ~7M | ~1-2M |

### Training Configuration

| Parameter | Original TRM | AES Adaptation |
|-----------|-------------|----------------|
| **Global batch size** | 32-128 | 16 |
| **Local batch size** | 8-32 (per GPU) | 16 (single device) |
| **Learning rate** | 1e-4 | 3e-4 |
| **LR schedule** | Cosine with warmup | Cosine with warmup |
| **Warmup steps** | 5,000 | 1,000 |
| **Weight decay** | 1.0 | 0.1 |
| **Optimizer** | AdamATan2 | AdamW |
| **EMA** | Yes (0.999) | Yes (0.999) |
| **Gradient clip** | None | 1.0 |
| **Epochs** | 50,000+ | 5,000-10,000 |

### Evaluation

| Metric | Original TRM | AES Adaptation |
|--------|-------------|----------------|
| **Primary** | Accuracy (exact match) | QWK (agreement) |
| **Secondary** | Per-pixel accuracy | MSE, RMSE |
| **Additional** | Pass@K | Accuracy, Adjacent Acc |
| **Interval** | Every 5,000 epochs | Every 250-500 epochs |
| **Target** | 45% (ARC-AGI-1) | 0.70-0.80 QWK |
| **Test time** | Hours | Minutes |

### Performance & Resources

| Resource | Original TRM | AES Adaptation |
|----------|-------------|----------------|
| **Training time** | 2-3 days | 2-4 hours |
| **Training cost** | $200-400 (cloud) | Free (local) |
| **Peak memory** | 40GB+ | 6GB |
| **Disk space** | 10GB+ | 5GB |
| **Inference time** | Seconds per puzzle | <1s per essay |
| **Energy** | High | Moderate |

### Usage & Commands

#### Dataset Preparation

**Original TRM:**
```bash
python -m dataset.build_arc_dataset \
  --input-file-prefix kaggle/combined/arc-agi \
  --output-dir data/arc1concept-aug-1000 \
  --subsets training evaluation concept \
  --test-set-name evaluation
```

**AES Adaptation:**
```bash
python dataset/build_asappp_dataset.py \
  --prompt-set 1-2 \
  --output-dir data/asappp \
  --num-aug 1
```

#### Training

**Original TRM:**
```bash
torchrun --nproc-per-node 4 \
  --rdzv_backend=c10d \
  --rdzv_endpoint=localhost:0 \
  pretrain.py \
  arch=trm \
  data_paths="[data/arc1concept-aug-1000]" \
  arch.L_layers=2 \
  arch.H_cycles=3 arch.L_cycles=4 \
  +run_name=my_experiment \
  ema=True
```

**AES Adaptation:**
```bash
python train_aes_m1.py \
  --data-path data/asappp_prompts_1-2 \
  --batch-size 16 \
  --epochs 5000 \
  --d-model 128 \
  --h-cycles 2 --l-cycles 3
```

#### Evaluation

**Original TRM:**
- Integrated in training loop
- Reports accuracy on eval set
- Generates puzzle solutions

**AES Adaptation:**
```bash
python evaluate_aes.py \
  --checkpoint checkpoints/best_model.pt \
  --data-path data/asappp_prompts_1-2 \
  --split test
```

### Code Structure

#### Original TRM Files
```
pretrain.py                    # Main training script
models/recursive_reasoning/    # TRM implementation
dataset/build_arc_dataset.py   # ARC dataset builder
evaluators/                    # Task evaluators
config/cfg_pretrain.yaml       # Training config
```

#### AES Adaptation Files
```
train_aes_m1.py               # M1-optimized training
evaluate_aes.py               # Evaluation script
dataset/build_asappp_dataset.py  # ASAPPP builder
evaluators/aes_evaluator.py   # AES metrics
config/cfg_aes.yaml           # AES config
quickstart.sh                 # One-click setup
```

### Workflows

#### Original TRM Workflow
1. Download ARC-AGI dataset from Kaggle
2. Process with build_arc_dataset.py
3. Configure training with Hydra
4. Launch multi-GPU training with torchrun
5. Monitor accuracy metrics
6. Submit predictions to competition

#### AES Adaptation Workflow
1. Login to HuggingFace
2. Run quickstart.sh OR:
   - Build dataset from HuggingFace
   - Train with train_aes_m1.py
   - Evaluate with evaluate_aes.py
3. Monitor QWK metrics
4. Tune hyperparameters

### Dependencies

#### Original TRM
- torch (CUDA-enabled)
- triton
- numba (CUDA support)
- hydra-core
- omegaconf
- wandb
- adam-atan2

#### AES Adaptation
- torch (standard)
- datasets (HuggingFace)
- scikit-learn
- pandas
- transformers
- wandb
- pydantic 2.0+

### Advantages & Trade-offs

| Aspect | Original TRM | AES Adaptation |
|--------|-------------|----------------|
| **Strengths** | - State-of-art on ARC<br>- Large scale<br>- Proven results | - Runs locally<br>- Fast training<br>- Easy setup<br>- Low cost |
| **Limitations** | - Expensive GPUs needed<br>- Long training time<br>- Complex setup | - Smaller model<br>- Single task<br>- M1 Mac required |
| **Best for** | Research, competitions | Education, prototyping |

### Expected Results

#### Original TRM
- **ARC-AGI-1**: 45% accuracy
- **ARC-AGI-2**: 8% accuracy
- **Sudoku**: High accuracy
- **Maze**: High accuracy

#### AES Adaptation
- **Prompts 1-2**: QWK 0.70-0.80
- **Prompts 3-6**: QWK 0.65-0.75
- **Prompt 7**: QWK 0.65-0.75
- **Adjacent Acc**: 85-95%

### When to Use Each

#### Use Original TRM if:
- Working on ARC-AGI competition
- Have access to multiple GPUs
- Need maximum performance
- Budget for cloud compute
- Researching recursive reasoning

#### Use AES Adaptation if:
- Have M1/M2/M3 Mac
- Learning about recursive reasoning
- Need quick prototypes
- Limited to local compute
- Working on essay scoring
- Teaching/educational purposes

### Migration Guide

#### From Original TRM to AES

1. **Install M1-compatible dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Switch dataset**:
   ```bash
   # Original: ARC-AGI
   python -m dataset.build_arc_dataset ...
   
   # AES: ASAPPP
   python dataset/build_asappp_dataset.py ...
   ```

3. **Use M1 training script**:
   ```bash
   # Original: Multi-GPU
   torchrun --nproc-per-node 4 pretrain.py ...
   
   # AES: Single M1
   python train_aes_m1.py ...
   ```

4. **Adjust expectations**:
   - Different metrics (QWK vs accuracy)
   - Faster training (hours vs days)
   - Different output format

#### From AES to Original TRM

1. **Setup CUDA environment**
2. **Use original pretrain.py**
3. **Prepare ARC-AGI dataset**
4. **Configure for multi-GPU**
5. **Adjust hyperparameters for scale**

### Compatibility Matrix

| Feature | Original TRM | AES Adaptation |
|---------|-------------|----------------|
| M1 Mac | ❌ No | ✅ Yes |
| Intel Mac | ✅ CPU only | ✅ CPU only |
| Linux + NVIDIA | ✅ Yes | ⚠️ Works but not optimized |
| Windows + NVIDIA | ✅ Yes | ⚠️ Works but not optimized |
| Multi-GPU | ✅ Yes | ❌ No |
| Cloud (AWS/GCP) | ✅ Yes | ✅ Yes (but unnecessary) |

### Support & Documentation

#### Original TRM
- Paper: https://arxiv.org/abs/2510.04871
- Repository: https://github.com/AlexiaJM/TinyRecursiveModels
- Issues: GitHub issues

#### AES Adaptation
- Documentation: README_AES.md, GETTING_STARTED.md
- Technical details: CHANGES.md
- Quick help: `python example_usage.py`
- Issues: GitHub issues

---

## Quick Decision Guide

**Choose Original TRM if you:**
- Have NVIDIA GPUs available
- Need state-of-art ARC-AGI results
- Are submitting to competition
- Have time and budget

**Choose AES Adaptation if you:**
- Have an M1/M2/M3 Mac
- Want to learn recursive reasoning
- Need quick experimentation
- Work on essay scoring
- Prefer local development

Both versions share the same core recursive reasoning principles but are optimized for different use cases and hardware constraints.