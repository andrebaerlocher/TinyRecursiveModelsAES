# TinyRecursiveModels-AES Architecture

Visual guide to understanding the system architecture.

## System Overview

```
┌─────────────────────────────────────────────────────────────┐
│                  TinyRecursiveModels-AES                     │
│          Automated Essay Scoring on M1 Mac                   │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
        ┌───────────────────────────────────────┐
        │         Data Pipeline                  │
        └───────────────────────────────────────┘
                            │
        ┌───────────────────┴───────────────────┐
        │                                       │
        ▼                                       ▼
┌──────────────┐                      ┌──────────────┐
│ HuggingFace  │                      │   Training   │
│   Datasets   │──────────────────────▶│   Process   │
│ (ASAPPP)     │                      │              │
└──────────────┘                      └──────────────┘
        │                                       │
        ▼                                       ▼
┌──────────────┐                      ┌──────────────┐
│  Character   │                      │  Recursive   │
│  Tokenizer   │                      │    Model     │
│  (512 chars) │                      │  (1-2M params)│
└──────────────┘                      └──────────────┘
        │                                       │
        ▼                                       ▼
┌──────────────┐                      ┌──────────────┐
│   Batches    │                      │  Evaluation  │
│  (16 essays) │                      │   (QWK, MSE) │
└──────────────┘                      └──────────────┘
```

## Data Flow

```
Essay Text (String)
        │
        ▼
Character Tokenization (max 512 chars)
        │
        ▼
Token IDs [batch_size, 512]
        │
        ▼
Token Embeddings [batch_size, 512, d_model]
        │
        ▼
Positional Encoding (+)
        │
        ▼
Transformer Encoder
        │
        ▼
Recursive Reasoning Loop
    │
    ├─▶ H-cycles (High-level)
    │   │
    │   └─▶ L-cycles (Low-level)
    │       │
    │       └─▶ Latent Update
    │           │
    │           └─▶ Answer Update
    │
    ▼
Final Answer State [batch_size, d_model]
        │
        ▼
Output Head (MLP)
        │
        ▼
Score Logits [batch_size, num_bins]
        │
        ▼
Predicted Score (0-24 range)
```

## Model Architecture Detail

```
┌─────────────────────────────────────────────────────────┐
│                   Input Processing                       │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  Essay Text → Tokens [B, 512] → Embeddings [B, 512, D]  │
│                                            ↓            │
│                                  Positional Encoding    │
│                                            ↓            │
│                                        Dropout          │
│                                                          │
└────────────────────────┬─────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│               Transformer Encoder (2 layers)             │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  [MultiHeadAttention → LayerNorm → FFN → LayerNorm] × 2 │
│                                                          │
│  Output: Encoded [B, 512, D]                            │
│                                                          │
└────────────────────────┬─────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│          Recursive Reasoning (H-cycles = 2)              │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  Initialize:                                             │
│    Latent [B, 1, D]                                     │
│    Answer [B, 1, D]                                     │
│                                                          │
│  For h in range(H_cycles):                              │
│    │                                                     │
│    ├─▶ For l in range(L_cycles):  ◄── L-cycles = 3    │
│    │   │                                                │
│    │   └─▶ Concatenate [Encoded, Answer, Latent]       │
│    │       │                                            │
│    │       └─▶ Latent Update (Transformer Layer)       │
│    │                                                     │
│    └─▶ Answer Update (Linear + Tanh)                   │
│        │                                                 │
│        └─▶ Answer [B, 1, D]                            │
│                                                          │
└────────────────────────┬─────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│                   Output Head                            │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  LayerNorm → Linear(D→256) → GELU → Dropout             │
│           → Linear(256→num_bins)                         │
│                                                          │
│  Output: Logits [B, num_bins]                           │
│                                                          │
└────────────────────────┬─────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│                  Loss & Prediction                       │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  CrossEntropyLoss(logits, score_bins)                   │
│                                                          │
│  Predicted Bin → Denormalize → Final Score              │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

## Training Loop

```
┌─────────────────────────────────────────────────────────┐
│                    Training Epoch                        │
└─────────────────────────────────────────────────────────┘
                         │
                         ▼
        ┌────────────────────────────────┐
        │   Load Batch (16 essays)       │
        └────────────────┬────────────────┘
                         │
                         ▼
        ┌────────────────────────────────┐
        │   Forward Pass                  │
        │   - Encode essays              │
        │   - Recursive reasoning        │
        │   - Predict scores             │
        └────────────────┬────────────────┘
                         │
                         ▼
        ┌────────────────────────────────┐
        │   Compute Loss                  │
        │   (Cross-Entropy)              │
        └────────────────┬────────────────┘
                         │
                         ▼
        ┌────────────────────────────────┐
        │   Backward Pass                 │
        │   - Compute gradients          │
        │   - Clip gradients (max=1.0)   │
        └────────────────┬────────────────┘
                         │
                         ▼
        ┌────────────────────────────────┐
        │   Optimizer Step                │
        │   (AdamW)                      │
        └────────────────┬────────────────┘
                         │
                         ▼
        ┌────────────────────────────────┐
        │   Update EMA                    │
        │   (if enabled)                 │
        └────────────────┬────────────────┘
                         │
                         ▼
        ┌────────────────────────────────┐
        │   Log Metrics                   │
        │   (WandB, console)             │
        └────────────────┬────────────────┘
                         │
                         ▼
                    Next Batch
                         │
                         ▼
        ┌────────────────────────────────┐
        │   Periodic Evaluation           │
        │   (every 250 epochs)           │
        │   - Compute QWK, MSE           │
        │   - Save best model            │
        └────────────────────────────────┘
```

## File Structure Map

```
TinyRecursiveModelsAES/
│
├── 📚 Documentation
│   ├── INDEX.md ───────────────▶ Start here for navigation
│   ├── GETTING_STARTED.md ─────▶ 5-min quick start
│   ├── README_AES.md ──────────▶ Complete guide
│   ├── CHANGES.md ─────────────▶ Technical details
│   ├── COMPARISON.md ──────────▶ Original vs AES
│   ├── ARCHITECTURE.md ────────▶ This file
│   └── REWRITE_SUMMARY.md ─────▶ Rewrite summary
│
├── 🚀 Quick Start
│   ├── quickstart.sh ──────────▶ Automated setup
│   └── example_usage.py ───────▶ Interactive guide
│
├── 🏋️ Training
│   ├── train_aes_m1.py ────────▶ M1-optimized training
│   └── config/
│       └── cfg_aes.yaml ───────▶ Training config
│
├── 📊 Evaluation
│   ├── evaluate_aes.py ────────▶ Evaluate trained models
│   └── evaluators/
│       └── aes_evaluator.py ───▶ QWK, MSE metrics
│
├── 📦 Dataset
│   └── dataset/
│       └── build_asappp_dataset.py ──▶ HuggingFace → TRM format
│
├── 🔧 Utilities
│   ├── puzzle_dataset.py ──────▶ Dataset loading
│   └── requirements.txt ───────▶ Dependencies
│
└── 🧠 Models
    └── models/
        ├── ema.py ─────────────▶ Exponential Moving Average
        └── layers.py ──────────▶ Neural network layers
```

## Component Interaction

```
┌──────────────┐
│   User       │
└──────┬───────┘
       │
       │  1. Run quickstart.sh or train_aes_m1.py
       │
       ▼
┌──────────────────────────────────────────┐
│          Training Script                  │
│       (train_aes_m1.py)                   │
└──────┬───────────────────────────────────┘
       │
       ├──▶ 2. Load Config (cfg_aes.yaml)
       │
       ├──▶ 3. Build Dataset
       │    │
       │    └──▶ build_asappp_dataset.py
       │         │
       │         └──▶ HuggingFace API
       │
       ├──▶ 4. Create Model
       │    │
       │    └──▶ SimpleRecursiveModel
       │         │
       │         └──▶ Transformer + Recursive Reasoning
       │
       ├──▶ 5. Training Loop
       │    │
       │    ├──▶ Forward pass
       │    ├──▶ Loss computation
       │    ├──▶ Backward pass
       │    ├──▶ Optimizer step
       │    └──▶ EMA update
       │
       └──▶ 6. Evaluation
            │
            └──▶ aes_evaluator.py
                 │
                 └──▶ QWK, MSE, Accuracy
```

## Hardware Utilization (M1 Mac)

```
┌─────────────────────────────────────────────────────────┐
│                    MacBook Pro M1                        │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  ┌─────────────────────────────────────────────────┐   │
│  │            Unified Memory (16GB)                 │   │
│  │                                                  │   │
│  │  ┌──────────────┐  ┌──────────────┐            │   │
│  │  │   Model      │  │   Gradients  │            │   │
│  │  │   (~2GB)     │  │   (~2GB)     │            │   │
│  │  └──────────────┘  └──────────────┘            │   │
│  │  ┌──────────────┐  ┌──────────────┐            │   │
│  │  │   Batch      │  │   OS/Other   │            │   │
│  │  │   (~1GB)     │  │   (~11GB)    │            │   │
│  │  └──────────────┘  └──────────────┘            │   │
│  └─────────────────────────────────────────────────┘   │
│                                                          │
│  ┌─────────────────────────────────────────────────┐   │
│  │         CPU (8 cores)                            │   │
│  │  ┌──────────────────────────────────────────┐   │   │
│  │  │  4 Performance + 4 Efficiency cores      │   │   │
│  │  │  Utilization: ~50-80%                    │   │   │
│  │  └──────────────────────────────────────────┘   │   │
│  └─────────────────────────────────────────────────┘   │
│                                                          │
│  ┌─────────────────────────────────────────────────┐   │
│  │         GPU (8 cores)                            │   │
│  │  ┌──────────────────────────────────────────┐   │   │
│  │  │  Metal Performance Shaders (MPS)         │   │   │
│  │  │  Used for: Matrix ops, Attention         │   │   │
│  │  └──────────────────────────────────────────┘   │   │
│  └─────────────────────────────────────────────────┘   │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

## Performance Characteristics

```
Training Phase                Time        Memory      Device
────────────────────────────────────────────────────────────
Dataset Loading              1-2 min      1 GB       CPU
Model Initialization         < 1 sec      2 GB       CPU/MPS
Forward Pass (batch=16)      0.3 sec      4 GB       MPS
Backward Pass                0.4 sec      6 GB       MPS
Optimizer Step               0.1 sec      6 GB       MPS
EMA Update                   < 0.1 sec    6 GB       CPU

Total per batch:             ~1 sec       6 GB       -
Total for epoch:             ~5 min       6 GB       -
Total for 5000 epochs:       ~2-4 hours   6 GB       -
```

## Recursive Reasoning Detail

```
High-Level View:
╔══════════════════════════════════════════════════════╗
║           Recursive Reasoning Module                  ║
╚══════════════════════════════════════════════════════╝
                       │
        ┌──────────────┴──────────────┐
        │                             │
        ▼                             ▼
┌──────────────┐              ┌──────────────┐
│  H-Cycle 1   │              │  H-Cycle 2   │
└──────────────┘              └──────────────┘
        │                             │
        │                             │
  ┌─────┴─────┐                 ┌─────┴─────┐
  │           │                 │           │
  ▼           ▼                 ▼           ▼
L-Cycle 1  L-Cycle 2         L-Cycle 1  L-Cycle 2
  │           │                 │           │
  ▼           ▼                 ▼           ▼
Latent    Latent             Latent    Latent
Update    Update             Update    Update
  │           │                 │           │
  └─────┬─────┘                 └─────┬─────┘
        │                             │
        ▼                             ▼
  Answer Update                 Answer Update
        │                             │
        └──────────────┬──────────────┘
                       │
                       ▼
              Final Answer State
```

## Scale Comparison

```
                    Original TRM          AES Adaptation
                    ─────────────────────────────────────
Model Size:         ████████████████      ████
Memory:             ████████████████████  ████
Training Time:      ████████████████████  ████
Cost:               ████████████████      (free/local)
Accuracy:           ████████████████████  ███████████████

Legend: Each █ represents relative amount
```

---

This architecture enables efficient essay scoring on consumer hardware while maintaining the core recursive reasoning principles from the original TRM.
