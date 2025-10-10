# Documentation Index

Welcome to the TinyRecursiveModels-AES repository! This index helps you find the right documentation for your needs.

## 🚀 Start Here

### New Users
1. **[GETTING_STARTED.md](GETTING_STARTED.md)** - 5-minute quick start guide
2. **[README_AES.md](README_AES.md)** - Complete AES documentation
3. Run `./quickstart.sh` - Automated setup and training

### Experienced Users
1. **[COMPARISON.md](COMPARISON.md)** - Compare Original TRM vs AES
2. **[CHANGES.md](CHANGES.md)** - Technical details of adaptations
3. **[README.md](README.md)** - Original TRM documentation

## 📚 Documentation Files

### Essential Guides

| File | Purpose | Read Time | Audience |
|------|---------|-----------|----------|
| **[GETTING_STARTED.md](GETTING_STARTED.md)** | Quick setup and first training run | 5 min | Beginners |
| **[README_AES.md](README_AES.md)** | Complete guide to AES adaptation | 15 min | All users |
| **[README.md](README.md)** | Original TRM documentation | 10 min | TRM users |

### Reference Documents

| File | Purpose | Read Time | Audience |
|------|---------|-----------|----------|
| **[COMPARISON.md](COMPARISON.md)** | Side-by-side comparison | 10 min | Decision makers |
| **[CHANGES.md](CHANGES.md)** | Technical implementation details | 15 min | Developers |
| **[instructions.md](instructions.md)** | Original adaptation requirements | 3 min | Context |
| **INDEX.md** | This file - navigation hub | 2 min | Everyone |

### Interactive Tools

| File | Purpose | Usage |
|------|---------|-------|
| **[quickstart.sh](quickstart.sh)** | Automated setup and training | `./quickstart.sh` |
| **[example_usage.py](example_usage.py)** | Usage examples and guides | `python example_usage.py` |

## 🎯 Find What You Need

### "I want to..."

#### ...get started quickly
→ Run `./quickstart.sh` or read [GETTING_STARTED.md](GETTING_STARTED.md)

#### ...understand the project
→ Read [README_AES.md](README_AES.md) sections 1-3

#### ...set up my environment
→ [GETTING_STARTED.md](GETTING_STARTED.md) - Option 2: Manual Setup

#### ...prepare the dataset
→ [README_AES.md](README_AES.md) - Dataset Preparation section
→ Or see `dataset/build_asappp_dataset.py --help`

#### ...train a model
→ [GETTING_STARTED.md](GETTING_STARTED.md) - Step 3
→ Or [README_AES.md](README_AES.md) - Training section

#### ...evaluate a model
→ [GETTING_STARTED.md](GETTING_STARTED.md) - Step 4
→ Or run `python evaluate_aes.py --help`

#### ...tune hyperparameters
→ [README_AES.md](README_AES.md) - Model Architecture section
→ [GETTING_STARTED.md](GETTING_STARTED.md) - Tune Hyperparameters

#### ...understand the metrics
→ [README_AES.md](README_AES.md) - Evaluation Metrics section
→ [GETTING_STARTED.md](GETTING_STARTED.md) - Understanding the Metrics

#### ...troubleshoot issues
→ [README_AES.md](README_AES.md) - Tips for M1 Mac
→ [GETTING_STARTED.md](GETTING_STARTED.md) - Common Issues
→ Run `python example_usage.py` - section 7

#### ...compare with original TRM
→ [COMPARISON.md](COMPARISON.md) - Complete comparison
→ [CHANGES.md](CHANGES.md) - Technical differences

#### ...understand what changed
→ [CHANGES.md](CHANGES.md) - Comprehensive change log
→ [COMPARISON.md](COMPARISON.md) - Quick reference

#### ...see code examples
→ Run `python example_usage.py`
→ [README_AES.md](README_AES.md) - Training section

#### ...migrate from original TRM
→ [COMPARISON.md](COMPARISON.md) - Migration Guide
→ [CHANGES.md](CHANGES.md) - Architecture Adaptations

## 📁 Code Files

### Training & Evaluation

| File | Purpose |
|------|---------|
| `train_aes_m1.py` | M1-optimized training script for AES |
| `evaluate_aes.py` | Evaluation script with AES metrics |
| `pretrain.py` | Original TRM training script |

### Dataset Processing

| File | Purpose |
|------|---------|
| `dataset/build_asappp_dataset.py` | Build ASAPPP datasets from HuggingFace |
| `dataset/build_arc_dataset.py` | Build ARC-AGI datasets (original) |
| `dataset/common.py` | Shared dataset utilities |

### Model Components

| File | Purpose |
|------|---------|
| `models/recursive_reasoning/` | TRM model implementations |
| `models/ema.py` | Exponential Moving Average |
| `models/layers.py` | Neural network layers |
| `evaluators/aes_evaluator.py` | AES-specific metrics (QWK, MSE, etc.) |

### Configuration

| File | Purpose |
|------|---------|
| `config/cfg_aes.yaml` | AES training configuration |
| `config/cfg_pretrain.yaml` | Original TRM configuration |

### Utilities

| File | Purpose |
|------|---------|
| `puzzle_dataset.py` | Dataset loading and batching |
| `requirements.txt` | Python dependencies |

## 🗺️ Learning Paths

### Path 1: Quick Start (30 minutes)
1. Read [GETTING_STARTED.md](GETTING_STARTED.md) (5 min)
2. Run `./quickstart.sh` (5 min setup)
3. Wait for training (20 min - let it run)
4. Check results

### Path 2: Deep Dive (2 hours)
1. Read [GETTING_STARTED.md](GETTING_STARTED.md) (5 min)
2. Read [README_AES.md](README_AES.md) (15 min)
3. Manual setup and training (1.5 hours)
4. Experiment with hyperparameters (30 min)

### Path 3: Understanding the Project (1 hour)
1. Read [README.md](README.md) - Original TRM (10 min)
2. Read [COMPARISON.md](COMPARISON.md) (10 min)
3. Read [CHANGES.md](CHANGES.md) (15 min)
4. Read [README_AES.md](README_AES.md) (15 min)
5. Run `python example_usage.py` (10 min)

### Path 4: Developer (3 hours)
1. Read all documentation (1 hour)
2. Set up environment manually (30 min)
3. Build dataset (30 min)
4. Run training with different configs (1 hour)
5. Evaluate and analyze results (30 min)

## 📊 Quick Reference Tables

### File Sizes & Content

| File | Lines | Purpose |
|------|-------|---------|
| README_AES.md | ~350 | Main documentation |
| CHANGES.md | ~420 | Technical details |
| COMPARISON.md | ~345 | Side-by-side comparison |
| GETTING_STARTED.md | ~290 | Quick start guide |
| train_aes_m1.py | ~575 | Training implementation |
| evaluate_aes.py | ~380 | Evaluation implementation |
| build_asappp_dataset.py | ~375 | Dataset builder |
| aes_evaluator.py | ~250 | Evaluation metrics |

### Documentation by Audience

#### Students & Learners
- Start: [GETTING_STARTED.md](GETTING_STARTED.md)
- Learn: [README_AES.md](README_AES.md)
- Practice: `python example_usage.py`

#### Researchers
- Context: [README.md](README.md)
- Adaptation: [CHANGES.md](CHANGES.md)
- Results: [README_AES.md](README_AES.md) - Expected Results

#### Developers
- Architecture: [CHANGES.md](CHANGES.md) - Architecture Adaptations
- Code: Review Python files
- API: `--help` flags on scripts

#### Decision Makers
- Overview: [COMPARISON.md](COMPARISON.md) - At a Glance
- Cost: [COMPARISON.md](COMPARISON.md) - Performance & Resources
- ROI: [COMPARISON.md](COMPARISON.md) - When to Use Each

## 🔍 Search Guide

### By Topic

**Installation**
- [GETTING_STARTED.md](GETTING_STARTED.md) - Step 1
- [README_AES.md](README_AES.md) - Installation section

**Dataset**
- [README_AES.md](README_AES.md) - Dataset Preparation
- [CHANGES.md](CHANGES.md) - Dataset Differences
- `dataset/build_asappp_dataset.py`

**Training**
- [GETTING_STARTED.md](GETTING_STARTED.md) - Step 3
- [README_AES.md](README_AES.md) - Training section
- `train_aes_m1.py`

**Evaluation**
- [README_AES.md](README_AES.md) - Evaluation Metrics
- [GETTING_STARTED.md](GETTING_STARTED.md) - Step 4
- `evaluate_aes.py`

**Model Architecture**
- [README_AES.md](README_AES.md) - Model Architecture
- [CHANGES.md](CHANGES.md) - Architecture Adaptations
- [COMPARISON.md](COMPARISON.md) - Model Architecture table

**Performance**
- [COMPARISON.md](COMPARISON.md) - Performance & Resources
- [CHANGES.md](CHANGES.md) - Performance Expectations
- [README_AES.md](README_AES.md) - Expected Results

**Troubleshooting**
- [GETTING_STARTED.md](GETTING_STARTED.md) - Common Issues
- [README_AES.md](README_AES.md) - Tips for M1 Mac
- `python example_usage.py` - section 7

**Hardware Requirements**
- [COMPARISON.md](COMPARISON.md) - Hardware & Environment
- [README_AES.md](README_AES.md) - Requirements section

## 💡 Tips for Navigation

1. **Start with GETTING_STARTED.md** if you're new
2. **Use COMPARISON.md** to understand differences
3. **Reference README_AES.md** for detailed information
4. **Check CHANGES.md** for technical implementation details
5. **Run example_usage.py** for interactive help

## 🔗 External Links

- **Original TRM Paper**: https://arxiv.org/abs/2510.04871
- **Original Repository**: https://github.com/AlexiaJM/TinyRecursiveModels
- **ASAPPP Dataset**: https://huggingface.co/datasets/llm-aes
- **Kaggle Competition**: https://www.kaggle.com/c/asap-aes

## 📞 Getting Help

1. **Quick questions**: Check [GETTING_STARTED.md](GETTING_STARTED.md) - Common Issues
2. **Technical issues**: See [README_AES.md](README_AES.md) - Troubleshooting
3. **Understanding code**: Run `python example_usage.py`
4. **Bug reports**: Open a GitHub issue
5. **Feature requests**: Open a GitHub issue

## ✅ Checklist for Success

Before training:
- [ ] Read GETTING_STARTED.md
- [ ] Environment set up
- [ ] Dataset prepared
- [ ] Understand target metrics

After training:
- [ ] Model evaluated (QWK > 0.60)
- [ ] Results documented
- [ ] Checkpoints saved

## 🎓 Citation

If you use this work, cite the original TRM paper:
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

---

**Last Updated**: Compatible with the current repository structure

**Maintained By**: Repository contributors

**License**: Same as original TRM repository

**Feedback**: Open an issue or pull request