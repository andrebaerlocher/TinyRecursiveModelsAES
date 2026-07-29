# Project Context: Adapting Tiny Recursive Models for AES

This document provides a high-level summary for an ML expert reviewing the project. It outlines the project's goal, key architectural features, and the iterative process that led to the current implementation.

## 1. Project Goal

The primary objective was to adapt the original Tiny Recursive Models (TRM) repository, which was designed for logical puzzle-solving on high-end CUDA systems, into a practical pipeline for Automated Essay Scoring (AES). 

The key constraints and goals were:
- To make the system work on consumer-level Apple Silicon hardware.
- To later create a high-performance version for a multi-GPU NVIDIA (H100/H200) system.
- To adapt the model from a puzzle-solver to a text-scoring regression model.
- To create a robust and user-friendly workflow for experimentation.

## 2. Key Architectural Features & Modifications

The project went through a significant evolution, moving from a simple character-level model to a modern, tokenizer-based architecture. The following are the most important features of the current implementation:

#### a. From Characters to Tokens
The initial approach used character-level tokenization. This was quickly identified as a bottleneck for linguistic understanding. The entire data pipeline (`dataset/build_asappp_dataset.py`) was refactored to use a standard Hugging Face tokenizer (`bert-base-uncased`). This allows the model to process text as meaningful word or sub-word units.

#### b. Pre-trained Embeddings
To dramatically improve training stability and performance, the model now loads pre-trained weights from BERT (`bert-base-uncased`) into its token embedding layer upon initialization. This gives the model a foundational understanding of English, rather than forcing it to learn language from scratch.

#### c. Architectural Simplification
A significant portion of the project involved debugging and simplifying the original TRM architecture, which contained complex, puzzle-specific components that were not suitable for this task. Key simplifications included:
- Replacing a fragile custom `CastedSparseEmbedding` layer with a standard `torch.nn.Embedding`.
- Rewriting a complex tensor-reshaping logic for handling input embeddings with a simple, robust `unsqueeze` and `torch.cat` operation.
- Removing the entire "puzzle embedding" feature, which was the source of numerous runtime errors, to create a cleaner architecture focused solely on text.

#### d. Regression-Based Scoring
The model was adapted from a classification task to a regression task. It now predicts a single, continuous score for each essay and is trained using **Mean Squared Error (MSE)** as the loss function.

## 3. The Experimental Workflow

The project is structured to support rapid and flexible experimentation.

#### a. Multi-Platform Training Scripts
There are three distinct training scripts:
- `train_aes_m2_regression.py`: Optimized for single-device **Apple Silicon** (MPS backend).
- `train_aes_h200_regression.py`: A high-performance script for **NVIDIA Multi-GPU** systems, using Distributed Data Parallel (DDP) and Automatic Mixed Precision (AMP).
- `train_aes_baseline_regression.py`: A diagnostic script that runs on Apple Silicon but **disables the model's recursion** (`H_cycles=1`, `L_cycles=1`). This was created to test whether the complex recursive architecture is beneficial for the AES task compared to a more standard encoder model.

#### b. Configurable Data Pipeline
The `dataset/build_asappp_dataset.py` script has several important features, controlled by command-line flags:
- `--prompt-set`: Allows building a dataset from specific essay prompt sets or combining them.
- `--max-tokens`: Allows configuring the model's sequence length (currently defaulted to 1024).
- `--limit-train-essays`: Allows creating smaller training sets for experiments on data efficiency and learning curves.
- `--analyze-lengths`: A utility to report on the token length distribution of datasets without building them.

#### c. TUI Controller
The `manage.py` script provides a simple, interactive Text-based User Interface (TUI) for managing the entire workflow. It guides the user through building datasets, launching training runs on different hardware, and evaluating checkpoints, removing the need to remember long command-line arguments.

## 4. Current Status

The project has successfully moved past a lengthy and complex debugging phase. The data pipeline and model architecture are now stable and robust. The current focus is on **improving model generalization**, as the model learns the training set well but has so far failed to achieve a strong QWK score on the test set.

The immediate next step, which is currently in progress, is to run the **baseline (non-recursive) model**. The results of this experiment will tell us if the TRM architecture itself is a good fit for this task or if a simpler approach is more effective.
