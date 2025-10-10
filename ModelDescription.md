# Model Description: Tiny Recursive Models for Automated Essay Scoring

This document provides a detailed description of the Tiny Recursive Model (TRM) and its specific adaptation for the task of Automated Essay Scoring (AES).

## 1. The Core Idea: Tiny Recursive Models (TRM)

The foundational concept of Tiny Recursive Models (TRM) is **recursive reasoning**. Instead of a single, deep feed-forward pass, a TRM uses a smaller network that iteratively refines its understanding of a problem over a series of steps.

This process is structured into two main loops:
-   **H-cycles (High-level cycles)**: These are the outer loops of reasoning. Each H-cycle represents a major step in solving the problem, analogous to a high-level thought process.
-   **L-cycles (Low-level cycles)**: Within each H-cycle, several L-cycles are performed. These are faster, inner loops that update a latent state, representing a more focused, low-level refinement of the current understanding.

The core data flow is as follows:
1.  An input (e.g., an essay or a puzzle) is encoded into an initial representation.
2.  The model enters the recursive reasoning loop, initializing a latent "answer" state.
3.  For each **H-cycle**, the model performs several **L-cycles** to update its internal latent state based on the input and the current answer.
4.  After the L-cycles, the model updates its "answer" state based on the newly refined latent state.
5.  This process repeats for a fixed number of H-cycles.
6.  The final "answer" state is passed to an output head to produce the prediction.

This recursive structure allows a small model (1-7M parameters) to perform complex reasoning by applying its logic repeatedly, mimicking a human's iterative thought process.

## 2. Adaptation for Automated Essay Scoring (AES)

The original TRM was designed for visual puzzle-solving tasks (like ARC-AGI) on large, multi-GPU systems. This project adapts the architecture for AES, with a focus on efficiency and the ability to run on consumer hardware like an Apple M1 Mac.

Key adaptations include:

| Aspect | Original TRM (Puzzle Solving) | AES Adaptation (Essay Scoring) |
| :--- | :--- | :--- |
| **Task** | Visual pattern recognition | Natural language scoring |
| **Input** | 2D grids of colored cells | 1D sequences of characters (max 512) |
| **Output** | A transformed 2D grid | A single continuous score (regression) |
| **Platform** | Multi-GPU (CUDA) | Single Apple Silicon Mac (MPS) |
| **Model Size** | ~7M parameters | ~1-2M parameters (for M1 version) |
| **Memory** | 24GB+ GPU RAM | 4-6GB Unified RAM |
| **Primary Metric** | Exact Match Accuracy | Quadratic Weighted Kappa (QWK) |

The core recursive reasoning mechanism remains, but it is now applied to the sequence of characters in an essay rather than a 2D grid of pixels.

## 3. The AES Regression Model (`train_aes_m2_regression.py`)

The specific model you are using is the `TinyRecursiveReasoningModel_ACTV1_Regression`. This version introduces further refinements:

-   **Regression-based Scoring**: Instead of classifying an essay into a score bin, this model directly predicts a continuous score. The loss function is **Mean Squared Error (MSE)**, which measures the average squared difference between the predicted scores and the actual scores.

-   **Adaptive Computation Time (ACT)**: This is a key feature of the `_ACTV1_` variant. The model can dynamically decide how many reasoning steps (cycles) to perform for each essay. It does this via a "halting" mechanism. After each step, the model makes a prediction. If the prediction is close enough to the target (within a `halt_threshold`), the model "halts" for that essay and uses the current prediction. This allows the model to spend more "thought" on difficult essays and less on easy ones.

-   **Evaluation**: While trained with MSE, the model's performance is primarily judged by **Quadratic Weighted Kappa (QWK)**. QWK is a standard metric in AES that measures the level of agreement between the model's predicted scores and the human-rater scores, correcting for chance agreement. For this metric, the model's continuous output is rounded to the nearest integer score.

### Output Head Architecture

A key change in this regression model is the use of two distinct output heads, both of which operate on the final state of the model's reasoning (`z_H`):

1.  **Regression Head**: This is a simple linear layer that maps the model's final hidden state of size `hidden_size` to a single continuous value. This value is the predicted essay score.
    -   `Linear(hidden_size -> 1)`

2.  **Halting Head (Q-Head)**: This is a separate linear layer that also takes the final hidden state but outputs two values. These values are the logits for the Adaptive Computation Time (ACT) mechanism, representing the model's confidence in either "halting" or "continuing" its reasoning process.
    -   `Linear(hidden_size -> 2)`

This dual-head structure separates the task of predicting the score from the meta-task of deciding when to stop thinking about the problem.

## 4. Default Training Parameters

These are the default hyperparameters used when running the `train_aes_m2_regression.py` script, which can be overridden via command-line arguments.

| Parameter | Default Value | Description |
| :--- | :--- | :--- |
| `--batch-size` | `64` | Number of essays to process in one batch. |
| `--epochs` | `100` | Total number of training epochs. |
| `--lr` | `3e-4` | The learning rate for the AdamW optimizer. |
| `--hidden_size` | `512` | The main embedding dimension of the model. |
| `--expansion` | `4` | The expansion factor in the MLP layers. |
| `--num_heads` | `8` | Number of attention heads in the Transformer layers. |
| `--L_layers` | `2` | Number of Transformer layers in the L-cycle block. |
| `--H_cycles` | `3` | Number of high-level reasoning cycles. |
| `--L_cycles` | `4` | Number of low-level reasoning cycles per H-cycle. |
| `--eval-interval` | `500` | How often (in epochs) to run evaluation. |
| `--checkpoint-path` | `checkpoints/aes_m2_regression` | Directory to save model checkpoints. |
| `--seed` | `42` | The random seed for reproducibility. |
| `--project-name` | `TRM-AES` | The project name for Weights & Biases logging. |
| `--early-stopping-patience` | `5` | Number of evaluations without improvement to wait before stopping. |
| `--halt-threshold` | `0.5` | The tolerance for the ACT halting mechanism in regression. |
