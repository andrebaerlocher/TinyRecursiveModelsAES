"""
ASAPPP Dataset Builder for Automated Essay Scoring (AES)
Loads datasets from HuggingFace and converts them to the format expected by TRM
"""

import os
import json
import argparse
from typing import List, Dict, Tuple
import numpy as np
from datasets import load_dataset
from tqdm import tqdm

# Special token IDs
PAD_ID = 0
IGNORE_LABEL_ID = -100
BLANK_IDENTIFIER_ID = 0

# Tokenization parameters
MAX_ESSAY_LENGTH = 512  # Maximum number of characters per essay
CHAR_VOCAB_SIZE = 256  # ASCII character set


def normalize_score(score: int, min_score: int, max_score: int) -> float:
    """Normalize score to [0, 1] range"""
    if max_score == min_score:
        return 0.5
    return (score - min_score) / (max_score - min_score)


def denormalize_score(normalized: float, min_score: int, max_score: int) -> int:
    """Convert normalized score back to original scale"""
    return int(round(normalized * (max_score - min_score) + min_score))


def tokenize_essay(essay: str, max_length: int = MAX_ESSAY_LENGTH) -> np.ndarray:
    """Convert essay text to character-level token IDs"""
    # Convert to bytes and take first max_length characters
    tokens = [min(ord(c), CHAR_VOCAB_SIZE - 1) for c in essay[:max_length]]
    # Pad to max_length
    if len(tokens) < max_length:
        tokens.extend([PAD_ID] * (max_length - len(tokens)))
    return np.array(tokens, dtype=np.int32)


def build_dataset_prompts_1_2(output_dir: str, num_aug: int = 1):
    """Build dataset for ASAPPP prompts 1-2"""
    print("Loading ASAPPP prompts 1-2 from HuggingFace...")

    # Load train and test splits
    train_ds = load_dataset("llm-aes/asappp-1-2-original", split="train")
    test_ds = load_dataset("llm-aes/asappp-1-2-original", split="test")

    # Score range for prompts 1-2 is 2-12 (domain1_score)
    min_score, max_score = 2, 12
    score_bins = max_score - min_score + 1  # 11 bins

    for split_name, dataset in [("train", train_ds), ("test", test_ds)]:
        print(f"Processing {split_name} split...")

        inputs_list = []
        labels_list = []
        puzzle_identifiers = []
        puzzle_indices = [0]
        group_indices = [0]

        current_example_idx = 0

        for idx, example in enumerate(tqdm(dataset)):
            essay = example["essay"]
            score = example["domain1_score"]
            essay_set = example["essay_set"]

            # Normalize score to [0, 1]
            normalized_score = normalize_score(score, min_score, max_score)

            # Apply augmentation (simple: just repeat the example)
            for aug_idx in range(num_aug):
                # Tokenize essay
                essay_tokens = tokenize_essay(essay, MAX_ESSAY_LENGTH)

                # Create input: essay tokens
                input_seq = essay_tokens

                # Create label: normalized score (as single value, repeated for sequence)
                # We'll use the last token position for the prediction
                label_seq = np.full(MAX_ESSAY_LENGTH, IGNORE_LABEL_ID, dtype=np.int32)
                # Put score at the last position (discretized into bins)
                score_bin = int(normalized_score * (score_bins - 1))
                label_seq[-1] = score_bin

                inputs_list.append(input_seq)
                labels_list.append(label_seq)
                puzzle_identifiers.append(idx)  # Group by original essay

                current_example_idx += 1
                puzzle_indices.append(current_example_idx)

            # Each essay is its own group
            group_indices.append(idx + 1)

        # Convert to numpy arrays
        inputs = np.array(inputs_list, dtype=np.int32)
        labels = np.array(labels_list, dtype=np.int32)
        puzzle_identifiers = np.array(puzzle_identifiers, dtype=np.int32)
        puzzle_indices = np.array(puzzle_indices, dtype=np.int32)
        group_indices = np.array(group_indices, dtype=np.int32)

        # Create output directory
        split_dir = os.path.join(output_dir, split_name)
        os.makedirs(split_dir, exist_ok=True)

        # Save arrays
        np.save(os.path.join(split_dir, f"all__inputs.npy"), inputs)
        np.save(os.path.join(split_dir, f"all__labels.npy"), labels)
        np.save(
            os.path.join(split_dir, f"all__puzzle_identifiers.npy"), puzzle_identifiers
        )
        np.save(os.path.join(split_dir, f"all__puzzle_indices.npy"), puzzle_indices)
        np.save(os.path.join(split_dir, f"all__group_indices.npy"), group_indices)

        # Create metadata
        metadata = {
            "pad_id": PAD_ID,
            "ignore_label_id": IGNORE_LABEL_ID,
            "blank_identifier_id": BLANK_IDENTIFIER_ID,
            "vocab_size": CHAR_VOCAB_SIZE,
            "seq_len": MAX_ESSAY_LENGTH,
            "num_puzzle_identifiers": len(puzzle_identifiers),
            "total_groups": len(group_indices) - 1,
            "mean_puzzle_examples": num_aug,
            "total_puzzles": len(dataset),
            "sets": ["all"],
            "min_score": min_score,
            "max_score": max_score,
            "score_bins": score_bins,
            "prompt_set": "1-2",
        }

        with open(os.path.join(split_dir, "dataset.json"), "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"Saved {split_name} split: {len(inputs)} examples")


def build_dataset_prompts_3_6(output_dir: str, num_aug: int = 1):
    """Build dataset for ASAPPP prompts 3-6"""
    print("Loading ASAPPP prompts 3-6 from HuggingFace...")

    # Load train and test splits
    train_ds = load_dataset("llm-aes/asappp-3-6-original", split="train")
    test_ds = load_dataset("llm-aes/asappp-3-6-original", split="test")

    # Score range for prompts 3-6 is 0-4 (domain1_score)
    min_score, max_score = 0, 4
    score_bins = max_score - min_score + 1  # 5 bins

    for split_name, dataset in [("train", train_ds), ("test", test_ds)]:
        print(f"Processing {split_name} split...")

        inputs_list = []
        labels_list = []
        puzzle_identifiers = []
        puzzle_indices = [0]
        group_indices = [0]

        current_example_idx = 0

        for idx, example in enumerate(tqdm(dataset)):
            essay = example["essay"]
            score = example["domain1_score"]
            essay_set = example["essay_set"]

            # Normalize score to [0, 1]
            normalized_score = normalize_score(score, min_score, max_score)

            # Apply augmentation
            for aug_idx in range(num_aug):
                # Tokenize essay
                essay_tokens = tokenize_essay(essay, MAX_ESSAY_LENGTH)

                # Create input: essay tokens
                input_seq = essay_tokens

                # Create label: normalized score
                label_seq = np.full(MAX_ESSAY_LENGTH, IGNORE_LABEL_ID, dtype=np.int32)
                score_bin = int(normalized_score * (score_bins - 1))
                label_seq[-1] = score_bin

                inputs_list.append(input_seq)
                labels_list.append(label_seq)
                puzzle_identifiers.append(idx)

                current_example_idx += 1
                puzzle_indices.append(current_example_idx)

            group_indices.append(idx + 1)

        # Convert to numpy arrays
        inputs = np.array(inputs_list, dtype=np.int32)
        labels = np.array(labels_list, dtype=np.int32)
        puzzle_identifiers = np.array(puzzle_identifiers, dtype=np.int32)
        puzzle_indices = np.array(puzzle_indices, dtype=np.int32)
        group_indices = np.array(group_indices, dtype=np.int32)

        # Create output directory
        split_dir = os.path.join(output_dir, split_name)
        os.makedirs(split_dir, exist_ok=True)

        # Save arrays
        np.save(os.path.join(split_dir, f"all__inputs.npy"), inputs)
        np.save(os.path.join(split_dir, f"all__labels.npy"), labels)
        np.save(
            os.path.join(split_dir, f"all__puzzle_identifiers.npy"), puzzle_identifiers
        )
        np.save(os.path.join(split_dir, f"all__puzzle_indices.npy"), puzzle_indices)
        np.save(os.path.join(split_dir, f"all__group_indices.npy"), group_indices)

        # Create metadata
        metadata = {
            "pad_id": PAD_ID,
            "ignore_label_id": IGNORE_LABEL_ID,
            "blank_identifier_id": BLANK_IDENTIFIER_ID,
            "vocab_size": CHAR_VOCAB_SIZE,
            "seq_len": MAX_ESSAY_LENGTH,
            "num_puzzle_identifiers": len(puzzle_identifiers),
            "total_groups": len(group_indices) - 1,
            "mean_puzzle_examples": num_aug,
            "total_puzzles": len(dataset),
            "sets": ["all"],
            "min_score": min_score,
            "max_score": max_score,
            "score_bins": score_bins,
            "prompt_set": "3-6",
        }

        with open(os.path.join(split_dir, "dataset.json"), "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"Saved {split_name} split: {len(inputs)} examples")


def build_dataset_prompt_7(output_dir: str, num_aug: int = 1):
    """Build dataset for ASAPPP prompt 7"""
    print("Loading ASAPPP prompt 7 from HuggingFace...")

    # Load train and test splits
    train_ds = load_dataset("llm-aes/asap-7-original", split="train")
    test_ds = load_dataset("llm-aes/asap-7-original", split="test")

    # Score range for prompt 7 is 2-24 (domain1_score)
    min_score, max_score = 2, 24
    score_bins = max_score - min_score + 1  # 23 bins

    for split_name, dataset in [("train", train_ds), ("test", test_ds)]:
        print(f"Processing {split_name} split...")

        inputs_list = []
        labels_list = []
        puzzle_identifiers = []
        puzzle_indices = [0]
        group_indices = [0]

        current_example_idx = 0

        for idx, example in enumerate(tqdm(dataset)):
            essay = example["essay"]
            score = example["domain1_score"]

            # Normalize score to [0, 1]
            normalized_score = normalize_score(score, min_score, max_score)

            # Apply augmentation
            for aug_idx in range(num_aug):
                # Tokenize essay
                essay_tokens = tokenize_essay(essay, MAX_ESSAY_LENGTH)

                # Create input: essay tokens
                input_seq = essay_tokens

                # Create label: normalized score
                label_seq = np.full(MAX_ESSAY_LENGTH, IGNORE_LABEL_ID, dtype=np.int32)
                score_bin = int(normalized_score * (score_bins - 1))
                label_seq[-1] = score_bin

                inputs_list.append(input_seq)
                labels_list.append(label_seq)
                puzzle_identifiers.append(idx)

                current_example_idx += 1
                puzzle_indices.append(current_example_idx)

            group_indices.append(idx + 1)

        # Convert to numpy arrays
        inputs = np.array(inputs_list, dtype=np.int32)
        labels = np.array(labels_list, dtype=np.int32)
        puzzle_identifiers = np.array(puzzle_identifiers, dtype=np.int32)
        puzzle_indices = np.array(puzzle_indices, dtype=np.int32)
        group_indices = np.array(group_indices, dtype=np.int32)

        # Create output directory
        split_dir = os.path.join(output_dir, split_name)
        os.makedirs(split_dir, exist_ok=True)

        # Save arrays
        np.save(os.path.join(split_dir, f"all__inputs.npy"), inputs)
        np.save(os.path.join(split_dir, f"all__labels.npy"), labels)
        np.save(
            os.path.join(split_dir, f"all__puzzle_identifiers.npy"), puzzle_identifiers
        )
        np.save(os.path.join(split_dir, f"all__puzzle_indices.npy"), puzzle_indices)
        np.save(os.path.join(split_dir, f"all__group_indices.npy"), group_indices)

        # Create metadata
        metadata = {
            "pad_id": PAD_ID,
            "ignore_label_id": IGNORE_LABEL_ID,
            "blank_identifier_id": BLANK_IDENTIFIER_ID,
            "vocab_size": CHAR_VOCAB_SIZE,
            "seq_len": MAX_ESSAY_LENGTH,
            "num_puzzle_identifiers": len(puzzle_identifiers),
            "total_groups": len(group_indices) - 1,
            "mean_puzzle_examples": num_aug,
            "total_puzzles": len(dataset),
            "sets": ["all"],
            "min_score": min_score,
            "max_score": max_score,
            "score_bins": score_bins,
            "prompt_set": "7",
        }

        with open(os.path.join(split_dir, "dataset.json"), "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"Saved {split_name} split: {len(inputs)} examples")


def main():
    parser = argparse.ArgumentParser(description="Build ASAPPP dataset for TRM")
    parser.add_argument(
        "--prompt-set",
        type=str,
        required=True,
        choices=["1-2", "3-6", "7", "all"],
        help="Which prompt set to build",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for processed dataset",
    )
    parser.add_argument(
        "--num-aug",
        type=int,
        default=1,
        help="Number of augmentations per essay (default: 1)",
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.prompt_set == "1-2" or args.prompt_set == "all":
        build_dataset_prompts_1_2(args.output_dir + "_prompts_1-2", args.num_aug)

    if args.prompt_set == "3-6" or args.prompt_set == "all":
        build_dataset_prompts_3_6(args.output_dir + "_prompts_3-6", args.num_aug)

    if args.prompt_set == "7" or args.prompt_set == "all":
        build_dataset_prompt_7(args.output_dir + "_prompts_7", args.num_aug)

    print("Dataset building complete!")


if __name__ == "__main__":
    main()
