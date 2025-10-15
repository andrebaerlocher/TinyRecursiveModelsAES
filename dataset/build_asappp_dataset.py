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
from transformers import AutoTokenizer

# Special token IDs
IGNORE_LABEL_ID = -100
BLANK_IDENTIFIER_ID = 0

# Tokenization parameters
MAX_TOKENS = 512
TOKENIZER_NAME = "bert-base-uncased"

print(f"Loading tokenizer: {TOKENIZER_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
PAD_ID = tokenizer.pad_token_id
VOCAB_SIZE = tokenizer.vocab_size


def normalize_score(score: int, min_score: int, max_score: int) -> float:
    """Normalize score to [0, 1] range"""
    if max_score == min_score:
        return 0.5
    return (score - min_score) / (max_score - min_score)


def denormalize_score(normalized: float, min_score: int, max_score: int) -> int:
    """Convert normalized score back to original scale"""
    return int(round(normalized * (max_score - min_score) + min_score))


def tokenize_essay(essay: str, max_length: int = MAX_TOKENS) -> np.ndarray:
    """Convert essay text to token IDs using a transformer tokenizer."""
    output = tokenizer(
        essay,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="np",
    )
    return output["input_ids"].squeeze()


def build_dataset_prompts_1_2(output_dir: str, num_aug: int = 1, max_char_length: int = 1500):
    """Build dataset for ASAPPP prompts 1-2"""
    print("Loading ASAPPP prompts 1-2 from HuggingFace...")

    dataset = load_dataset("llm-aes/asappp-1-2-original", split="train")
    dataset_splits = dataset.train_test_split(test_size=0.1, seed=42)
    train_ds = dataset_splits["train"]
    test_ds = dataset_splits["test"]

    min_score, max_score = 2, 12
    score_bins = max_score - min_score + 1

    for split_name, dataset in [("train", train_ds), ("test", test_ds)]:
        print(f"Processing {split_name} split...")

        inputs_list = []
        labels_list = []
        puzzle_identifiers = []
        puzzle_indices = [0]
        group_indices = [0]

        current_example_idx = 0
        kept_essay_idx = 0
        filtered_count = 0

        for idx, example in enumerate(tqdm(dataset)):
            essay = example["essay"]
            score = example["domain1_score"]

            if len(essay) > max_char_length:
                filtered_count += 1
                continue

            normalized_score = normalize_score(score, min_score, max_score)

            for aug_idx in range(num_aug):
                essay_tokens = tokenize_essay(essay, MAX_TOKENS)
                input_seq = essay_tokens

                label_seq = np.full(MAX_TOKENS, IGNORE_LABEL_ID, dtype=np.int32)
                score_bin = int(normalized_score * (score_bins - 1))
                label_seq[-1] = score_bin

                inputs_list.append(input_seq)
                labels_list.append(label_seq)
                puzzle_identifiers.append(kept_essay_idx)

                current_example_idx += 1
                puzzle_indices.append(current_example_idx)

            group_indices.append(kept_essay_idx + 1)
            kept_essay_idx += 1

        print(f"Filtered out {filtered_count} essays longer than {max_char_length} characters.")

        inputs = np.array(inputs_list, dtype=np.int32)
        labels = np.array(labels_list, dtype=np.int32)
        puzzle_identifiers = np.array(puzzle_identifiers, dtype=np.int32)
        puzzle_indices = np.array(puzzle_indices, dtype=np.int32)
        group_indices = np.array(group_indices, dtype=np.int32)

        split_dir = os.path.join(output_dir, split_name)
        os.makedirs(split_dir, exist_ok=True)

        np.save(os.path.join(split_dir, f"all__inputs.npy"), inputs)
        np.save(os.path.join(split_dir, f"all__labels.npy"), labels)
        np.save(os.path.join(split_dir, f"all__puzzle_identifiers.npy"), puzzle_identifiers)
        np.save(os.path.join(split_dir, f"all__puzzle_indices.npy"), puzzle_indices)
        np.save(os.path.join(split_dir, f"all__group_indices.npy"), group_indices)

        metadata = {
            "pad_id": PAD_ID,
            "ignore_label_id": IGNORE_LABEL_ID,
            "blank_identifier_id": BLANK_IDENTIFIER_ID,
            "vocab_size": VOCAB_SIZE,
            "seq_len": MAX_TOKENS,
            "num_puzzle_identifiers": len(np.unique(puzzle_identifiers)),
            "total_groups": len(group_indices) - 1,
            "mean_puzzle_examples": num_aug,
            "total_puzzles": len(dataset) - filtered_count,
            "sets": ["all"],
            "min_score": min_score,
            "max_score": max_score,
            "score_bins": score_bins,
            "prompt_set": "1-2",
        }

        with open(os.path.join(split_dir, "dataset.json"), "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"Saved {split_name} split: {len(inputs)} examples")


def build_dataset_prompts_3_6(output_dir: str, num_aug: int = 1, max_char_length: int = 1500):
    """Build dataset for ASAPPP prompts 3-6"""
    print("Loading ASAPPP prompts 3-6 from HuggingFace...")

    dataset = load_dataset("llm-aes/asappp-3-6-original", split="train")
    dataset_splits = dataset.train_test_split(test_size=0.1, seed=42)
    train_ds = dataset_splits["train"]
    test_ds = dataset_splits["test"]

    min_score, max_score = 0, 4
    score_bins = max_score - min_score + 1

    for split_name, dataset in [("train", train_ds), ("test", test_ds)]:
        print(f"Processing {split_name} split...")

        inputs_list = []
        labels_list = []
        puzzle_identifiers = []
        puzzle_indices = [0]
        group_indices = [0]

        current_example_idx = 0
        kept_essay_idx = 0
        filtered_count = 0

        for idx, example in enumerate(tqdm(dataset)):
            essay = example["essay"]
            score = example["domain1_score"]

            if len(essay) > max_char_length:
                filtered_count += 1
                continue

            normalized_score = normalize_score(score, min_score, max_score)

            for aug_idx in range(num_aug):
                essay_tokens = tokenize_essay(essay, MAX_TOKENS)
                input_seq = essay_tokens

                label_seq = np.full(MAX_TOKENS, IGNORE_LABEL_ID, dtype=np.int32)
                score_bin = int(normalized_score * (score_bins - 1))
                label_seq[-1] = score_bin

                inputs_list.append(input_seq)
                labels_list.append(label_seq)
                puzzle_identifiers.append(kept_essay_idx)

                current_example_idx += 1
                puzzle_indices.append(current_example_idx)

            group_indices.append(kept_essay_idx + 1)
            kept_essay_idx += 1

        print(f"Filtered out {filtered_count} essays longer than {max_char_length} characters.")

        inputs = np.array(inputs_list, dtype=np.int32)
        labels = np.array(labels_list, dtype=np.int32)
        puzzle_identifiers = np.array(puzzle_identifiers, dtype=np.int32)
        puzzle_indices = np.array(puzzle_indices, dtype=np.int32)
        group_indices = np.array(group_indices, dtype=np.int32)

        split_dir = os.path.join(output_dir, split_name)
        os.makedirs(split_dir, exist_ok=True)

        np.save(os.path.join(split_dir, f"all__inputs.npy"), inputs)
        np.save(os.path.join(split_dir, f"all__labels.npy"), labels)
        np.save(os.path.join(split_dir, f"all__puzzle_identifiers.npy"), puzzle_identifiers)
        np.save(os.path.join(split_dir, f"all__puzzle_indices.npy"), puzzle_indices)
        np.save(os.path.join(split_dir, f"all__group_indices.npy"), group_indices)

        metadata = {
            "pad_id": PAD_ID,
            "ignore_label_id": IGNORE_LABEL_ID,
            "blank_identifier_id": BLANK_IDENTIFIER_ID,
            "vocab_size": VOCAB_SIZE,
            "seq_len": MAX_TOKENS,
            "num_puzzle_identifiers": len(np.unique(puzzle_identifiers)),
            "total_groups": len(group_indices) - 1,
            "mean_puzzle_examples": num_aug,
            "total_puzzles": len(dataset) - filtered_count,
            "sets": ["all"],
            "min_score": min_score,
            "max_score": max_score,
            "score_bins": score_bins,
            "prompt_set": "3-6",
        }

        with open(os.path.join(split_dir, "dataset.json"), "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"Saved {split_name} split: {len(inputs)} examples")


def build_dataset_prompt_7(output_dir: str, num_aug: int = 1, max_char_length: int = 1500):
    """Build dataset for ASAPPP prompt 7"""
    print("Loading ASAPPP prompt 7 from HuggingFace...")

    dataset = load_dataset("llm-aes/asap-7-original", split="train")
    dataset_splits = dataset.train_test_split(test_size=0.1, seed=42)
    train_ds = dataset_splits["train"]
    test_ds = dataset_splits["test"]

    min_score, max_score = 2, 24
    score_bins = max_score - min_score + 1

    for split_name, dataset in [("train", train_ds), ("test", test_ds)]:
        print(f"Processing {split_name} split...")

        inputs_list = []
        labels_list = []
        puzzle_identifiers = []
        puzzle_indices = [0]
        group_indices = [0]

        current_example_idx = 0
        kept_essay_idx = 0
        filtered_count = 0

        for idx, example in enumerate(tqdm(dataset)):
            essay = example["essay"]
            score = example["domain1_score"]

            if len(essay) > max_char_length:
                filtered_count += 1
                continue

            normalized_score = normalize_score(score, min_score, max_score)

            for aug_idx in range(num_aug):
                essay_tokens = tokenize_essay(essay, MAX_TOKENS)
                input_seq = essay_tokens

                label_seq = np.full(MAX_TOKENS, IGNORE_LABEL_ID, dtype=np.int32)
                score_bin = int(normalized_score * (score_bins - 1))
                label_seq[-1] = score_bin

                inputs_list.append(input_seq)
                labels_list.append(label_seq)
                puzzle_identifiers.append(kept_essay_idx)

                current_example_idx += 1
                puzzle_indices.append(current_example_idx)

            group_indices.append(kept_essay_idx + 1)
            kept_essay_idx += 1

        print(f"Filtered out {filtered_count} essays longer than {max_char_length} characters.")

        inputs = np.array(inputs_list, dtype=np.int32)
        labels = np.array(labels_list, dtype=np.int32)
        puzzle_identifiers = np.array(puzzle_identifiers, dtype=np.int32)
        puzzle_indices = np.array(puzzle_indices, dtype=np.int32)
        group_indices = np.array(group_indices, dtype=np.int32)

        split_dir = os.path.join(output_dir, split_name)
        os.makedirs(split_dir, exist_ok=True)

        np.save(os.path.join(split_dir, f"all__inputs.npy"), inputs)
        np.save(os.path.join(split_dir, f"all__labels.npy"), labels)
        np.save(os.path.join(split_dir, f"all__puzzle_identifiers.npy"), puzzle_identifiers)
        np.save(os.path.join(split_dir, f"all__puzzle_indices.npy"), puzzle_indices)
        np.save(os.path.join(split_dir, f"all__group_indices.npy"), group_indices)

        metadata = {
            "pad_id": PAD_ID,
            "ignore_label_id": IGNORE_LABEL_ID,
            "blank_identifier_id": BLANK_IDENTIFIER_ID,
            "vocab_size": VOCAB_SIZE,
            "seq_len": MAX_TOKENS,
            "num_puzzle_identifiers": len(np.unique(puzzle_identifiers)),
            "total_groups": len(group_indices) - 1,
            "mean_puzzle_examples": num_aug,
            "total_puzzles": len(dataset) - filtered_count,
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
    parser.add_argument(
        "--max-char-length",
        type=int,
        default=1500,
        help="Maximum character length for essays. Longer essays will be filtered out.",
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.prompt_set == "1-2" or args.prompt_set == "all":
        build_dataset_prompts_1_2(args.output_dir + "_prompts_1-2", args.num_aug, args.max_char_length)

    if args.prompt_set == "3-6" or args.prompt_set == "all":
        build_dataset_prompts_3_6(args.output_dir + "_prompts_3-6", args.num_aug, args.max_char_length)

    if args.prompt_set == "7" or args.prompt_set == "all":
        build_dataset_prompt_7(args.output_dir + "_prompts_7", args.num_aug, args.max_char_length)

    print("Dataset building complete!")


if __name__ == "__main__":
    main()