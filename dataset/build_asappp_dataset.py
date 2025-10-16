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


def tokenize_input(prompt: str, essay: str, max_length: int = MAX_TOKENS) -> np.ndarray:
    """Convert prompt and essay text to token IDs using a transformer tokenizer."""
    combined_text = prompt + " [SEP] " + essay
    output = tokenizer(
        combined_text,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="np",
    )
    return output["input_ids"].squeeze()


def analyze_token_lengths(dataset, tokenizer, prompt_set_name):
    print(f"\n--- Analyzing Token Lengths for Prompt Set: {prompt_set_name} ---")
    
    length_buckets = {
        "0-512": 0,
        "513-1024": 0,
        "1025-2048": 0,
        "2049+": 0,
    }
    
    total_essays = 0
    for example in tqdm(dataset, desc=f"Analyzing {prompt_set_name}"):
        prompt = example.get("prompt", "")
        essay = example.get("essay", "")
        combined_text = prompt + " [SEP] " + essay
        # Use encode to get the actual number of tokens without padding/truncation
        num_tokens = len(tokenizer.encode(combined_text))
        
        if 0 <= num_tokens <= 512:
            length_buckets["0-512"] += 1
        elif 513 <= num_tokens <= 1024:
            length_buckets["513-1024"] += 1
        elif 1025 <= num_tokens <= 2048:
            length_buckets["1025-2048"] += 1
        else:
            length_buckets["2049+"] += 1
        
        total_essays += 1
        
    print("\n--- Length Analysis Report ---")
    print(f"Total Essays Analyzed: {total_essays}")
    for bucket, count in length_buckets.items():
        percentage = (count / total_essays) * 100 if total_essays > 0 else 0
        print(f"  - {bucket} tokens: {count} essays ({percentage:.2f}%)")
    print("----------------------------\n")

def build_dataset(output_dir: str, dataset_name: str, prompt_set_name: str, score_field: str, min_score: int, max_score: int, num_aug: int = 1, max_char_length: int = 1500, max_tokens: int = MAX_TOKENS):
    """Generic function to build a dataset from a HuggingFace source."""
    print(f"Loading {dataset_name} from HuggingFace...")

    dataset = load_dataset(dataset_name, split="train")
    dataset_splits = dataset.train_test_split(test_size=0.1, seed=42)
    train_ds = dataset_splits["train"]
    test_ds = dataset_splits["test"]

    score_bins = max_score - min_score + 1

    for split_name, ds in [("train", train_ds), ("test", test_ds)]:
        print(f"Processing {split_name} split for prompt set {prompt_set_name}...")

        inputs_list, labels_list, puzzle_identifiers = [], [], []
        puzzle_indices, group_indices = [0], [0]
        current_example_idx, kept_essay_idx, filtered_count = 0, 0, 0

        for example in tqdm(ds, desc=f"Building {split_name}"):
            prompt = example.get("prompt", "")
            essay = example.get("essay", "")
            score = example[score_field]

            if len(essay) > max_char_length:
                filtered_count += 1
                continue

            for _ in range(num_aug):
                input_tokens = tokenize_input(prompt, essay, max_tokens)
                
                label_seq = np.full(max_tokens, float(IGNORE_LABEL_ID), dtype=np.float32)
                label_seq[0] = float(score)

                inputs_list.append(input_tokens)
                labels_list.append(label_seq)
                puzzle_identifiers.append(kept_essay_idx)

                current_example_idx += 1
                puzzle_indices.append(current_example_idx)

            group_indices.append(kept_essay_idx + 1)
            kept_essay_idx += 1

        print(f"Filtered out {filtered_count} essays longer than {max_char_length} characters.")

        # Convert to numpy arrays and save
        split_dir = os.path.join(output_dir, split_name)
        os.makedirs(split_dir, exist_ok=True)
        
        np.save(os.path.join(split_dir, "all__inputs.npy"), np.array(inputs_list, dtype=np.int32))
        np.save(os.path.join(split_dir, "all__labels.npy"), np.array(labels_list, dtype=np.float32))
        np.save(os.path.join(split_dir, "all__puzzle_identifiers.npy"), np.array(puzzle_identifiers, dtype=np.int32))
        np.save(os.path.join(split_dir, "all__puzzle_indices.npy"), np.array(puzzle_indices, dtype=np.int32))
        np.save(os.path.join(split_dir, "all__group_indices.npy"), np.array(group_indices, dtype=np.int32))

        metadata = {
            "pad_id": PAD_ID, "ignore_label_id": IGNORE_LABEL_ID, "blank_identifier_id": BLANK_IDENTIFIER_ID,
            "vocab_size": VOCAB_SIZE, "seq_len": max_tokens, "num_puzzle_identifiers": len(np.unique(puzzle_identifiers)),
            "total_groups": len(group_indices) - 1, "mean_puzzle_examples": num_aug,
            "total_puzzles": len(ds) - filtered_count, "sets": ["all"], "min_score": min_score, "max_score": max_score,
            "score_bins": score_bins, "prompt_set": prompt_set_name,
        }

        with open(os.path.join(split_dir, "dataset.json"), "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"Saved {split_name} split: {len(inputs_list)} examples")


def main():
    parser = argparse.ArgumentParser(description="Build ASAPPP dataset for TRM")
    parser.add_argument("--prompt-set", type=str, required=True, choices=["1-2", "3-6", "7", "all"], help="Which prompt set to build")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory for processed dataset")
    parser.add_argument("--num-aug", type=int, default=1, help="Number of augmentations per essay (default: 1)")
    parser.add_argument("--max-char-length", type=int, default=1500, help="Maximum character length for essays. Longer essays will be filtered out.")
    parser.add_argument("--analyze-lengths", action="store_true", help="If set, script will only analyze and report on essay token lengths instead of building the dataset.")

    args = parser.parse_args()

    prompt_sets = {
        "1-2": ("llm-aes/asappp-1-2-original", "domain1_score", 2, 12),
        "3-6": ("llm-aes/asappp-3-6-original", "domain1_score", 0, 4),
        "7": ("llm-aes/asap-7-original", "domain1_score", 2, 24),
    }

    sets_to_process = prompt_sets.keys() if args.prompt_set == "all" else [args.prompt_set]

    if args.analyze_lengths:
        for set_name in sets_to_process:
            dataset_name, _, _, _ = prompt_sets[set_name]
            dataset = load_dataset(dataset_name, split="train")
            analyze_token_lengths(dataset, tokenizer, set_name)
    else:
        os.makedirs(args.output_dir, exist_ok=True)
        for set_name in sets_to_process:
            dataset_name, score_field, min_score, max_score = prompt_sets[set_name]
            build_dataset(
                output_dir=f"{args.output_dir}_prompts_{set_name}",
                dataset_name=dataset_name,
                prompt_set_name=set_name,
                score_field=score_field,
                min_score=min_score,
                max_score=max_score,
                num_aug=args.num_aug,
                max_char_length=args.max_char_length
            )

    print("Script finished!")


if __name__ == "__main__":
    main()