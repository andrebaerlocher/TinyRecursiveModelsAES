"""
ASAPPP Dataset Builder for Automated Essay Scoring (AES)
Loads and combines datasets from HuggingFace into the format expected by TRM.
"""

import os
import json
import argparse
import numpy as np
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer

# Special token IDs
IGNORE_LABEL_ID = -100
BLANK_IDENTIFIER_ID = 0

# Tokenization parameters
TOKENIZER_NAME = "bert-base-uncased"

print(f"Loading tokenizer: {TOKENIZER_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
PAD_ID = tokenizer.pad_token_id
VOCAB_SIZE = tokenizer.vocab_size


def tokenize_input(prompt: str, essay: str, max_length: int) -> np.ndarray:
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
    length_buckets = {"0-512": 0, "513-1024": 0, "1025-2048": 0, "2049+": 0}
    total_essays = 0
    for example in tqdm(dataset, desc=f"Analyzing {prompt_set_name}"):
        prompt = example.get("prompt", "")
        essay = example.get("essay", "")
        combined_text = prompt + " [SEP] " + essay
        num_tokens = len(tokenizer.encode(combined_text))
        
        if 0 <= num_tokens <= 512: length_buckets["0-512"] += 1
        elif 513 <= num_tokens <= 1024: length_buckets["513-1024"] += 1
        elif 1025 <= num_tokens <= 2048: length_buckets["1025-2048"] += 1
        else: length_buckets["2049+"] += 1
        total_essays += 1
        
    print("\n--- Length Analysis Report ---")
    print(f"Total Essays Analyzed: {total_essays}")
    for bucket, count in length_buckets.items():
        percentage = (count / total_essays) * 100 if total_essays > 0 else 0
        print(f"  - {bucket} tokens: {count} essays ({percentage:.2f}%)")
    print("----------------------------\n")

def main():
    parser = argparse.ArgumentParser(description="Build ASAPPP dataset for TRM")
    parser.add_argument("--prompt-set", type=str, required=True, choices=["1-2", "3-6", "7", "all"], help="Which prompt set to build")
    parser.add_argument("--output-dir", type=str, help="Output directory for processed dataset. Required unless --analyze-lengths is used.")
    parser.add_argument("--num-aug", type=int, default=1, help="Number of augmentations per essay (default: 1)")
    parser.add_argument("--max-char-length", type=int, default=8000, help="Filter out essays longer than this character count.")
    parser.add_argument("--max-tokens", type=int, default=1024, help="Maximum token sequence length for the model.")
    parser.add_argument("--analyze-lengths", action="store_true", help="If set, script will only analyze and report on essay token lengths.")

    args = parser.parse_args()

    if not args.analyze_lengths and not args.output_dir:
        parser.error("--output-dir is required unless --analyze-lengths is specified.")

    prompt_sets = {
        "1-2": {"name": "llm-aes/asappp-1-2-original", "score_field": "domain1_score", "min": 2, "max": 12},
        "3-6": {"name": "llm-aes/asappp-3-6-original", "score_field": "domain1_score", "min": 0, "max": 4},
        "7": {"name": "llm-aes/asap-7-original", "score_field": "domain1_score", "min": 2, "max": 24},
    }

    sets_to_process = list(prompt_sets.keys()) if args.prompt_set == "all" else [args.prompt_set]

    if args.analyze_lengths:
        for set_name in sets_to_process:
            dataset_info = prompt_sets[set_name]
            dataset = load_dataset(dataset_info["name"], split="train")
            analyze_token_lengths(dataset, tokenizer, set_name)
        print("Analysis finished!")
        return

    # --- Build Mode ---
    print(f"Building dataset for prompt sets: {sets_to_process} with max_tokens={args.max_tokens}")
    if "3-6" in sets_to_process:
        print("WARNING: Prompt set 3-6 contains very long essays that will be heavily truncated.")

    # Process and collect data from all specified sets for both train and test splits
    for split_name in ["train", "test"]:
        print(f"\nProcessing all datasets for split: '{split_name}'...")
        all_inputs, all_labels, all_puzzle_identifiers = [], [], []
        all_puzzle_indices, all_group_indices = [0], [0]
        total_kept_essays = 0

        for set_name in sets_to_process:
            dataset_info = prompt_sets[set_name]
            full_dataset = load_dataset(dataset_info["name"], split="train")
            split_dataset = full_dataset.train_test_split(test_size=0.1, seed=42)[split_name]
            
            print(f"Processing {len(split_dataset)} examples from prompt set {set_name}...")
            for example in tqdm(split_dataset, desc=f"Set {set_name} ({split_name})"):
                essay = example.get("essay", "")
                if len(essay) > args.max_char_length:
                    continue

                prompt = example.get("prompt", "")
                score = example[dataset_info["score_field"]]

                for _ in range(args.num_aug):
                    input_tokens = tokenize_input(prompt, essay, args.max_tokens)
                    label_seq = np.full(args.max_tokens, float(IGNORE_LABEL_ID), dtype=np.float32)
                    label_seq[0] = float(score)

                    all_inputs.append(input_tokens)
                    all_labels.append(label_seq)
                    all_puzzle_identifiers.append(total_kept_essays)
                    all_puzzle_indices.append(len(all_inputs))
                
                all_group_indices.append(total_kept_essays + 1)
                total_kept_essays += 1

        # Save combined data
        print(f"\nSaving combined '{split_name}' split...")
        split_dir = os.path.join(args.output_dir, split_name)
        os.makedirs(split_dir, exist_ok=True)

        np.save(os.path.join(split_dir, "all__inputs.npy"), np.array(all_inputs, dtype=np.int32))
        np.save(os.path.join(split_dir, "all__labels.npy"), np.array(all_labels, dtype=np.float32))
        np.save(os.path.join(split_dir, "all__puzzle_identifiers.npy"), np.array(all_puzzle_identifiers, dtype=np.int32))
        np.save(os.path.join(split_dir, "all__puzzle_indices.npy"), np.array(all_puzzle_indices, dtype=np.int32))
        np.save(os.path.join(split_dir, "all__group_indices.npy"), np.array(all_group_indices, dtype=np.int32))

        # Create combined metadata
        # Note: min/max score will be over all combined sets, which might not be ideal but is simplest
        min_score_overall = min(prompt_sets[s]["min"] for s in sets_to_process)
        max_score_overall = max(prompt_sets[s]["max"] for s in sets_to_process)

        metadata = {
            "pad_id": PAD_ID, "ignore_label_id": IGNORE_LABEL_ID, "blank_identifier_id": BLANK_IDENTIFIER_ID,
            "vocab_size": VOCAB_SIZE, "seq_len": args.max_tokens, "num_puzzle_identifiers": total_kept_essays,
            "total_groups": total_kept_essays, "mean_puzzle_examples": args.num_aug,
            "total_puzzles": total_kept_essays, "sets": ["all"], "min_score": min_score_overall, "max_score": max_score_overall,
            "score_bins": max_score_overall - min_score_overall + 1, "prompt_set": args.prompt_set,
        }

        with open(os.path.join(split_dir, "dataset.json"), "w") as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Saved combined {split_name} split with {total_kept_essays} unique essays ({len(all_inputs)} examples).")

    print("\nDataset building complete!")

if __name__ == "__main__":
    main()