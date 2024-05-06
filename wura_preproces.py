import torch
import json
import os
from datasets import load_dataset
from transformers import AutoTokenizer

dataset = load_dataset("castorini/wura", "fra", level="passage", verification_mode="no_checks")

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct", token='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')


def tokenize_data(dataset):
    """
    Tokenizes the text data from the dataset and adds a length column.

    Args:
        dataset: A dataset loaded using `load_dataset`.

    Returns:
        A new dataset with "text", "input_ids", and "length" columns.
    """
    # Tokenize all texts in the dataset using the tokenizer
    tokenized_data = dataset.map(
        lambda examples: {
            'text': examples['text'],  # Keep original text
            'input_ids': tokenizer(examples['text'], padding="max_length", truncation=True)['input_ids'],
            'length': [len(ids) for ids in
                       tokenizer(examples['text'], padding="max_length", truncation=True)['input_ids']]
        },
        batched=True
    )

    # Set the format of the dataset if needed (optional)
    return tokenized_data

tokenized_dataset = tokenize_data(dataset)


output_file = "fra_tokenized.jsonl"

if isinstance(tokenized_dataset, dict):
    for split_name, split_data in tokenized_dataset.items():
        split_output_file = f"{split_name}_{output_file}"
        split_data.to_json(split_output_file, orient="records", lines=True)
        print(f"Saved {split_name} split to {split_output_file}")
else:
    tokenized_dataset.to_json(output_file, orient="records", lines=True)
    print(f"Dataset saved in JSONL format as {output_file}.")