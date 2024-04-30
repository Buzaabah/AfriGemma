"""
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json


from transformers import AutoTokenizer, AutoModelForCausalLM

#AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B", use_auth_token='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', cache_dir=".cache")
#AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B", use_auth_token='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', cache_dir=".cache")


# Load the dataset
#vdataset = load_dataset("castorini/wura")

# Load the Llama 3-8B tokenizer
model_id = "meta-llama/Meta-Llama-3-8B"
# tokenizer = AutoTokenizer.from_pretrained("model_id")


# Function to tokenize and save the data
def tokenize_and_save(dataset, tokenizer, filename="tokenized_data.jsonl"):
    with open(filename, 'w') as file:
        for entry in dataset['train']:  # Assuming you're interested in the 'train' split
            # Tokenize the text
            tokenized_entry = tokenizer(entry['headline', 'content', ])

            # Prepare the output format
            output = {
                "text": entry['text'],
                "tokens": tokenized_entry.input_ids
            }

            # Write to file
            file.write(json.dumps(output) + '\n')


# Run the tokenization and saving process
tokenize_and_save(dataset, tokenizer)
"""

import json
from transformers import AutoTokenizer
import os

# Path to the JSONL file
file_path = "data/wura/documents-v1.0/train/kin.jsonl"

# Load the Llama 3-8B tokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B", use_auth_token='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')

def tokenize_and_save(file_path, tokenizer, output_filename):
    with open(file_path, 'r', encoding='utf-8') as file, \
         open(output_filename, 'w', encoding='utf-8') as outfile:

        for line in file:
            entry = json.loads(line)
            # Tokenize the 'headline' and 'content'
            tokenized_headline = tokenizer(entry['headline'])
            tokenized_content = tokenizer(entry['content'])

            # Prepare the output format
            output = {
                "headline": entry['headline'],
                "tokens_headline": tokenized_headline.input_ids,
                "length_headline": len(tokenized_headline.input_ids),
                "content": entry['content'],
                "tokens_content": tokenized_content.input_ids,
                "length_content": len(tokenized_content.input_ids),
                "category": entry['category'],
                "url": entry['url']
            }

            # Write to file as JSON
            json.dump(output, outfile)
            outfile.write('\n')

# Output file path
output_file_path = file_path.replace('.jsonl', '_tokenized.jsonl')

# Run the tokenization and saving process
tokenize_and_save(file_path, tokenizer, output_file_path)
