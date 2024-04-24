from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json

# Load the dataset
dataset = load_dataset("castorini/wura")

# Load the Llama 3-8B tokenizer
model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained("model_id")


# Function to tokenize and save the data
def tokenize_and_save(dataset, tokenizer, filename="tokenized_data.jsonl"):
    with open(filename, 'w') as file:
        for entry in dataset['train']:  # Assuming you're interested in the 'train' split
            # Tokenize the text
            tokenized_entry = tokenizer(entry['text'])

            # Prepare the output format
            output = {
                "text": entry['text'],
                "tokens": tokenized_entry.input_ids
            }

            # Write to file
            file.write(json.dumps(output) + '\n')


# Run the tokenization and saving process
tokenize_and_save(dataset, tokenizer)
