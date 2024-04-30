import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Define the file path
file_path = "data/wura/documents-v1.0/train/kin.jsonl"

# Load the data from the file
with open(file_path, 'r') as f:
    data = [json.loads(line) for line in f]

# Initialize the tokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")

# Add special tokens to the tokenizer
tokenizer.add_special_tokens({'additional_special_tokens': ['<special_token_1>', '<special_token_2>']})

# Load the model
model = AutoModelForSequenceClassification.from_pretrained("meta-llama/Meta-Llama-3-8B")

# Resize the token embeddings to include the new special tokens
model.resize_token_embeddings(len(tokenizer))

# Process each entry in the data
output_data = []
for entry in data:
    if 'headline' in entry and 'content' in entry:
        # Tokenize the headline and content
        tokenized_headline = tokenizer(entry['headline'], return_tensors='pt', max_length=512, truncation=True)
        tokenized_content = tokenizer(entry['content'], return_tensors='pt', max_length=512, truncation=True)

        # Create the output dictionary
        output = {
            "headline": entry['headline'],
            "tokens_headline": tokenized_headline.input_ids.tolist()[0],
            "length_headline": len(tokenized_headline.input_ids.tolist()[0]),
            "content": entry['content'],
            "tokens_content": tokenized_content.input_ids.tolist()[0],
            "length_content": len(tokenized_content.input_ids.tolist()[0]),
        }

        # Add the output to the list
        output_data.append(output)

# Save the output to a new file
with open('output.jsonl', 'w') as f:
    for entry in output_data:
        json.dump(entry, f)
        f.write('\n')



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

