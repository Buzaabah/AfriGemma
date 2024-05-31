from datasets import load_dataset
from transformers import AutoTokenizer

dataset = load_dataset("castorini/wura", "kin", level="passage", verification_mode="no_checks")
print(dataset["train"][100])

# loading tokenizer

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B", token='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')


# We need a function to process the text and include a padding and truncation strategy to handle any variable sequence lengths.
# To process the dataset in one step, we use Datasets map method to apply a preprocessing function over the entire dataset:

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)


tokenized_datasets = dataset.map(tokenize_function, batched=True)

# We can choose a small sample size to fine-tune first with less compute
small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))


