import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset, load_metric
import evaluate

# Define model and dataset names
model_name = "../checkpoints/Meta-Llama-3-8B_wura_data-packed_bsz256_steps3000_lr6e-5_warmup0.05_afr+amh+eng+fra+hau+ibo+kin+mlg+nya+orm+por+sna+som+sot+swa+tir+xho+yor+zul"
dataset = load_dataset("masakhane/afrimgsm", "amh")


# Load the fine-tuned model and tokenizer
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load the benchmark dataset
#dataset = load_dataset(benchmark_name)

# Define the evaluation metric (assuming a classification task)
#metric = load_metric("accuracy")
metric = load_metric("accuracy", trust_remote_code=True)
#metric = evaluate.load("accuracy", trust_remote_code=True)

# Tokenize the dataset
def preprocess_function(examples):
    return tokenizer(examples['text'], truncation=True, padding=True)

encoded_dataset = dataset.map(preprocess_function, batched=True)

# Define the compute_metrics function
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = torch.argmax(logits, dim=-1)
    return metric.compute(predictions=predictions, references=labels)

# Define the training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    per_device_eval_batch_size=16,
    logging_dir='./logs',
    logging_steps=10,
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    eval_dataset=encoded_dataset["test"],
    compute_metrics=compute_metrics
)

# Evaluate the model
eval_results = trainer.evaluate()

# Print the evaluation results
print(f"Evaluation results: {eval_results}")
