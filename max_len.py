import json

with open('wura_data/train/train_afr_tokenized.jsonl') as f:
  data = [json.loads(line) for line in f]

  # Add parentheses around the generator expression

  print(max(d['length'] for d in data))
