from datasets import load_dataset

dataset = load_dataset("castorini/wura", "kin", level="passage", verification_mode="no_checks")
print(dataset["train"][100])
