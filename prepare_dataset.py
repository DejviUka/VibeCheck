from datasets import load_dataset
from transformers import AutoTokenizer

# Load dataset
dataset = load_dataset("imdb")

# Load tokenizer for DistilBERT
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

# Tokenization function
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=256)

# Apply tokenization
tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Remove original text column to save memory
tokenized_datasets = tokenized_datasets.remove_columns(["text"])

# Rename label column to match model format
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")

# Set dataset format for PyTorch
tokenized_datasets.set_format("torch")

# Save processed dataset
tokenized_datasets.save_to_disk("processed_dataset")

print("Dataset preprocessing complete! ✅")
