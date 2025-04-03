import torch
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import load_from_disk
from torch.utils.data import DataLoader

# Load processed dataset
dataset = load_from_disk("processed_dataset")

# Load pre-trained DistilBERT model for sentiment classification
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

# Define training arguments
training_args = TrainingArguments(
    output_dir="sentiment_model",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    logging_dir="logs",
    logging_steps=10,
    load_best_model_at_end=True,
    save_total_limit=2
)

# Create Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"]
)

# Train the model
trainer.train()

# Save the fine-tuned model
model.save_pretrained("fine_tuned_model")

print("Fine-tuning complete! ✅ Model saved to 'fine_tuned_model'")
