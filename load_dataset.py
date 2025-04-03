from datasets import load_dataset

# Load the IMDB dataset
dataset = load_dataset("imdb")

# Print some examples
print(dataset["train"][0])  # Show first review
