from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import torch.nn.functional as F

# Load fine-tuned model and tokenizer
model_path = "fine_tuned_model"
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

# Function to predict sentiment with confidence score
def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=256)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    logits = outputs.logits
    probabilities = F.softmax(logits, dim=-1)  # Convert logits to probabilities
    
    # Get prediction and confidence score
    prediction = torch.argmax(probabilities, dim=-1).item()
    confidence = probabilities[0][prediction].item() * 100  # Convert to percentage
    
    sentiment = "POSITIVE" if prediction == 1 else "NEGATIVE"
    return sentiment, round(confidence, 2)

test_texts = [
    "I absolutely love this product! It's amazing.",
    "This is the worst experience I've ever had.",
    "The service was okay, but not great.",
    "I would recommend this to my friends.",
    "I am very disappointed with the quality.",
    "This is the best thing I've ever bought.",
    "I don't like it at all.",
    "It's just okay, nothing special.",
    "I had high hopes, but it didn't meet my expectations.",
    "I can't believe how good this is!",
    "The worst purchase I've made.",
    "It exceeded my expectations.",
    "I will never buy this again.",
    "I'm very satisfied with my purchase.",
    "This is a fantastic product!",
]   

for text in test_texts:
    sentiment, confidence = predict_sentiment(text)
    print(f"Text: {text}\nPredicted Sentiment: {sentiment} ({confidence}%)\n")