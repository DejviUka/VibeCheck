from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import torch.nn.functional as F

# Initialize FastAPI app
app = FastAPI()

# Load fine-tuned model and tokenizer from Hugging Face
model_path = "dejvi-uka/VibeCheck-model"
tokenizer_path = model_path  # Same for the tokenizer if you're using the fine-tuned one
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained("dejvi-uka/VibeCheck-model")

# Request body structure
class TextInput(BaseModel):
    text: str

# Function to predict sentiment with confidence
def predict_sentiment(text):
    # Tokenizing input text and preparing for model input
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=256)
    
    # Inference
    with torch.no_grad():
        outputs = model(**inputs)
    
    logits = outputs.logits
    probabilities = F.softmax(logits, dim=-1)  # Convert logits to probabilities
    
    # Get prediction and confidence score
    prediction = torch.argmax(probabilities, dim=-1).item()
    confidence = probabilities[0][prediction].item() * 100  # Convert to percentage
    
    # Return the sentiment and confidence
    sentiment = "POSITIVE" if prediction == 1 else "NEGATIVE"
    return {"sentiment": sentiment, "confidence": round(confidence, 2)}

# API route for sentiment analysis
@app.post("/predict")
def get_sentiment(data: TextInput):
    result = predict_sentiment(data.text)
    return result
