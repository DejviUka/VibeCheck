from transformers import pipeline

# Load PT sentiment analysis model
sentiment_pipeline = pipeline("sentiment-analysis")

# Test
text = "I love this mascara, it brings brightness."
result = sentiment_pipeline(text)

print(result)
