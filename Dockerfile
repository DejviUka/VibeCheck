# Use official Python image
FROM python:3.10

# Set the working directory
WORKDIR /app

# Copy files to container
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port for FastAPI
EXPOSE 8000

# Command to run the app
CMD ["uvicorn", "sentiment_api:app", "--host", "0.0.0.0", "--port", "8000"]
