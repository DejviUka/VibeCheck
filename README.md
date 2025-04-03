
 🎉 VibeCheck: AI-Powered Sentiment Analysis API 🎉

 Welcome to **VibeCheck** – the smart solution for sales & marketing teams to gauge customer sentiment 
 and track brand perception in real time. Powered by a fine-tuned Transformer model, VibeCheck delivers 
 actionable insights by analyzing text data with speed and accuracy. 🚀

 ## 🎯 Purpose

 **VibeCheck** helps marketing professionals quickly understand public sentiment towards their brand. 
 Whether you're monitoring social media, customer reviews, or any other text-based feedback, our API 
 transforms raw text into actionable intelligence – complete with confidence scores. 💡

 ## 🌟 Key Features
 - **AI-Driven Insights:** Leverage a fine-tuned Transformer model for accurate sentiment classification. 🤖
 - **REST API Powered by FastAPI:** Integrate effortlessly into your existing workflows. ⚡
 - **Flexible Deployment:** Run VibeCheck via Docker or directly on your local machine. 🐳💻
 - **Real-Time Analysis:** Get instantaneous sentiment feedback with confidence metrics. ⏱️

 ## 📂 Repository Overview

 - **sentiment_api.py:** The main FastAPI application that loads the fine-tuned model and tokenizer from Hugging Face and exposes the `/predict` endpoint.
 - **fine_tune.py:** Script used for fine-tuning the Transformer model on your sentiment analysis dataset.
 - **load_dataset.py & prepare_dataset.py:** Utilities for loading and preprocessing sentiment analysis datasets.
 - **test_model.py:** Script to test the fine-tuned model locally with sample texts.
 - **Dockerfile:** Contains the instructions to build a Docker image for VibeCheck, ensuring seamless deployment.
 - **requirements.txt:** Lists the Python dependencies required to run the application.
 - **tzdl.py, sentiment_analysis.py, sentiment_api copy.py, test/**: Additional scripts and directories that support model training, testing, and experimentation.
 - **distilbert_tokenizer:** Contains the DistilBERT tokenizer files, which are critical for processing input text.

 ## 🛠️ Installation & Usage

 ### Option 1: Run with Docker (Recommended) 🐳

 1. **Clone the Repository**
    ```sh
    git clone https://github.com/YOUR_USERNAME/VibeCheck.git
    cd VibeCheck
    ```

 2. **Build the Docker Image**
    ```sh
    docker build -t vibecheck-api .
    ```

 3. **Run the Docker Container**
    ```sh
    docker run -p 8000:8000 vibecheck-api
    ```

 4. **Test the API**
    You can use cURL or Postman:
    ```sh
    curl -X POST "http://127.0.0.1:8000/predict" \
         -H "Content-Type: application/json" \
         -d '{"text": "I absolutely love this product!"}'
    ```

 ### Option 2: Run Locally (Without Docker) 💻

 1. **Install Dependencies**
    ```sh
    pip install -r requirements.txt
    ```

 2. **Run the FastAPI Application**
    ```sh
    uvicorn sentiment_api:app --host 0.0.0.0 --port 8000
    ```

 3. **Test the API**
    Use cURL, Postman, or your browser to access the endpoint as shown above.

 ## 🚀 Final Notes

 **VibeCheck** is your go-to tool for real-time sentiment analysis, providing clear and confident insights 
 to help drive your marketing strategy. Whether you’re using Docker for containerized deployment or running 
 the API locally, our comprehensive suite of scripts ensures you have everything you need – from data preparation 
 and model fine-tuning to testing and deployment.

 Get ready to transform raw text into actionable intelligence with VibeCheck! 🎉🤩

