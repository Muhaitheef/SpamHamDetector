from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import os
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel

# Initialize FastAPI app
app = FastAPI()

# Enable CORS for all origins (adjust for production if needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model, scaler, and AraBERT
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "arabert_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "arabert_scaler.pkl")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

arabert_model_name = "aubmindlab/bert-base-arabertv2"
tokenizer = AutoTokenizer.from_pretrained(arabert_model_name)
bert_model = AutoModel.from_pretrained(arabert_model_name).to(device)
bert_model.eval()

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)
with open(SCALER_PATH, "rb") as f:
    scaler = pickle.load(f)

# Warm-up the model
print("🔥 Warming up the model...")
dummy_texts = ["Hello this is a test tweet"]
dummy_features = tokenizer(dummy_texts, padding='max_length', truncation=True, max_length=128, return_tensors='pt')
dummy_features = {k: v.to(device) for k, v in dummy_features.items()}
with torch.no_grad():
    outputs = bert_model(**dummy_features)
warmup_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
warmup_embeddings_scaled = scaler.transform(warmup_embeddings)
_ = model.predict(warmup_embeddings_scaled)
print("✅ Model warm-up complete.")

# Define helper to extract BERT features
def extract_bert_features(texts, max_length=128, batch_size=32):
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        encoded = tokenizer(batch_texts, padding='max_length', truncation=True,
                            max_length=max_length, return_tensors='pt')
        encoded = {k: v.to(device) for k, v in encoded.items()}
        with torch.no_grad():
            outputs = bert_model(**encoded)
        embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        all_embeddings.append(embeddings)
    return np.vstack(all_embeddings)

# Define request model
class TweetInput(BaseModel):
    tweet: str

# Routes
@app.post("/predict")
async def predict(input: TweetInput):
    texts = [input.tweet]
    features = extract_bert_features(texts)
    features_scaled = scaler.transform(features)

    prediction = int(model.predict(features_scaled)[0])
    confidence = float(model.predict_proba(features_scaled)[0][1])

    return {
        "prediction": prediction,
        "confidence": round(confidence, 4)
    }

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "message": "Model and server are running"
    }