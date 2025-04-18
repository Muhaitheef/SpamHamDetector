from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import os

# Initialize FastAPI app
app = FastAPI()

# Enable CORS for all origins (adjust for production if needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model and vectorizer
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "spam_model.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "vectorizer.pkl")

with open(MODEL_PATH, "rb") as model_file:
    model = pickle.load(model_file)

with open(VECTORIZER_PATH, "rb") as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Warm-up the model
print("🔥 Warming up the model...")
_ = model.predict(vectorizer.transform(["This is a warm-up tweet"]))
print("✅ Model warm-up complete.")

# Define request model
class TweetInput(BaseModel):
    tweet: str

# Routes
@app.post("/predict")
async def predict(input: TweetInput):
    tweet_tfidf = vectorizer.transform([input.tweet])
    prediction = int(model.predict(tweet_tfidf)[0])
    confidence = float(model.predict_proba(tweet_tfidf)[0][1])
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
