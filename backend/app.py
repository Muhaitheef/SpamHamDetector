from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import os

# Initialize Flask app
app = Flask(__name__)
CORS(app, resources={r"/predict": {"origins": "*"}})  # Allow all origins

# Load Model & Vectorizer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "spam_model.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "vectorizer.pkl")

if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
    raise FileNotFoundError("Model or vectorizer file not found! Train the model first.")

with open(MODEL_PATH, "rb") as model_file:
    model = pickle.load(model_file)

with open(VECTORIZER_PATH, "rb") as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Prediction Route
@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    print(f"🔍 Received data: {data}")

    tweet = data.get("tweet", "")
    if not tweet:
        return jsonify({"error": "No tweet provided"}), 400

    # Preprocess input
    tweet_tfidf = vectorizer.transform([tweet])
    
    # Predict
    prediction = model.predict(tweet_tfidf)[0]
    print(f"✅ Prediction: {prediction}")
    
    return jsonify({"prediction": int(prediction)})

# Start Flask Server
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy", "message": "Model and server are running"}), 200
if __name__ == "__main__":
    print("🚀 Flask is running on http://127.0.0.1:5000")
    app.run(debug=True, host="0.0.0.0", port=5000)
