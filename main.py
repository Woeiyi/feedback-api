from fastapi import FastAPI
import uvicorn
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load model + tokenizer
model = tf.keras.models.load_model("sentiment_lstm_model.h5")
word_index = joblib.load("tokenizer.pkl")

app = FastAPI()

@app.get("/")
def home():
    return {"message": "✅ Sentiment API is running!"}

@app.post("/predict/")
def predict(review: str):
    # Convert review to sequence of integers
    tokens = [word_index.get(w, 2) for w in review.lower().split()]
    padded = pad_sequences([tokens], maxlen=200)
    pred = model.predict(padded)[0][0]
    sentiment = "positive" if pred >= 0.5 else "negative"
    return {
        "review": review,
        "sentiment": sentiment,
        "score": float(pred)
    }
