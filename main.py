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
def predict(review: str, user_email: str, appointment_id: str):
    # Convert review to sequence of integers
    tokens = []
    for w in review.lower().split():
        idx = word_index.get(w)
        if idx is not None and idx < 10000:  # respect num_words
            tokens.append(idx + 3)  # +3 offset for reserved tokens
        else:
            tokens.append(2)  # unknown token

    # Pad sequence
    padded = pad_sequences([tokens], maxlen=200)

    # Predict sentiment
    pred = model.predict(padded)[0][0]
    sentiment = "positive" if pred >= 0.5 else "negative"

    # ✅ Only return sentiment, no email sending
    return {
        "review": review,
        "sentiment": sentiment,
        "score": float(pred)
    }


# For running locally
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
