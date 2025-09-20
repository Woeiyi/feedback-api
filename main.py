from fastapi import FastAPI
import uvicorn
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Load model + tokenizer
model = tf.keras.models.load_model("sentiment_lstm_model.h5")
word_index = joblib.load("tokenizer.pkl")

# Email credentials for sending (demo purpose)
SENDER_EMAIL = "woeiyitwy@gmail.com"
SENDER_PASSWORD = "esah kxwg hdch epge"

app = FastAPI()

@app.get("/")
def home():
    return {"message": "✅ Sentiment API is running!"}

def send_negative_feedback_email(user_email: str, review: str, appointment_id: str):
    """
    Sends an automated email to the user apologizing for negative experience.
    """
    try:
        msg = MIMEMultipart()
        msg["From"] = SENDER_EMAIL
        msg["To"] = user_email
        msg["Subject"] = "We Value Your Feedback"

        body = f"""
    Hi there,

        We noticed that your recent feedback from appointment {appointment_id} was negative:

        "{review}"

        We sincerely apologize for any inconvenience caused. Could you please provide more details for future improvements?

        Thank you for your understanding and please do not hesitate to contact us.
        Best regards,
        Support Team
        """
        msg.attach(MIMEText(body, "plain"))

        # Connect to Gmail SMTP server
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(SENDER_EMAIL, SENDER_PASSWORD)
        server.send_message(msg)
        server.quit()
        print(f"✅ Sent email to {user_email}")
    except Exception as e:
        print(f"❌ Failed to send email: {e}")

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
    
    # Predict
    pred = model.predict(padded)[0][0]
    sentiment = "positive" if pred >= 0.5 else "negative"

    # Send email if negative
    if sentiment == "negative":
        send_negative_feedback_email(user_email, review, appointment_id)
    
    return {
        "review": review,
        "sentiment": sentiment,
        "score": float(pred)
    }

# For running locally
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
