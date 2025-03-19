import streamlit as st
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import numpy as np


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


model_path = 'saved_bert_model'
model = BertForSequenceClassification.from_pretrained(model_path)
tokenizer = BertTokenizer.from_pretrained(model_path)
model = model.to(device)
model.eval()

def predict_rating(text):
    """Predict rating using fine-tuned BERT"""
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=128).to(device)
    with torch.no_grad():
        output = model(**inputs)
        predicted_rating = output.logits.squeeze().item()
    return np.clip(predicted_rating, 1, 5)

st.title("📖 BERT Rating Prediction")
st.write("Enter a sentence, and BERT will predict its rating (1-5 stars)")

# User input
user_text = st.text_area("Enter review:", )

if st.button("Predict Rating"):
    if user_text.strip():
        predicted_rating = predict_rating(user_text)
        st.write(f"### ⭐ Predicted Rating: {predicted_rating:.2f} / 5")
    else:
        st.warning("Please enter a review!")
