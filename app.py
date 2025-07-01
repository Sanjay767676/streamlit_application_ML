# app.py

import streamlit as st
import joblib

# Load the trained model
model = joblib.load('model/spam_classifier.pkl')

# Web app UI
st.set_page_config(page_title="Spam Detector", layout="centered")
st.title("📩 Real-Time Spam SMS/Email Classifier")
st.write("Enter your message below:")

# Text input box
user_input = st.text_area("Message", height=150)

# Predict button
if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter a message.")
    else:
        prediction = model.predict([user_input])[0]
        prediction_proba = model.predict_proba([user_input])[0]

        if prediction == 1:
            st.error("🚨 This message is classified as **SPAM**.")
        else:
            st.success("✅ This message is classified as **NOT SPAM**.")

        st.markdown(f"**Confidence:** `{max(prediction_proba)*100:.2f}%`")
