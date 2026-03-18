import streamlit as st
import pickle
import re

# Load model
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    return text

# UI
st.title("🎬 Movie Review Sentiment Analyzer")
st.write("Enter a movie review to find out if it's positive or negative.")

user_input = st.text_area("Your Review", height=150)

if st.button("Analyze Sentiment"):
    if user_input.strip() != "":
        cleaned = clean_text(user_input)
        vectorized = vectorizer.transform([cleaned])
        result = model.predict(vectorized)
        proba = model.predict_proba(vectorized)

        if result[0] == "positive":
            st.success(f"😊 Positive Review — {proba[0][1]*100:.1f}% confidence")
        else:
            st.error(f"😞 Negative Review — {proba[0][0]*100:.1f}% confidence")
    else:
        st.warning("Please enter a review first!")
