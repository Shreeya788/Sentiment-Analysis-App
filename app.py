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

def predict(text):
    cleaned = clean_text(text)
    vectorized = vectorizer.transform([cleaned])
    result = model.predict(vectorized)
    proba = model.predict_proba(vectorized)
    return result[0], proba

# Sample reviews
samples = [
    {"label": "😊 Positive", "text": "This movie was absolutely fantastic! The acting was superb and the storyline kept me hooked till the very end."},
    {"label": "😞 Negative", "text": "Terrible film. Boring plot, bad acting, and a complete waste of time. I regret watching it."},
    {"label": "😊 Positive", "text": "One of the best movies I've seen in years. Brilliant direction and emotional depth."},
    {"label": "😞 Negative", "text": "I fell asleep halfway through. Nothing interesting happens and the characters are flat."},
]

# UI
st.title("🎬 Movie Review Sentiment Analyzer")
st.write("Enter a movie review to find out if it's positive or negative.")

# --- Sample Prediction Button ---
if st.button("🎲 Show Sample Prediction"):
    import random
    sample = random.choice(samples)
    st.info(f"**Sample Review ({sample['label']}):**\n\n_{sample['text']}_")

    result, proba = predict(sample['text'])
    if result == "positive":
        st.success(f"😊 Positive Review — {proba[0][1]*100:.1f}% confidence")
    else:
        st.error(f"😞 Negative Review — {proba[0][0]*100:.1f}% confidence")

st.divider()

# --- Manual Input ---
user_input = st.text_area("Or write your own review", height=150)

if st.button("Analyze Sentiment"):
    if user_input.strip() != "":
        result, proba = predict(user_input)
        if result == "positive":
            st.success(f"😊 Positive Review — {proba[0][1]*100:.1f}% confidence")
        else:
            st.error(f"😞 Negative Review — {proba[0][0]*100:.1f}% confidence")
    else:
        st.warning("Please enter a review first!")