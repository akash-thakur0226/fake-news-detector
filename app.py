# =============================================
# Fake News Detector — Streamlit Web App
# =============================================

import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

# ---------- Load saved model & vectorizer ----------
@st.cache_resource
def load_model():
    with open("best_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("tfidf_vectorizer.pkl", "rb") as f:
        tfidf = pickle.load(f)
    return model, tfidf

model, tfidf = load_model()

# ---------- Same cleaning function from Phase 3 ----------
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = word_tokenize(text)
    tokens = [w for w in tokens if w not in stop_words]
    return " ".join(tokens)

# ---------- UI Layout ----------
st.set_page_config(page_title="Fake News Detector", page_icon="🔍")

st.title("🔍 Fake News Detector")
st.markdown("Paste any news article or headline below to check if it's **REAL** or **FAKE**.")
st.markdown("---")

# Text input area
user_input = st.text_area("📰 Paste your news article here:", height=200,
                           placeholder="e.g. Scientists discover new treatment for cancer...")

# Predict button
if st.button("🔎 Analyze News"):

    if user_input.strip() == "":
        st.warning("⚠️ Please paste some text first!")

    else:
        # Preprocess and predict
        cleaned = clean_text(user_input)
        vectorized = tfidf.transform([cleaned])
        prediction = model.predict(vectorized)[0]
        confidence = model.predict_proba(vectorized)[0]

        fake_conf = round(confidence[0] * 100, 2)
        real_conf = round(confidence[1] * 100, 2)

        st.markdown("---")

        # Show result
        if prediction == 0:
            st.error(f"🚨 This news appears to be **FAKE**  ({fake_conf}% confidence)")
        else:
            st.success(f"✅ This news appears to be **REAL**  ({real_conf}% confidence)")

        # Confidence bar chart
        st.markdown("#### Confidence Breakdown")
        col1, col2 = st.columns(2)
        col1.metric("🔴 Fake", f"{fake_conf}%")
        col2.metric("🟢 Real", f"{real_conf}%")

        st.progress(int(real_conf))

        # Extra info
        st.markdown("---")
        st.caption("⚡ Model: ML Classifier trained on 6000+ real-world news articles")
