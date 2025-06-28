import streamlit as st
import pickle
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Download NLTK resources (if not already)
nltk.download("punkt")
nltk.download("stopwords")

# Reuse the same transform_text function
ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    cleaned = [token for token in tokens if token.isalnum()]
    filtered = [token for token in cleaned if token not in stopwords.words("english")]
    stemmed = [ps.stem(token) for token in filtered]
    return " ".join(stemmed)

# Load model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

st.title("📧 Spam Detection App")

input_sms = st.text_area("Enter the message:")

if st.button("Predict"):
    transformed_sms = transform_text(input_sms)
    vector_input = vectorizer.transform([transformed_sms]).toarray()
    result = model.predict(vector_input)[0]
    prob = model.predict_proba(vector_input)

    if result == 1:
        st.error("🚨 Spam Detected!")
    else:
        st.success("✅ Not Spam.")

    st.write("Prediction Probabilities:")
    st.json({
        "Ham Probability": f"{prob[0][0]:.2f}",
        "Spam Probability": f"{prob[0][1]:.2f}"
    })
