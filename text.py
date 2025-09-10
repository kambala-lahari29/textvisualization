import streamlit as st
import fitz  # PyMuPDF
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter
import nltk
from textblob import TextBlob

nltk.download('punkt')

# Extract text from PDF
def extract_text_from_pdf(uploaded_file):
    text = ""
    with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc:
        for page in doc:
            text += page.get_text()
    return text

# Generate word frequency
def get_word_freq(text):
    words = nltk.word_tokenize(text.lower())
    words = [word for word in words if word.isalpha()]
    return Counter(words)

# Generate sentiment
def get_sentiment_df(text):
    sentences = nltk.sent_tokenize(text)
    sentiment_data = {
        "Sentence": sentences,
        "Polarity": [TextBlob(sentence).sentiment.polarity for sentence in sentences]
    }
    return pd.DataFrame(sentiment_data)

# Streamlit App
st.title("📊 Text Visualization App")

uploaded_file = st.file_uploader("Upload a PDF or TXT file", type=["pdf", "txt"])

if uploaded_file:
    if uploaded_file.type == "application/pdf":
        text = extract_text_from_pdf(uploaded_file)
    else:
        text = uploaded_file.read().decode("utf-8")

    st.subheader("📄 Extracted Text (First 500 chars)")
    st.write(text[:500] + "...")

    # Word Cloud
    st.subheader("☁️ Word Cloud")
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    fig, ax = plt.subplots()
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis("off")
    st.pyplot(fig)

    # Most Common Words
    st.subheader("🔢 Most Common Words")
    word_freq = get_word_freq(text)
    common_words = word_freq.most_common(10)
    words_df = pd.DataFrame(common_words, columns=["Word", "Frequency"])
    st.bar_chart(words_df.set_index("Word"))

    # Sentiment Analysis
    st.subheader("📈 Sentiment Analysis (Polarity)")
    sentiment_df = get_sentiment_df(text)
    st.line_chart(sentiment_df["Polarity"])
