import re

import matplotlib.pyplot as plt
import nltk
import pandas as pd
import streamlit as st
from collections import Counter
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer

st.set_page_config(page_title="Movie Review NLP Analyzer", page_icon="🎬", layout="wide")


@st.cache_resource
def load_stopwords():
    nltk.download("stopwords", quiet=True)
    return set(stopwords.words("english"))


@st.cache_data
def load_csv(file):
    return pd.read_csv(file)


def preprocess(text, stop_words, stemmer, strip_html=True, use_stemming=True):
    text = text.lower()
    if strip_html:
        text = re.sub(r"<[^>]+>", " ", text)  # removes tags like <br />
    text = re.sub(r"[^a-z\s]", "", text)
    words = [w for w in text.split() if w not in stop_words]
    if use_stemming:
        words = [stemmer.stem(w) for w in words]
    return words


# ---------- Header ----------
st.title("🎬 Movie Review Analysis")
st.caption("Text preprocessing, word frequency and TF-IDF keyword extraction")

# ---------- Sidebar ----------
st.sidebar.header("⚙️ Settings")
uploaded = st.sidebar.file_uploader("Upload reviews CSV", type=["csv"])

df = None
if uploaded is not None:
    df = load_csv(uploaded)
else:
    try:
        df = load_csv("reviews.csv")
        st.sidebar.info("Using the bundled reviews.csv")
    except FileNotFoundError:
        pass

if df is None:
    st.info("👈 Upload a CSV file with a review text column to get started.")
    st.stop()

text_cols = list(df.columns)
default_idx = text_cols.index("review") if "review" in text_cols else 0
review_column = st.sidebar.selectbox("Review column", text_cols, index=default_idx)

max_reviews = len(df[review_column].dropna())
num_reviews = st.sidebar.slider(
    "Number of reviews", 1, max(max_reviews, 2), min(200, max_reviews)
)
top_n = st.sidebar.slider("Top N words / keywords", 5, 30, 10)
strip_html = st.sidebar.checkbox("Remove HTML tags (e.g. <br />)", value=True)
use_stemming = st.sidebar.checkbox("Apply Porter stemming", value=True)

# ---------- Processing ----------
stop_words = load_stopwords()
stemmer = PorterStemmer()

reviews = df[review_column].dropna().astype(str).head(num_reviews)
processed = reviews.apply(
    lambda t: preprocess(t, stop_words, stemmer, strip_html, use_stemming)
)

# ---------- Metrics ----------
all_words = [w for review in processed for w in review]
c1, c2, c3 = st.columns(3)
c1.metric("Reviews analyzed", len(reviews))
c2.metric("Total tokens", f"{len(all_words):,}")
c3.metric("Unique words", f"{len(set(all_words)):,}")

tab1, tab2, tab3, tab4 = st.tabs(
    ["📄 Data & Preprocessing", "📊 Word Frequency", "🔑 TF-IDF Keywords", "✍️ Try Your Own"]
)

# ---------- Tab 1 ----------
with tab1:
    st.subheader("Dataset preview")
    st.write("Columns:", df.columns.tolist())
    st.dataframe(df.head(10), use_container_width=True)

    st.subheader("Original vs preprocessed")
    preview = pd.DataFrame(
        {
            "Original": reviews.head(5).str.slice(0, 150) + "...",
            "Preprocessed": processed.head(5).apply(lambda w: " ".join(w[:20]) + " ..."),
        }
    )
    st.dataframe(preview, use_container_width=True)

# ---------- Tab 2 ----------
with tab2:
    st.subheader(f"Top {top_n} most common words")
    top_words = Counter(all_words).most_common(top_n)
    if top_words:
        words, counts = zip(*top_words)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(words, counts, color="#4C78A8")
        ax.set_xlabel("Words")
        ax.set_ylabel("Frequency")
        ax.set_title(f"Top {top_n} Most Common Words in Reviews")
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
        st.dataframe(
            pd.DataFrame(top_words, columns=["Word", "Count"]), use_container_width=True
        )
    else:
        st.warning("No words available to plot.")

# ---------- Tab 3 ----------
with tab3:
    st.subheader(f"Top {top_n} TF-IDF keywords")
    texts = [" ".join(w) for w in processed if w]
    if texts:
        vectorizer = TfidfVectorizer()
        matrix = vectorizer.fit_transform(texts)
        scores = matrix.sum(axis=0).A1
        keywords = vectorizer.get_feature_names_out()
        idx = scores.argsort()[-top_n:][::-1]
        tfidf_df = pd.DataFrame(
            {"Keyword": keywords[idx], "TF-IDF Score": scores[idx].round(4)}
        )

        left, right = st.columns([1, 1])
        with left:
            st.dataframe(tfidf_df, use_container_width=True)
        with right:
            fig2, ax2 = plt.subplots(figsize=(6, 5))
            ax2.barh(tfidf_df["Keyword"][::-1], tfidf_df["TF-IDF Score"][::-1], color="#F58518")
            ax2.set_xlabel("Summed TF-IDF score")
            plt.tight_layout()
            st.pyplot(fig2)

        st.download_button(
            "⬇️ Download keywords as CSV",
            tfidf_df.to_csv(index=False),
            file_name="tfidf_keywords.csv",
            mime="text/csv",
        )
    else:
        st.warning("No text available for TF-IDF analysis.")

# ---------- Tab 4 ----------
with tab4:
    st.subheader("Preprocess any text")
    user_text = st.text_area(
        "Paste a review", "This movie was absolutely wonderful, the acting was great!"
    )
    if user_text:
        st.write("**Tokens after preprocessing:**")
        st.code(str(preprocess(user_text, stop_words, stemmer, strip_html, use_stemming)))
