# 🎬 Movie Review NLP Analyzer

A Streamlit app that performs text preprocessing (lowercasing, cleaning, stopword removal, Porter stemming), word-frequency analysis and TF-IDF keyword extraction on movie reviews.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Dataset

Upload a CSV with a review text column (e.g. the Kaggle IMDB dataset with `review` and `sentiment` columns) from the sidebar, or place a `reviews.csv` next to `app.py`.

## Deploy on Streamlit Community Cloud

1. Push this folder to a GitHub repo (`app.py`, `requirements.txt`, optionally `reviews.csv`).
2. Go to https://share.streamlit.io and sign in with GitHub.
3. Click **New app**, choose the repo and branch, set the main file to `app.py`, then **Deploy**.
