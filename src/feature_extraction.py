import re
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import os

MODELS_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")


def extract_url_features(text):
    urls = re.findall(r"http\S+", str(text))
    if not urls:
        return 0, 0, 0
    suspicious_tlds = [".tk", ".xyz", ".ml", ".ga", ".cf", ".gq", ".top", ".work"]
    suspicious_count = sum(1 for url in urls if any(tld in url for tld in suspicious_tlds))
    shortened = sum(1 for url in urls if re.search(r"(bit\.ly|tinyurl|tiny\.cc|ow\.ly|is\.gd)", url, re.I))
    return len(urls), suspicious_count, shortened


def extract_keyword_features(text):
    text_lower = str(text).lower()
    phishing_keywords = [
        "urgent", "account suspended", "verify", "click here", "limited",
        "prize", "winner", "lottery", "claim", "free",
        "password expired", "security alert", "update your account",
        "login attempt", "unusual activity", "confirm your identity",
        "payment failed", "deactivated", "refund", "action required",
        "irs", "tax refund", "COVID", "relief", "selected",
        "guaranteed", "investment", "million", "exclusive offer",
    ]
    legitimate_keywords = [
        "meeting", "report", "invoice", "reminder", "statement",
        "subscription", "confirmed", "appointment", "newsletter",
        "feedback", "scheduled", "balance", "training", "welcome",
        "thanks", "thank you", "submitted", "review", "processed",
        "available",
    ]
    phishing_score = sum(2 for kw in phishing_keywords if kw in text_lower)
    legitimate_score = sum(1 for kw in legitimate_keywords if kw in text_lower)
    return phishing_score, legitimate_score


def extract_special_char_features(text):
    exclamation = str(text).count("!")
    question = str(text).count("?")
    caps_ratio = sum(1 for c in str(text) if c.isupper()) / max(len(str(text)), 1)
    return exclamation, question, round(caps_ratio, 4)


def build_feature_pipeline(df, text_column="cleaned_text", fit_vectorizer=True, vectorizer=None):
    df = df.copy()
    url_feats = df[text_column].apply(lambda x: pd.Series(extract_url_features(str(x))))
    url_feats.columns = ["url_count", "suspicious_url_count", "shortened_url_count"]
    keyword_feats = df[text_column].apply(lambda x: pd.Series(extract_keyword_features(str(x))))
    keyword_feats.columns = ["phishing_keyword_score", "legitimate_keyword_score"]
    special_feats = df[text_column].apply(lambda x: pd.Series(extract_special_char_features(str(x))))
    special_feats.columns = ["exclamation_count", "question_count", "caps_ratio"]
    tfidf_feats = None
    if fit_vectorizer:
        vectorizer = TfidfVectorizer(max_features=500, ngram_range=(1, 2), stop_words="english")
        tfidf_matrix = vectorizer.fit_transform(df[text_column])
        tfidf_feats = pd.DataFrame(
            tfidf_matrix.toarray(),
            columns=[f"tfidf_{i}" for i in range(tfidf_matrix.shape[1])],
            index=df.index,
        )
    else:
        if vectorizer is None:
            raise ValueError("vectorizer must be provided when fit_vectorizer=False")
        tfidf_matrix = vectorizer.transform(df[text_column])
        tfidf_feats = pd.DataFrame(
            tfidf_matrix.toarray(),
            columns=[f"tfidf_{i}" for i in range(tfidf_matrix.shape[1])],
            index=df.index,
        )
    combined = pd.concat([url_feats, keyword_feats, special_feats, tfidf_feats], axis=1)
    return combined, vectorizer


def save_vectorizer(vectorizer, filepath=None):
    if filepath is None:
        filepath = os.path.join(MODELS_PATH, "tfidf_vectorizer.joblib")
    joblib.dump(vectorizer, filepath)
    print(f"[INFO] Vectorizer saved to {filepath}")


def load_vectorizer(filepath=None):
    if filepath is None:
        filepath = os.path.join(MODELS_PATH, "tfidf_vectorizer.joblib")
    return joblib.load(filepath)
