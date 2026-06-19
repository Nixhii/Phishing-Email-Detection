import re
import nltk
import pandas as pd
from nltk.corpus import stopwords

nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)

STOPWORDS = set(stopwords.words("english"))


def clean_email_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", " URL ", text)
    text = re.sub(r"www\.\S+", " URL ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def preprocess_dataframe(df, text_column="text"):
    df = df.copy()
    df["cleaned_text"] = df[text_column].apply(clean_email_text)
    return df
