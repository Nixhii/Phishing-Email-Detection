import sys
import os
import re
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
from src.preprocess import clean_email_text
from src.feature_extraction import (
    build_feature_pipeline,
    extract_url_features,
    extract_keyword_features,
    extract_special_char_features,
    load_vectorizer,
)
from src.train import load_model
import joblib


class PhishingDetector:
    def __init__(self, model_path=None, vectorizer_path=None):
        self.model = load_model(model_path) if model_path else load_model()
        self.vectorizer = load_vectorizer(vectorizer_path) if vectorizer_path else load_vectorizer()

    def predict(self, email_text, return_prob=True):
        df = pd.DataFrame({"text": [email_text]})
        df["cleaned_text"] = df["text"].apply(clean_email_text)
        features, _ = build_feature_pipeline(
            df, fit_vectorizer=False, vectorizer=self.vectorizer
        )
        pred = self.model.predict(features)[0]
        label = "PHISHING" if pred == 1 else "LEGITIMATE"
        if return_prob and hasattr(self.model, "predict_proba"):
            prob = self.model.predict_proba(features)[0]
            confidence = prob[pred]
            return label, confidence
        return label

    def analyze(self, email_text):
        result = {"text": email_text, "is_phishing": None, "confidence": None, "signals": {}}
        url_count, sus_count, short_count = extract_url_features(email_text)
        phish_score, legit_score = extract_keyword_features(email_text)
        excl_count, ques_count, caps_ratio = extract_special_char_features(email_text)
        result["signals"]["url_count"] = url_count
        result["signals"]["suspicious_urls"] = sus_count
        result["signals"]["shortened_urls"] = short_count
        result["signals"]["phishing_keywords"] = phish_score
        result["signals"]["legitimate_keywords"] = legit_score
        result["signals"]["exclamation_count"] = excl_count
        result["signals"]["caps_ratio"] = caps_ratio
        label, conf = self.predict(email_text)
        result["is_phishing"] = label == "PHISHING"
        result["confidence"] = conf
        return result


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m src.predict \"<email-text>\"")
        print("   OR: python -m src.predict --file path/to/email.txt")
        sys.exit(1)
    detector = PhishingDetector()
    if sys.argv[1] == "--file" and len(sys.argv) > 2:
        with open(sys.argv[2], "r", encoding="utf-8") as f:
            email_text = f.read()
    else:
        email_text = " ".join(sys.argv[1:])
    result = detector.analyze(email_text)
    print("\n" + "=" * 55)
    print("  PHISHING EMAIL DETECTION - ANALYSIS REPORT")
    print("=" * 55)
    print(f"  Email Text : {result['text'][:80]}{'...' if len(result['text']) > 80 else ''}")
    print(f"  Verdict    : {'[!] PHISHING' if result['is_phishing'] else '[+] LEGITIMATE'}")
    print(f"  Confidence : {result['confidence']:.2%}")
    print("-" * 55)
    print("  Signals Detected:")
    for key, val in result["signals"].items():
        print(f"    {key.replace('_', ' ').title():25s}: {val}")
    print("=" * 55)


if __name__ == "__main__":
    main()
