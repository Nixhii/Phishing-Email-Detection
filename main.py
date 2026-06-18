import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(__file__))

from src.data_loader import load_dataset, generate_synthetic_dataset
from src.preprocess import preprocess_dataframe
from src.feature_extraction import build_feature_pipeline, save_vectorizer
from src.train import split_data, train_model, save_model
from src.evaluate import evaluate_model, plot_confusion_matrix, plot_roc_curve


def main():
    parser = argparse.ArgumentParser(
        description="Phishing Email Detection Model - Training Pipeline"
    )
    parser.add_argument("--samples", type=int, default=2000, help="Number of synthetic samples to generate")
    parser.add_argument("--model", type=str, default="random_forest",
                        choices=["random_forest", "logistic_regression", "svm"],
                        help="Model type to train")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split ratio")
    args = parser.parse_args()

    print("\n" + "=" * 55)
    print("  PHISHING EMAIL DETECTION SYSTEM")
    print("  Machine Learning based Email Security Classifier")
    print("=" * 55)

    print("\n[STEP 1/5] Loading dataset...")
    df = load_dataset()
    if len(df) < 100:
        print("[INFO] Dataset too small. Generating synthetic data...")
        df = generate_synthetic_dataset(n_samples=args.samples)

    print(f"  -> Dataset size: {len(df)} emails")
    print(f"  -> Legitimate: {len(df[df['label'] == 0])} | Phishing: {len(df[df['label'] == 1])}")

    print("\n[STEP 2/5] Preprocessing emails...")
    df = preprocess_dataframe(df)
    print("  -> Text cleaned (lowercase, URLs tokenized, special chars removed)")

    print("\n[STEP 3/5] Extracting features...")
    X, vectorizer = build_feature_pipeline(df)
    y = df["label"]
    print(f"  -> Feature matrix shape: {X.shape}")
    print("  -> Features: URL stats, keyword scores, TF-IDF vectors")

    print("\n[STEP 4/5] Training model...")
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=args.test_size)
    print(f"  -> Train set: {len(X_train)} | Test set: {len(X_test)}")
    model = train_model(X_train, y_train, model_type=args.model)
    print(f"  -> Model: {args.model.replace('_', ' ').title()}")

    print("\n[STEP 5/5] Evaluating model...")
    metrics = evaluate_model(model, X_test, y_test, model_name=args.model.replace("_", " ").title())
    plot_confusion_matrix(metrics["confusion_matrix"], model_name=args.model.replace("_", " ").title())
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
        plot_roc_curve(y_test, y_prob, model_name=args.model.replace("_", " ").title())

    print("\nSaving model and vectorizer...")
    save_model(model, "phishing_model.joblib")
    save_vectorizer(vectorizer)

    print("\n" + "=" * 55)
    print("  TRAINING COMPLETE!")
    print(f"  Model: {args.model.replace('_', ' ').title()}")
    print(f"  Accuracy: {metrics['accuracy']:.2%}")
    print(f"  F1-Score: {metrics['f1_score']:.4f}")
    print("=" * 55)
    print("\nTo scan a new email, run:")
    print('  python -m src.predict "Your email text here"')
    print()


if __name__ == "__main__":
    main()
