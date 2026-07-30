"""
Classify a brand-new, raw support ticket into (category, priority).

Usage:
    python src/predict.py --text "My credit card was charged twice this month, please help."
    python src/predict.py --subject "Refund needed" --description "I was charged twice."
"""

import argparse
import joblib

import config
from preprocess import clean_text, combine_subject_description


def load_artifacts():
    cat_model = joblib.load(config.CATEGORY_MODEL_PATH)
    cat_vectorizer = joblib.load(config.CATEGORY_VECTORIZER_PATH)
    pri_model = joblib.load(config.PRIORITY_MODEL_PATH)
    pri_vectorizer = joblib.load(config.PRIORITY_VECTORIZER_PATH)
    encoders = joblib.load(config.LABEL_ENCODERS_PATH)
    return cat_model, cat_vectorizer, pri_model, pri_vectorizer, encoders


def classify_ticket(raw_text: str) -> dict:
    cat_model, cat_vectorizer, pri_model, pri_vectorizer, encoders = load_artifacts()
    cat_le, pri_le = encoders["category"], encoders["priority"]

    cleaned = clean_text(raw_text)

    cat_vec = cat_vectorizer.transform([cleaned])
    cat_pred_idx = cat_model.predict(cat_vec)[0]
    category = cat_le.inverse_transform([cat_pred_idx])[0]

    pri_vec = pri_vectorizer.transform([cleaned])
    pri_pred_idx = pri_model.predict(pri_vec)[0]
    priority = pri_le.inverse_transform([pri_pred_idx])[0]

    return {"category": category, "priority": priority, "cleaned_text": cleaned}


def main():
    parser = argparse.ArgumentParser(description="Classify a support ticket.")
    parser.add_argument("--text", type=str, default=None, help="Full raw ticket text")
    parser.add_argument("--subject", type=str, default=None, help="Ticket subject line")
    parser.add_argument("--description", type=str, default=None, help="Ticket description body")
    args = parser.parse_args()

    if args.text:
        raw_text = args.text
    elif args.subject or args.description:
        raw_text = combine_subject_description(args.subject or "", args.description or "")
    else:
        parser.error("Provide either --text, or --subject/--description")
        return

    result = classify_ticket(raw_text)
    print(f"\nInput:    {raw_text}")
    print(f"Category: {result['category']}")
    print(f"Priority: {result['priority']}")


if __name__ == "__main__":
    main()
