"""
Trains two classifiers:
  1. Category classifier  (Ticket Type: Technical issue / Billing inquiry / ...)
  2. Priority classifier  (Ticket Priority: Low / Medium / High / Critical)

For each task, both LinearSVC and Logistic Regression are trained; the one
with the higher macro-F1 on the validation split is kept and saved.

Run:
    python src/train.py
"""

import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, classification_report

import config
from preprocess import clean_text, combine_subject_description
from features import build_vectorizer


def load_dataset() -> pd.DataFrame:
    """Load the real Kaggle CSV if present, otherwise fall back to the
    synthetic sample dataset (auto-generating it if needed)."""
    if os.path.exists(config.REAL_DATA_PATH):
        print(f"Loading real dataset: {config.REAL_DATA_PATH}")
        df = pd.read_csv(config.REAL_DATA_PATH)
    else:
        if not os.path.exists(config.SAMPLE_DATA_PATH):
            print("No dataset found — generating synthetic sample dataset...")
            from generate_sample_data import generate

            generate().to_csv(config.SAMPLE_DATA_PATH, index=False)
        print(f"Real dataset not found at {config.REAL_DATA_PATH}")
        print(f"Falling back to synthetic sample: {config.SAMPLE_DATA_PATH}")
        df = pd.read_csv(config.SAMPLE_DATA_PATH)
    return df


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.dropna(subset=[config.CATEGORY_COL, config.PRIORITY_COL])

    df[config.SUBJECT_COL] = df.get(config.SUBJECT_COL, "")
    df[config.DESCRIPTION_COL] = df.get(config.DESCRIPTION_COL, "")

    df["raw_text"] = df.apply(
        lambda r: combine_subject_description(
            r[config.SUBJECT_COL], r[config.DESCRIPTION_COL]
        ),
        axis=1,
    )
    print("Cleaning text (this can take a moment on large datasets)...")
    df[config.TEXT_COL] = df["raw_text"].apply(clean_text)
    df = df[df[config.TEXT_COL].str.len() > 0]
    return df


def train_task(X_train_text, X_val_text, y_train, y_val, task_name: str):
    """Train + compare LinearSVC vs LogisticRegression for one task,
    return the best (vectorizer, model, label_encoder, report)."""
    vectorizer = build_vectorizer()
    X_train = vectorizer.fit_transform(X_train_text)
    X_val = vectorizer.transform(X_val_text)

    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_val_enc = le.transform(y_val)

    candidates = {
        "LinearSVC": LinearSVC(class_weight="balanced", random_state=config.RANDOM_STATE),
        "LogisticRegression": LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=config.RANDOM_STATE,
        ),
    }

    best_name, best_model, best_f1 = None, None, -1
    for name, model in candidates.items():
        model.fit(X_train, y_train_enc)
        preds = model.predict(X_val)
        macro_f1 = f1_score(y_val_enc, preds, average="macro")
        print(f"  [{task_name}] {name}: macro-F1 = {macro_f1:.4f}")
        if macro_f1 > best_f1:
            best_name, best_model, best_f1 = name, model, macro_f1

    print(f"  -> Best model for {task_name}: {best_name} (macro-F1={best_f1:.4f})")
    val_preds = best_model.predict(X_val)
    report = classification_report(
        y_val_enc, val_preds, target_names=le.classes_, zero_division=0
    )
    return vectorizer, best_model, le, report


def main():
    df = load_dataset()
    df = prepare_data(df)
    print(f"\nTotal usable rows: {len(df)}")

    X_train_text, X_val_text, y_cat_train, y_cat_val, y_pri_train, y_pri_val = train_test_split(
        df[config.TEXT_COL],
        df[config.CATEGORY_COL],
        df[config.PRIORITY_COL],
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=df[config.CATEGORY_COL],
    )

    print("\n=== Training CATEGORY classifier ===")
    cat_vectorizer, cat_model, cat_le, cat_report = train_task(
        X_train_text, X_val_text, y_cat_train, y_cat_val, "category"
    )
    print(cat_report)

    print("\n=== Training PRIORITY classifier ===")
    pri_vectorizer, pri_model, pri_le, pri_report = train_task(
        X_train_text, X_val_text, y_pri_train, y_pri_val, "priority"
    )
    print(pri_report)

    joblib.dump(cat_model, config.CATEGORY_MODEL_PATH)
    joblib.dump(cat_vectorizer, config.CATEGORY_VECTORIZER_PATH)
    joblib.dump(pri_model, config.PRIORITY_MODEL_PATH)
    joblib.dump(pri_vectorizer, config.PRIORITY_VECTORIZER_PATH)
    joblib.dump({"category": cat_le, "priority": pri_le}, config.LABEL_ENCODERS_PATH)

    print(f"\nSaved models to: {config.MODELS_DIR}")

    with open(os.path.join(config.OUTPUTS_DIR, "training_report.txt"), "w") as f:
        f.write("CATEGORY CLASSIFICATION REPORT\n")
        f.write(cat_report)
        f.write("\n\nPRIORITY CLASSIFICATION REPORT\n")
        f.write(pri_report)
    print(f"Saved training report to: {config.OUTPUTS_DIR}/training_report.txt")


if __name__ == "__main__":
    main()
