"""
Loads the saved models and re-evaluates them on a fresh held-out split,
saving confusion matrix plots to outputs/.

Run:
    python src/evaluate.py
"""

import os
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

import config
from train import load_dataset, prepare_data


def plot_confusion(y_true, y_pred, labels, title, filename):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    out_path = os.path.join(config.OUTPUTS_DIR, filename)
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {out_path}")


def main():
    cat_model = joblib.load(config.CATEGORY_MODEL_PATH)
    cat_vectorizer = joblib.load(config.CATEGORY_VECTORIZER_PATH)
    pri_model = joblib.load(config.PRIORITY_MODEL_PATH)
    pri_vectorizer = joblib.load(config.PRIORITY_VECTORIZER_PATH)
    encoders = joblib.load(config.LABEL_ENCODERS_PATH)
    cat_le, pri_le = encoders["category"], encoders["priority"]

    df = load_dataset()
    df = prepare_data(df)

    X_train_text, X_val_text, y_cat_train, y_cat_val, y_pri_train, y_pri_val = train_test_split(
        df[config.TEXT_COL],
        df[config.CATEGORY_COL],
        df[config.PRIORITY_COL],
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=df[config.CATEGORY_COL],
    )

    X_val_cat = cat_vectorizer.transform(X_val_text)
    y_val_cat_enc = cat_le.transform(y_cat_val)
    cat_preds = cat_model.predict(X_val_cat)

    print("\n=== CATEGORY performance ===")
    print(classification_report(y_val_cat_enc, cat_preds, target_names=cat_le.classes_, zero_division=0))
    plot_confusion(
        y_val_cat_enc, cat_preds, list(range(len(cat_le.classes_))),
        "Category Confusion Matrix", "confusion_category.png"
    )

    X_val_pri = pri_vectorizer.transform(X_val_text)
    y_val_pri_enc = pri_le.transform(y_pri_val)
    pri_preds = pri_model.predict(X_val_pri)

    print("\n=== PRIORITY performance ===")
    print(classification_report(y_val_pri_enc, pri_preds, target_names=pri_le.classes_, zero_division=0))
    plot_confusion(
        y_val_pri_enc, pri_preds, list(range(len(pri_le.classes_))),
        "Priority Confusion Matrix", "confusion_priority.png"
    )


if __name__ == "__main__":
    main()
