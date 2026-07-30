"""
Central configuration: paths, column names, and hyperparameters.
Keeping these in one place means every script (train/evaluate/predict)
stays consistent without copy-pasted magic strings.
"""

import os

# ---- Paths -----------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")

REAL_DATA_PATH = os.path.join(DATA_DIR, "customer_support_tickets.csv")
SAMPLE_DATA_PATH = os.path.join(DATA_DIR, "sample_customer_support_tickets.csv")

CATEGORY_MODEL_PATH = os.path.join(MODELS_DIR, "category_model.joblib")
PRIORITY_MODEL_PATH = os.path.join(MODELS_DIR, "priority_model.joblib")
CATEGORY_VECTORIZER_PATH = os.path.join(MODELS_DIR, "category_vectorizer.joblib")
PRIORITY_VECTORIZER_PATH = os.path.join(MODELS_DIR, "priority_vectorizer.joblib")
LABEL_ENCODERS_PATH = os.path.join(MODELS_DIR, "label_encoders.joblib")

# ---- Dataset schema (matches the Kaggle "Customer Support Ticket Dataset") ----
SUBJECT_COL = "Ticket Subject"
DESCRIPTION_COL = "Ticket Description"
CATEGORY_COL = "Ticket Type"        # target 1: category
PRIORITY_COL = "Ticket Priority"    # target 2: priority

TEXT_COL = "clean_text"  # engineered combined+cleaned text column

# ---- Training hyperparameters -----------------------------------------
TEST_SIZE = 0.2
RANDOM_STATE = 42

TFIDF_MAX_FEATURES = 20000
TFIDF_NGRAM_RANGE = (1, 2)
TFIDF_MIN_DF = 2

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
