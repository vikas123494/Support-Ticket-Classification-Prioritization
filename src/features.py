"""
Feature engineering: turns cleaned text into TF-IDF numerical vectors.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
import config


def build_vectorizer() -> TfidfVectorizer:
    """Create a fresh TF-IDF vectorizer with project-standard settings."""
    return TfidfVectorizer(
        max_features=config.TFIDF_MAX_FEATURES,
        ngram_range=config.TFIDF_NGRAM_RANGE,
        min_df=config.TFIDF_MIN_DF,
        sublinear_tf=True,
    )
