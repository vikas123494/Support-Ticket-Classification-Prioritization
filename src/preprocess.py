"""
Text cleaning / preprocessing utilities.

Pipeline: lowercase -> strip URLs/emails/numbers -> remove punctuation
-> tokenize -> remove stopwords -> lemmatize -> rejoin.
"""

import re
import nltk

# Download required NLTK resources quietly (only fetches if missing).
for resource in ["punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4"]:
    try:
        nltk.data.find(
            f"tokenizers/{resource}" if "punkt" in resource else f"corpora/{resource}"
        )
    except LookupError:
        nltk.download(resource, quiet=True)

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

STOPWORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()

URL_RE = re.compile(r"http\S+|www\.\S+")
EMAIL_RE = re.compile(r"\S+@\S+")
NUMBER_RE = re.compile(r"\d+")
NON_ALPHA_RE = re.compile(r"[^a-zA-Z\s]")
MULTI_SPACE_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Clean a single raw ticket text string."""
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = URL_RE.sub(" ", text)
    text = EMAIL_RE.sub(" ", text)
    text = NUMBER_RE.sub(" ", text)
    text = NON_ALPHA_RE.sub(" ", text)
    text = MULTI_SPACE_RE.sub(" ", text).strip()

    tokens = word_tokenize(text)
    tokens = [
        LEMMATIZER.lemmatize(tok)
        for tok in tokens
        if tok not in STOPWORDS and len(tok) > 1
    ]
    return " ".join(tokens)


def combine_subject_description(subject: str, description: str) -> str:
    """Combine subject + description into one text blob before cleaning."""
    subject = subject if isinstance(subject, str) else ""
    description = description if isinstance(description, str) else ""
    return f"{subject}. {description}"


if __name__ == "__main__":
    sample = "I was charged $49.99 TWICE for my subscription!! Please refund me at john@email.com, see http://example.com/invoice"
    print("Raw:  ", sample)
    print("Clean:", clean_text(sample))
