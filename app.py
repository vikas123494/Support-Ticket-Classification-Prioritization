"""
Streamlit frontend for Support Ticket Classification & Prioritization.
Run: streamlit run app.py  (from the support_ticket_ml/ directory)
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import streamlit as st
from predict import classify_ticket
from preprocess import combine_subject_description

PRIORITY_COLORS = {"Low": "🟢", "Medium": "🟡", "High": "🔴", "Critical": "🚨"}
CATEGORY_ICONS = {
    "Technical issue":      "🔧",
    "Billing inquiry":      "💳",
    "Refund request":       "💰",
    "Cancellation request": "❌",
    "Product inquiry":      "📦",
}

st.set_page_config(page_title="Support Ticket Classifier", page_icon="🎫", layout="centered")
st.title("🎫 Support Ticket Classifier")
st.caption("Automatically categorize and prioritize customer support tickets using ML.")
st.markdown("---")

subject = st.text_input("Ticket Subject", placeholder="e.g. Charged twice for subscription")
description = st.text_area("Ticket Description", placeholder="Describe the issue in detail...", height=150)

if st.button("Classify", use_container_width=True):
    if not subject.strip() and not description.strip():
        st.warning("Please enter a subject or description.")
    else:
        raw = combine_subject_description(subject, description)
        with st.spinner("Classifying..."):
            result = classify_ticket(raw)

        st.markdown("---")
        st.subheader("Result")
        col1, col2 = st.columns(2)
        cat = result["category"]
        pri = result["priority"]
        col1.metric("Category", f"{CATEGORY_ICONS.get(cat, '📋')} {cat}")
        col2.metric("Priority", f"{PRIORITY_COLORS.get(pri, '⚪')} {pri}")

st.markdown("---")
with st.expander("ℹ️ About this app"):
    st.markdown("""
    - **Category** — ticket type: Technical issue, Billing inquiry, Refund request, Cancellation request, Product inquiry
    - **Priority** — urgency level: Low, Medium, High, Critical
    - Models are trained with TF-IDF + LinearSVC on customer support ticket data.
    """)
