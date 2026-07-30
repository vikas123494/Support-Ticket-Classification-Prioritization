"""
Generates a synthetic support-ticket dataset that mirrors the schema of the
Kaggle "Customer Support Ticket Dataset" (Ticket Subject, Ticket Description,
Ticket Type, Ticket Priority, ...).

Use this ONLY to test-drive the pipeline before you have the real CSV.
Run:
    python src/generate_sample_data.py
"""

import random
import pandas as pd
import config

random.seed(config.RANDOM_STATE)

CATEGORIES = {
    "Technical issue": {
        "priority_bias": {"Low": 0.15, "Medium": 0.35, "High": 0.35, "Critical": 0.15},
        "subjects": [
            "App keeps crashing on launch",
            "Unable to log into my account",
            "Website is throwing a 500 error",
            "Sync feature not working",
            "Software freezes during export",
            "Cannot connect to the server",
            "Mobile app stuck on loading screen",
            "Integration with third-party tool broken",
        ],
        "bodies": [
            "Every time I open the app it crashes immediately, even after reinstalling. This started after the last update.",
            "I keep getting a server error when I try to log in with my correct credentials. I have tried resetting my password twice.",
            "The dashboard fails to load and shows a 500 internal server error whenever I try to view my reports.",
            "The sync between my desktop and mobile app stopped working two days ago, my data is not updating.",
            "The software freezes at 80 percent every time I try to export a large file, I have to force close it.",
            "I cannot connect to your server from my office network, but it works fine on my home wifi.",
            "The mobile app is stuck on the loading screen and never gets past it no matter how many times I restart it.",
            "Our Zapier integration stopped passing data through after your latest release, nothing shows up on the other end.",
        ],
    },
    "Billing inquiry": {
        "priority_bias": {"Low": 0.2, "Medium": 0.45, "High": 0.3, "Critical": 0.05},
        "subjects": [
            "Charged twice for the same subscription",
            "Question about my last invoice",
            "Unexpected charge on my card",
            "Need clarification on billing cycle",
            "Invoice total does not match plan price",
            "Subscription renewed at wrong price",
        ],
        "bodies": [
            "My card was charged twice for the same monthly subscription, I would like to understand why and get this corrected.",
            "I have a question about a line item on my latest invoice, it doesn't match what I was quoted at signup.",
            "There is a charge on my card from your company that I do not recognize, can you explain what it is for.",
            "Can you clarify exactly when my billing cycle starts and ends each month, the dates seem inconsistent.",
            "The total on my invoice this month is higher than my usual plan price and I am not sure why.",
            "My subscription renewed at a higher price than what I originally signed up for, please explain the increase.",
        ],
    },
    "Refund request": {
        "priority_bias": {"Low": 0.1, "Medium": 0.3, "High": 0.45, "Critical": 0.15},
        "subjects": [
            "Requesting a refund for last month",
            "Product did not work as advertised, want refund",
            "Refund needed for accidental purchase",
            "Duplicate charge, please refund immediately",
        ],
        "bodies": [
            "I would like to request a full refund for last month's charge because I did not use the service at all.",
            "The product did not work as advertised and I have already spent hours troubleshooting, I want a refund.",
            "I accidentally purchased the annual plan instead of monthly, please refund the difference right away.",
            "I was charged twice for the same order, please refund the duplicate charge as soon as possible.",
        ],
    },
    "Cancellation request": {
        "priority_bias": {"Low": 0.35, "Medium": 0.4, "High": 0.2, "Critical": 0.05},
        "subjects": [
            "Want to cancel my subscription",
            "How do I cancel my plan",
            "Please cancel my account",
            "Cancelling due to lack of use",
        ],
        "bodies": [
            "I would like to cancel my subscription effective immediately, please confirm once this is done.",
            "Can you walk me through how to cancel my current plan, I cannot find the option in settings.",
            "Please cancel my account and stop any future billing, I no longer need the service.",
            "I am cancelling because I have not used the product in months, please process this and confirm.",
        ],
    },
    "Product inquiry": {
        "priority_bias": {"Low": 0.55, "Medium": 0.3, "High": 0.13, "Critical": 0.02},
        "subjects": [
            "Question about upgrading my plan",
            "Does this support multiple users",
            "Asking about enterprise features",
            "Comparing plan tiers",
        ],
        "bodies": [
            "I'm considering upgrading my plan and wanted to know what additional features come with the higher tier.",
            "Does your product support multiple user seats under one account, and how is pricing calculated for that.",
            "I'm evaluating your enterprise features for my team, could you share more details on SSO support.",
            "Could you help me compare the mid and top tier plans, I am not sure which fits our team size.",
        ],
    },
}


def generate(n_rows: int = 1200) -> pd.DataFrame:
    rows = []
    cats = list(CATEGORIES.keys())
    for i in range(n_rows):
        cat = random.choice(cats)
        info = CATEGORIES[cat]
        subject = random.choice(info["subjects"])
        body = random.choice(info["bodies"])
        priority = random.choices(
            list(info["priority_bias"].keys()),
            weights=list(info["priority_bias"].values()),
        )[0]
        rows.append(
            {
                "Ticket ID": i + 1,
                config.SUBJECT_COL: subject,
                config.DESCRIPTION_COL: body,
                config.CATEGORY_COL: cat,
                config.PRIORITY_COL: priority,
                "Ticket Channel": random.choice(["Email", "Chat", "Phone", "Social media"]),
            }
        )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = generate()
    df.to_csv(config.SAMPLE_DATA_PATH, index=False)
    print(f"Synthetic dataset written to: {config.SAMPLE_DATA_PATH}")
    print(f"Rows: {len(df)}")
    print("\nCategory distribution:\n", df[config.CATEGORY_COL].value_counts())
    print("\nPriority distribution:\n", df[config.PRIORITY_COL].value_counts())
