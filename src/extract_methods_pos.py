import os
import pandas as pd
import spacy
import nltk


nltk.data.path.append("C:/nltk_data")

nltk.download("punkt")
from nltk.tokenize import sent_tokenize


# Load spaCy model
nlp = spacy.load("en_core_web_sm")

DATA_DIR = "data/raw_articles"

rows = []
idx = 1

ACTION_VERBS = [
    "use", "carry", "avoid", "replace", "switch", "bring",
    "choose", "buy", "reuse", "refill", "opt", "try", "reduce",
    "prefer", "select", "take", "keep"
]

def action_score(sentence):

    doc = nlp(sentence.lower())

    score = 0

    # +2 if starts with verb
    if doc[0].pos_ == "VERB":
        score += 2

    # +2 if contains action verb
    if any(tok.lemma_ in ACTION_VERBS for tok in doc):
        score += 2

    # +1 if has noun
    if any(tok.pos_ == "NOUN" for tok in doc):
        score += 1

    # -2 if has many numbers (stats)
    if any(tok.like_num for tok in doc):
        score -= 2

    # -1 if too short
    if len(doc) < 6:
        score -= 1

    return score

def is_action_sentence(sentence):

    doc = nlp(sentence.lower())

    # Must start with verb or "try/use/avoid..."
    starts_with_action = doc[0].lemma_ in ACTION_VERBS

    # Must have at least one noun
    has_noun = any(tok.pos_ == "NOUN" for tok in doc)

    # Reject if looks like statistic/fact
    has_number = any(tok.like_num for tok in doc)

    # Reject headings (short + no verb structure)
    too_short = len(doc) < 6

    return (
        starts_with_action
        and has_noun
        and not has_number
        and not too_short
    )


# Read all articles
for file in os.listdir(DATA_DIR):

    if file.endswith(".txt"):

        path = os.path.join(DATA_DIR, file)

        with open(path, "r", encoding="utf-8") as f:
            text = f.read()

        # Split into sentences
        sentences = sent_tokenize(text)

        for s in sentences:

            s = s.strip()

            # Basic cleaning
            if len(s) < 40:
                continue

            # POS-based filtering
            score = action_score(s)
            if score >= 2:
                rows.append([
                    idx,
                    "plastic",
                    "general",
                    "low",
                    "medium",
                    s
                ])

                idx += 1


# Save to CSV
df = pd.DataFrame(
    rows,
    columns=["id", "domain", "context", "cost", "effort", "text"]
)

df.to_csv("data/methods.csv", index=False)

print("Saved", len(df), "action sentences.")
