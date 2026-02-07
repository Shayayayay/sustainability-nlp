import pandas as pd
import numpy as np
import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

use_paraphrase = False
if use_paraphrase:
    from paraphraser import paraphrase

print("=== Recommendation System Running ===")
def load_data():
    df = pd.read_csv("data/methods.csv")
    embeddings = np.load("data/embeddings.npy")
    return df, embeddings

def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

def extract_keywords(text):
    return set(re.findall(r"\w+", text.lower()))

def clean_text(text):
    fillers = [
        "almost all",
        "nearly all",
        "here are",
        "did you know",
        "in 2019",
        "for example"
    ]

    t = text.lower()
    for f in fillers:
        t = t.replace(f, "")

    t = t.strip().capitalize()

    if not t.endswith("."):
        t += "."

    return t

def recommend(query, df, embeddings, model, top_k=3):

    if not query.strip():
        return []
    query_vec = model.encode([query])
    scores = cosine_similarity(query_vec, embeddings)[0]
    query_words = extract_keywords(query)
    scored = []

    for i, row in df.iterrows():

        text = row["text"].lower()
        doc_words = extract_keywords(text)

        overlap = len(query_words & doc_words)

        hybrid = 0.7 * scores[i] + 0.3 * overlap

        scored.append((i, hybrid))

    THRESHOLD = 0.45
    relevant = [(i, s) for i, s in scored if s >= THRESHOLD]
    relevant.sort(key=lambda x: x[1], reverse=True)
    top = relevant[:top_k]
    results = []

    for i, _ in top:
        original = df.iloc[i]["text"]
        cleaned = clean_text(original)
        if use_paraphrase:
            try:
                rewritten = paraphrase(cleaned)
                results.append(rewritten)
            except Exception as e:
                # Fallback if paraphraser fails
                results.append(cleaned)
        else:
            results.append(cleaned)

    return results
def main():
    print("\n=== Sustainability Recommendation System ===\n")
    df, embeddings = load_data()
    model = load_model()

    while True:
        query = input("Enter your goal (or type exit): ")
        if query.lower() == "exit":
            print("Goodbye.")
            break

        results = recommend(query, df, embeddings, model)
        print("\nTop Recommendations:\n")

        if not results:
            print("No highly relevant recommendations found.\n")
            continue

        for r in results:
            print(" -", r)
        print("\n-----------------------------\n")

if __name__ == "__main__":
    main()
