from transformers import pipeline

classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli"
)

LABELS = [
    "plastic reduction",
    "energy saving",
    "water conservation",
    "low cost",
    "student hostel",
    "home usage",
    "office usage"
]


def classify_intent(text):
    result = classifier(text, LABELS)

    intents = []

    for label, score in zip(result["labels"], result["scores"]):
        if score > 0.3:
            intents.append((label, round(score, 2)))

    return intents


if __name__ == "__main__":
    q = input("Enter query: ")

    intents = classify_intent(q)

    print("Detected intents:")
    for i in intents:
        print(i)
