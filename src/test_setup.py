from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

text = "I want to save electricity"

embedding = model.encode(text)

print("Embedding length: ", len(embedding))
print("Setup successful!")