import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

# Load dataset
df = pd.read_csv("data/methods.csv")

# Load pretrained model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Convert text to vectors
embeddings = model.encode(df["text"].tolist())

# Save vectors
np.save("data/embeddings.npy", embeddings)

print("Embeddings created:", embeddings.shape)
