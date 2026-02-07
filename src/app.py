import streamlit as st
import sys
import os

# Allow importing from src/
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from recommend import load_data, load_model, recommend


# -----------------------------
# Load system once
# -----------------------------
@st.cache_resource
def load_system():
    df, embeddings = load_data()
    model = load_model()
    return df, embeddings, model


df, embeddings, model = load_system()


# -----------------------------
# UI
# -----------------------------
st.set_page_config(
    page_title="Sustainability AI",
    page_icon="🌱",
    layout="centered"
)

st.title("🌱 Sustainability Recommendation System")
st.write(
    "Enter your goal and get eco-friendly, practical recommendations."
)

query = st.text_input(
    "Your goal",
    placeholder="e.g. reduce plastic in coffee"
)

if st.button("Get Recommendations"):

    if not query.strip():
        st.warning("Please enter a valid goal.")

    else:
        with st.spinner("Thinking..."):

            results = recommend(query, df, embeddings, model)

        if not results:
            st.info("No highly relevant recommendations found.")

        else:
            st.subheader("Recommended Actions")

            for i, r in enumerate(results, 1):
                st.success(f"{i}. {r}")
