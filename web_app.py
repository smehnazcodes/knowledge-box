import os
import glob
import streamlit as st
import torch
from sentence_transformers import SentenceTransformer, util
st.set_page_config(page_title="Knowledge Box", page_icon="🧠")
st.title("🧠 Knowledge Box Search")

# Load model
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# Load files
documents = []
for filepath in glob.glob("data/*.md"):
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
        for snippet in content.split("\n\n"):
            if snippet.strip():
                documents.append({"file": os.path.basename(filepath), "text": snippet.strip()})

if documents:
    texts = [doc["text"] for doc in documents]
    embeddings = model.encode(texts, convert_to_tensor=True)

    query = st.text_input("Ask a question about your notes:")
    
    if query:
        query_embedding = model.encode(query, convert_to_tensor=True)
        cosine_scores = util.cos_sim(query_embedding, embeddings)[0]
        best_idx = torch.argmax(cosine_scores).item()
        best_score = cosine_scores[best_idx].item()

        st.subheader("Result")
        st.write(f"**Score:** {best_score:.2f}")
        st.write(f"**Source:** `{documents[best_idx]['file']}`")
        st.info(documents[best_idx]['text'])
else:
    st.warning("No markdown files found in the data/ folder.")