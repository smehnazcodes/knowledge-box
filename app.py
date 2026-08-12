
import os
import torch
from sentence_transformers import SentenceTransformer, util

# 1. Load pre-trained PyTorch NLP model
print("⏳ Loading PyTorch Embedding Model...")
model = SentenceTransformer('all-MiniLM-L6-v2')

# 2. Read all markdown files from the 'data' folder
data_dir = "./data"
documents = []

print("📂 Reading knowledge base files...")
for filename in os.listdir(data_dir):
    if filename.endswith(".md") or filename.endswith(".txt"):
        filepath = os.path.join(data_dir, filename)
        with open(filepath, "r", encoding="utf-8") as file:
            content = file.read()
            # Split by lines to index individual takeaways
            lines = [line.strip() for line in content.split("\n") if line.strip()]
            for line in lines:
                documents.append({"file": filename, "text": line})

print(f"✅ Loaded {len(documents)} snippets from your notes.\n")

# 3. Convert text into PyTorch Tensors
texts = [doc["text"] for doc in documents]
embeddings = model.encode(texts, convert_to_tensor=True)

# 4. Interactive Search Loop
while True:
    query = input("🔍 Ask your Knowledge Box (or type 'exit' to quit): ")
    if query.lower() == 'exit':
        break

    # Convert query to PyTorch vector tensor
    query_embedding = model.encode(query, convert_to_tensor=True)

    # Use PyTorch tensor operations to calculate Cosine Similarity
    cosine_scores = util.cos_sim(query_embedding, embeddings)[0]
    best_idx = torch.argmax(cosine_scores).item()
    best_score = cosine_scores[best_idx].item()

    print("\n" + "="*40)
    print(f"🎯 Match Score: {best_score:.2f}")
    print(f"📄 Source File: {documents[best_idx]['file']}")
    print(f"💡 Snippet:     {documents[best_idx]['text']}")
    print("="*40 + "\n")
    
