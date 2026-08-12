# 🧠 Knowledge Box - PyTorch Semantic Search Engine

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://knowledge-box-hmxnsgxm4g6db4e5cnt45u.streamlit.app/)

An AI-powered local knowledge retrieval engine built with **PyTorch** and **Sentence Transformers**. It reads local Markdown notes, converts text chunks into vector embeddings, and performs semantic similarity search to find relevant information based on meaning rather than exact keyword matching.

---

## 🚀 Live Demo
Try the interactive web app here:  
👉 **[Launch Knowledge Box Web App](https://knowledge-box-hmxnsgxm4g6db4e5cnt45u.streamlit.app/)**

---

## ✨ Key Features
* 🔍 **Semantic Search:** Uses vector embeddings (`all-MiniLM-L6-v2`) to capture context and intent behind queries.
* ⚡ **PyTorch Engine:** Leverages PyTorch tensor math and cosine similarity for high-speed matching.
* 📁 **Local Knowledge Vault:** Reads `.md` files directly from the `data/` folder.
* 🌐 **Interactive Web Interface:** Built and deployed with **Streamlit Community Cloud**.

---

## 🛠️ Project Structure

```text
knowledge-box/
│
├── data/                  # Folder containing markdown notes (.md)
│   └── sample_note.md
├── app.py                 # Terminal-based search application
├── web_app.py             # Streamlit web dashboard interface
├── requirements.txt       # Dependencies for local setup & deployment
└── README.md              # Project documentation
