# Controlled LLM Inference with Embedding-Based Semantic Search

This project integrates:

- Document upload (PDF/TXT)
- Text preprocessing & chunking
- Embedding-based semantic search using FAISS
- Controlled LLM inference (Temperature, Top-P, Max Tokens)
- Deterministic vs Non-deterministic output demonstration
- Retrieval-Augmented Generation (RAG)

## Technologies Used

- Python
- HuggingFace Transformers
- Sentence Transformers
- FAISS
- PyPDF2

## How to Run

### Install dependencies:

pip install -r requirements.txt

### Run application:

python app.py

### Upload document:

Place PDF/TXT inside data/uploads/

## Deterministic Output

Set Temperature = 0 to generate consistent outputs.

## Semantic Search

System retrieves relevant document chunks using vector similarity.

---

Developed as part of LLM Assignment
