import os
from document_loader import load_document
from chunker import chunk_text
from rag_pipeline import RAGPipeline
from llm_engine import LLMEngine


def main():
    file_name = input("Enter document name (inside data/uploads): ")

    file_path = os.path.join("data", "uploads", file_name)

    if not os.path.exists(file_path):
        print("❌ File not found! Make sure it's inside data/uploads/")
        print("Example: mydoc.pdf")
        return

    print("\nReading document...")
    text = load_document(file_path)

    chunks = chunk_text(text)

    print(f"✅ Created {len(chunks)} text chunks")

    rag = RAGPipeline()
    rag.build_index(chunks)

    llm = LLMEngine()

    while True:
        print("\n--- Semantic Search + LLM ---")
        query = input("Ask a question (or type exit): ")

        if query.lower() == "exit":
            break

        temp = float(input("Temperature (0-1): "))
        top_p = float(input("Top-P (0-1): "))
        max_tokens = int(input("Max tokens: "))

        retrieved = rag.search(query, k=1)

        context = "\n".join(retrieved)

        prompt = f"""
Use the context below to answer clearly:

{context}

Question: {query}
Answer:
"""

        print("\n🎲 Random Output:")
        print(llm.generate(prompt, temp, top_p, max_tokens))

        print("\n📌 Deterministic Output:")
        print(llm.generate(prompt, temp, top_p, max_tokens, deterministic=True))


if __name__ == "__main__":
    main()
