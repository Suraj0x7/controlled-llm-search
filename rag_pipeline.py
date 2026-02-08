from embedder import Embedder
from vector_store import VectorStore

class RAGPipeline:
    def __init__(self):
        self.embedder = Embedder()
        self.store = None

    def build_index(self, chunks):
        embeddings = self.embedder.encode(chunks)
        self.store = VectorStore(dim=len(embeddings[0]))
        self.store.add(embeddings, chunks)

    def search(self, query, k=5):
        q_emb = self.embedder.encode([query])[0]
        return self.store.search(q_emb, k)
