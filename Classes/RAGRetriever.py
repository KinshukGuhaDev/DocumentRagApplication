from typing import List, Dict, Any
from VectorStore import VectorStore
from EmbeddingManager import EmbeddingManager

class RAGRetriever:
    def __init__(self, vector_store: VectorStore, embedding_manager: EmbeddingManager):
        self.vector_store = vector_store
        self.embedding_manager = embedding_manager

    def retrieve(self, query: str, top_k: int = 5, score_threshold: float = 0.2) -> List[Dict[str, Any]]:
        print(f"Retrieving documents for query: '{query}' with top_k={top_k} and score_threshold={score_threshold}")
        query = query.strip()
        if not query:
            raise ValueError("query must not be empty.")
        if top_k <= 0:
            raise ValueError("top_k must be positive.")
        if not -1 <= score_threshold <= 1:
            raise ValueError("score_threshold must be between -1 and 1.")
        count = self.vector_store.collection.count()
        if count == 0:
            print("Collection is empty. Run the document ingestion cell first.")
            return []
        query_embedding = self.embedding_manager.generate_embeddings([query])[0]

        try:
            results = self.vector_store.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=min(top_k, count),
                include=["documents", "metadatas", "distances"],
            )
            retrieved_docs = []

            if results['documents'] and results['documents'][0]:
                documents = results['documents'][0]
                metadatas = results['metadatas'][0]
                distances = results['distances'][0]
                ids = results['ids'][0]

                for i, (doc_id, document, metadata, distance) in enumerate(zip(ids, documents, metadatas, distances)):
                    similarity_score = max(-1.0, min(1.0, 1.0 - float(distance)))  # Cosine collection
                    if similarity_score >= score_threshold:
                        retrieved_docs.append({
                            "id": doc_id,
                            "content": document,
                            "metadata": metadata,
                            "similarity_score": similarity_score,
                            'distance': distance,
                            'rank': i + 1
                        })

                print(f"Retrieved {len(retrieved_docs)} documents (after filtering)")
                if not retrieved_docs and distances:
                    best_score = 1.0 - min(distances)
                    print(f"Best cosine score: {best_score:.3f}; threshold: {score_threshold:.3f}")
            else:
                print("No documents retrieved from the vector store.")
            return retrieved_docs
        except Exception as e:
            print(f"Error querying the vector store: {e}")
            raise