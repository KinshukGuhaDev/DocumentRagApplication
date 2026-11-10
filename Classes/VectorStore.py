import os
import numpy as np
from typing import List, Any, Tuple
import hashlib
import chromadb

class VectorStore:
    def __init__(self, collection_name: str = "pdf_documents_cosine_v1", persist_directory: str = "../../Database/vector_store"):
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.client = None
        self.collection = None
        self._initialize_chromadb()

    def _initialize_chromadb(self):
        try:
            os.makedirs(self.persist_directory, exist_ok=True)
            self.client = chromadb.PersistentClient(path=self.persist_directory)
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={
                    "description": "Collection of PDF document embeddings",
                    "hnsw:space": "cosine"
                }
            )

            print(f"ChromaDB initialized with collection '{self.collection_name}' at '{self.persist_directory}'")
            print(f"Existing documents in collection: {self.collection.count()}")
        except Exception as e:
            print(f"Error initializing ChromaDB: {e}")
            raise

    def add_documents(self, documents: List[Any], embeddings: np.ndarray, batch_size: int = 256):
        if len(documents) != len(embeddings):
            raise ValueError("Number of documents and embeddings must match.")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")

        # Respect Chroma's write limit while keeping temporary memory bounded.
        if hasattr(self.client, "get_max_batch_size"):
            batch_size = min(batch_size, self.client.get_max_batch_size())
        for start in range(0, len(documents), batch_size):
            records = {}
            for i in range(start, min(start + batch_size, len(documents))):
                doc = documents[i]
                metadata = dict(doc.metadata)
                metadata.update(doc_index=i, content_length=len(doc.page_content))
                identity = f"{metadata.get('source', '')}|{metadata.get('page', '')}|{doc.page_content}"
                doc_id = hashlib.sha256(identity.encode("utf-8")).hexdigest()
                records[doc_id] = (doc.page_content, metadata, embeddings[i].tolist())

            # Stable IDs make repeated ingestion update records instead of duplicating them.
            self.collection.upsert(
                ids=list(records),
                documents=[record[0] for record in records.values()],
                metadatas=[record[1] for record in records.values()],
                embeddings=[record[2] for record in records.values()],
            )
        print(f"Stored {len(documents)} chunks; collection contains {self.collection.count()} records.")