from pathlib import Path
from typing import Any, List
import hashlib
import math

import chromadb
import numpy as np

from config import (
    EMBEDDING_MODEL_NAME,
    EMBEDDING_MODEL_TYPE,
    VECTOR_STORE_PATH,
    VECTOR_STORE_COLLECTION_NAME,
)


class VectorStore:
    def __init__(
        self,
        collection_name: str = VECTOR_STORE_COLLECTION_NAME,
        persist_directory: str = VECTOR_STORE_PATH,
        embedding_model_name: str = EMBEDDING_MODEL_NAME,
        embedding_model_type: str = EMBEDDING_MODEL_TYPE,
    ):
        if not collection_name or not collection_name.strip():
            raise ValueError("collection_name cannot be empty")

        self.collection_name = collection_name
        self.embedding_model_name = embedding_model_name
        self.embedding_model_type = embedding_model_type

        requested_path = Path(persist_directory)
        if not requested_path.is_absolute():
            requested_path = Path(__file__).resolve().parents[1] / requested_path

        self.persist_directory = str(requested_path.resolve())
        self.client = None
        self.collection = None

        self._initialize_chromadb()

    def _initialize_chromadb(self):
        print("Vector Storing...")
        print("_" * 50)
        Path(self.persist_directory).mkdir(parents=True, exist_ok=True)

        self.client = chromadb.PersistentClient(path=self.persist_directory)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={
                "description": "Document RAG embeddings",
                "hnsw:space": "cosine",
                "embedding_model": self.embedding_model_name,
                "embedding_model_type": self.embedding_model_type,
            },
        )

        print(f"✅ ChromaDB initialized with collection '{self.collection_name}'")
        print(f"📁 Storage path: {self.persist_directory}")
        print(f"📊 Existing records: {self.collection.count()}")

    @staticmethod
    def _metadata_value(value: Any) -> Any:
        if value is None:
            return ""

        if isinstance(value, (str, int, float, bool)):
            if isinstance(value, float) and not math.isfinite(value):
                return str(value)
            return value

        if isinstance(value, (list, tuple, set, dict)):
            return str(value)

        return str(value)

    def _clean_metadata(self, metadata: dict) -> dict:
        return {
            str(key): self._metadata_value(value)
            for key, value in metadata.items()
        }

    @staticmethod
    def _validate_embeddings(documents: List[Any], embeddings: np.ndarray) -> np.ndarray:
        if documents is None or embeddings is None:
            raise ValueError("documents and embeddings are required")

        vectors = np.asarray(embeddings, dtype=np.float32)

        if vectors.ndim != 2:
            raise ValueError(
                "embeddings must be a 2D array with shape "
                "(number_of_documents, embedding_dimension)"
            )

        if len(documents) != vectors.shape[0]:
            raise ValueError(
                "Number of documents and embeddings must match: "
                f"{len(documents)} != {vectors.shape[0]}"
            )

        if vectors.shape[1] == 0:
            raise ValueError("Embedding dimension cannot be zero")

        if not np.isfinite(vectors).all():
            raise ValueError("Embeddings contain NaN or infinite values")

        return vectors

    def _existing_dimension(self) -> int | None:
        if self.collection.count() == 0:
            return None

        existing = self.collection.get(limit=1, include=["embeddings"])
        stored_embeddings = existing.get("embeddings")

        if stored_embeddings is None or len(stored_embeddings) == 0:
            return None

        return len(stored_embeddings[0])

    def _make_document_id(self, document: Any, metadata: dict) -> str:
        source_file = metadata.get("source_file", metadata.get("source", ""))
        page = metadata.get("page", "")
        chunk_index = metadata.get("chunk_index", "")

        identity = "|".join([
            str(source_file),
            str(page),
            str(chunk_index),
            document.page_content,
        ])

        return hashlib.sha256(identity.encode("utf-8")).hexdigest()

    def add_documents(
        self,
        documents: List[Any],
        embeddings: np.ndarray,
        batch_size: int = 256,
    ):
        if not documents:
            raise ValueError("documents cannot be empty")

        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        vectors = self._validate_embeddings(documents, embeddings)
        dimension = vectors.shape[1]
        existing_dimension = self._existing_dimension()

        if existing_dimension is not None and existing_dimension != dimension:
            raise ValueError(
                "Embedding dimension does not match the existing Chroma "
                f"collection: {dimension} != {existing_dimension}. "
                "Use a new collection when changing embedding models."
            )

        if hasattr(self.client, "get_max_batch_size"):
            batch_size = min(batch_size, self.client.get_max_batch_size())

        total_written = 0

        for start in range(0, len(documents), batch_size):
            end = min(start + batch_size, len(documents))
            ids = []
            texts = []
            metadatas = []
            batch_embeddings = []
            seen_ids = set()

            for index in range(start, end):
                document = documents[index]

                if not hasattr(document, "page_content"):
                    raise TypeError(
                        "Each document must have page_content and metadata"
                    )

                text = document.page_content.strip()
                if not text:
                    raise ValueError(f"Document at index {index} is empty")

                metadata = dict(getattr(document, "metadata", {}))
                metadata.update({
                    "doc_index": index,
                    "content_length": len(text),
                    "embedding_model": self.embedding_model_name,
                    "embedding_model_type": self.embedding_model_type,
                    "embedding_dimension": dimension,
                })
                metadata = self._clean_metadata(metadata)

                document_id = self._make_document_id(document, metadata)

                if document_id in seen_ids:
                    raise ValueError(
                        "Duplicate document ID generated within one batch: "
                        f"{document_id}"
                    )

                seen_ids.add(document_id)
                ids.append(document_id)
                texts.append(text)
                metadatas.append(metadata)
                batch_embeddings.append(vectors[index].tolist())

            self.collection.upsert(
                ids=ids,
                documents=texts,
                metadatas=metadatas,
                embeddings=batch_embeddings,
            )
            total_written += len(ids)

        print(
            f"✅ Upserted {total_written} chunks. "
            f"Collection now contains {self.collection.count()} records."
        )
        print("_" * 50)
        
    def count(self) -> int:
        return self.collection.count()
