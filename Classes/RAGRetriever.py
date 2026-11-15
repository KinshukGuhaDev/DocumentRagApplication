from typing import Any, Dict, List, Optional
import re
import numpy as np
from Classes.EmbeddingManager import EmbeddingManager
from Classes.VectorStore import VectorStore
from config import (
    TOP_K,
    SCORE_THRESHOLD,
    OPENAI_API_KEY,
    GEMINI_API_KEY,
    LLM_MODEL_TYPE,
    LLM_MODEL_NAME,
    LLM_MODEL_TEMPERATURE,
)
import time
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

class RAGRetriever:
    def __init__(
        self,
        vector_store: Optional[VectorStore] = None,
        embedding_manager: Optional[EmbeddingManager] = None,
        top_k: int = TOP_K,
        score_threshold: float = SCORE_THRESHOLD,
    ):
        self.vector_store = (
            vector_store
            if vector_store is not None
            else VectorStore()
        )
        self.embedding_manager = (
            embedding_manager
            if embedding_manager is not None
            else EmbeddingManager()
        )
        self.top_k = top_k
        self.score_threshold = score_threshold
        self.history = []
        
        if LLM_MODEL_TYPE == "openai":
            if not OPENAI_API_KEY:
                raise ValueError(
                    "OPENAI_API_KEY is required for OpenAI embeddings"
                )
            self.llm = ChatOpenAI(
                model=LLM_MODEL_NAME,
                temperature=LLM_MODEL_TEMPERATURE,
                api_key=OPENAI_API_KEY,
            )
        elif LLM_MODEL_TYPE == "google":
            if not GEMINI_API_KEY:
                raise ValueError(
                    "GEMINI_API_KEY is required for Google embeddings"
                )
            self.llm = ChatGoogleGenerativeAI(
                model=LLM_MODEL_NAME,
                google_api_key=GEMINI_API_KEY,
            )
        else:
            raise ValueError(
                f"Unsupported LLM model type: {LLM_MODEL_TYPE}"
            )
            
        if self.vector_store is None:
                    raise ValueError("vector_store is required")
        
        if self.embedding_manager is None:
            raise ValueError("embedding_manager is required")
        self._validate_collection_compatibility()

    def _validate_collection_compatibility(self):
        count = self.vector_store.collection.count()
        if count == 0:
            return

        stored = self.vector_store.collection.get(
            limit=1,
            include=["embeddings", "metadatas"],
        )

        stored_embeddings = stored.get("embeddings")
        if stored_embeddings is None or len(stored_embeddings) == 0:
            return

        stored_dimension = len(stored_embeddings[0])
        current_dimension = self.embedding_manager.embedding_dimension

        if (
            current_dimension is not None
            and stored_dimension != current_dimension
        ):
            raise ValueError(
                "Query embedding dimension does not match the Chroma "
                f"collection: {current_dimension} != {stored_dimension}. "
                "Use the same embedding model used during ingestion."
            )

        metadatas = stored.get("metadatas") or []
        if metadatas:
            stored_model = metadatas[0].get("embedding_model")
            if (
                stored_model
                and stored_model != self.embedding_manager.model_name
            ):
                raise ValueError(
                    "Embedding model does not match the stored collection: "
                    f"{self.embedding_manager.model_name} != {stored_model}"
                )

    def retrieve_from_vector_store(
        self,
        query: str,
        where: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")

        if self.top_k <= 0:
            raise ValueError("top_k must be positive")

        if not -1 <= self.score_threshold <= 1:
            raise ValueError(
                "score_threshold must be between -1 and 1"
            )

        collection_count = self.vector_store.collection.count()
        if collection_count == 0:
            return []

        query_embedding = self.embedding_manager.embed_query(query.strip())
        query_embedding = np.asarray(
            query_embedding,
            dtype=np.float32,
        )

        if query_embedding.ndim != 1:
            raise ValueError(
                "Query embedding must be a one-dimensional vector"
            )

        # Retrieve a larger semantic candidate pool before reranking. This
        # improves recall for exact facts that dense similarity ranks lower.
        result_count = min(max(self.top_k * 4, 12), collection_count)
        query_kwargs = {
            "query_embeddings": [query_embedding.tolist()],
            "n_results": result_count,
            "include": [
                "documents",
                "metadatas",
                "distances",
            ],
        }

        if where:
            query_kwargs["where"] = where

        results = self.vector_store.collection.query(**query_kwargs)

        documents = (results.get("documents") or [[]])[0]
        metadatas = (results.get("metadatas") or [[]])[0]
        distances = (results.get("distances") or [[]])[0]
        ids = (results.get("ids") or [[]])[0]

        candidates = []
        seen_ids = set()

        for rank, values in enumerate(
            zip(ids, documents, metadatas, distances),
            start=1,
        ):
            document_id, document, metadata, distance = values

            if document_id in seen_ids:
                continue

            seen_ids.add(document_id)

            distance = float(distance)
            similarity_score = max(-1.0, min(1.0, 1.0 - distance))
            lexical_score = self._lexical_score(query, document)
            rerank_score = (
                0.70 * similarity_score
                + 0.30 * lexical_score
            )

            candidates.append({
                "id": document_id,
                "content": document,
                "metadata": metadata or {},
                "similarity_score": similarity_score,
                "lexical_score": lexical_score,
                "rerank_score": rerank_score,
                "distance": distance,
                "vector_rank": rank,
            })

        candidates.sort(
            key=lambda item: item["rerank_score"],
            reverse=True,
        )

        retrieved = []
        for rank, candidate in enumerate(
            candidates[:self.top_k],
            start=1,
        ):
            if candidate["rerank_score"] < self.score_threshold:
                continue

            candidate["rank"] = rank
            retrieved.append(candidate)

        return retrieved

    def _lexical_score(self, query: str, document: str) -> float:
        stop_words = {
            "a", "an", "and", "are", "as", "at", "be", "by", "for",
            "from", "how", "in", "is", "it", "of", "on", "or", "that",
            "the", "to", "was", "what", "when", "where", "which", "who",
            "why", "with",
        }

        def words(text: str) -> List[str]:
            return [
                word
                for word in re.findall(r"[a-z0-9]+", text.lower())
                if word not in stop_words
            ]

        query_words = set(words(query))
        document_words = set(words(document))

        if not query_words:
            return 0.0

        overlap = len(query_words & document_words) / len(query_words)
        normalized_query = " ".join(words(query))
        normalized_document = " ".join(words(document))

        phrase_bonus = 0.20 if (
            normalized_query
            and normalized_query in normalized_document
        ) else 0.0

        return min(1.0, overlap + phrase_bonus)

    def retrieve(self,question: str) -> Dict[str, Any]:
        question = question.strip()
        if not question:
            raise ValueError("question cannot be empty")

        results = self.retrieve_from_vector_store(question)
        sources = []
        context_blocks = []
        seen_sources = set()

        for index, doc in enumerate(results, start=1):
            metadata = doc.get("metadata", {})
            source = metadata.get(
                "source_file",
                metadata.get("source", "unknown"),
            )

            page = metadata.get("page", "unknown")
            content = doc.get("content", "").strip()

            source_key = (
                source,
                page,
                metadata.get("chunk_id", doc.get("id")),
            )
            if source_key in seen_sources:
                continue

            seen_sources.add(source_key)
            sources.append({
                "index": index,
                "source": source,
                "page": page,
                "score": float(doc.get("similarity_score", 0.0)),
                "preview": content[:120],
            })

            context_blocks.append(
                f"[Source {index}]\n"
                f"File: {source}\n"
                f"Page: {page}\n"
                f"Content:\n{content}"
            )

        if not context_blocks:
            answer = "No relevant context was found in the document collection."

        else:
            context = "\n\n".join(context_blocks)
            prompt = f"""
                You are a document question-answering assistant.

                Answer the question using only the provided context.

                Rules:
                - Do not use outside knowledge.
                - Do not invent facts.
                - If the context does not contain the answer, say:
                    "The documents do not contain enough information to answer this."
                - Keep the answer concise and factual.
                - Cite supporting sources using [Source 1], [Source 2], etc.

                Context:
                {context}

                Question:
                {question}

                Answer:
            """

            response = self.llm.invoke(prompt)
            answer = getattr(
                response,
                "content",
                str(response),
            ).strip()

        citations = [
            f"[Source {source['index']}] "
            f"{source['source']} "
            f"(page {source['page']})"
            for source in sources
        ]

        final_answer = answer
        if citations:
            final_answer += "\n\nSources:\n" + "\n".join(citations)

        history_item = {
            "question": question,
            "answer": final_answer,
            "sources": sources,
        }

        self.history.append(history_item)
        return {
            "question": question,
            "answer": answer,
            "status": "insufficient_context" if not results else "success",
            "sources": [
                {
                    "index": i + 1,
                    "source": src["source"],
                    "page": src["page"],
                    "similarity_score": round(src["score"], 4),
                    "preview": src["preview"]
                }
                for i, src in enumerate(sources)
            ],
            "history": self.history
        }
