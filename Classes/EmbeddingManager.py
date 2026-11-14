import numpy as np
from typing import List, Union
from sentence_transformers import SentenceTransformer
from langchain_openai import OpenAIEmbeddings
from typing import List
from config import (
    EMBEDDING_MODEL_NAME,
    EMBEDDING_MODEL_TYPE,
    OPENAI_API_KEY
)
import tiktoken
from langchain_core.documents import Document
TextInput = Union[str, Document]

class EmbeddingManager:
    def __init__(
        self,
        model_type: str = EMBEDDING_MODEL_TYPE,
        model_name: str = EMBEDDING_MODEL_NAME,
        openai_api_key: str = OPENAI_API_KEY,
        batch_size: int = 32,
        normalize_embeddings: bool = True,
        max_input_tokens: int | None = None,
    ):
        self.model_type = model_type.lower().strip()
        self.model_name = model_name
        self.openai_api_key = openai_api_key
        self.batch_size = batch_size
        self.normalize_embeddings = normalize_embeddings
        self.configured_max_input_tokens = max_input_tokens

        self.model = None
        self.embedding_dimension = None
        self.tokenizer = None

        if self.batch_size <= 0:
            raise ValueError("batch_size must be greater than zero")

        if self.model_type not in {
            "sentence-transformers",
            "openai",
        }:
            raise ValueError(
                "model_type must be either "
                "'sentence-transformers' or 'openai'"
            )

        self._load_model()
    
    def _load_model(self):
        print("Embedding chunks...")
        print("_" * 50)
        print(f"🔄 Loading embedding model: {self.model_name}")

        if self.model_type == "sentence-transformers":
            self.model = SentenceTransformer(self.model_name)

            self.embedding_dimension = (
                self.model.get_embedding_dimension()
            )

            self.tokenizer = self.model.tokenizer

            print(
                f"✅ Loaded {self.model_name}. "
                f"Embedding dimension: {self.embedding_dimension}"
            )

        elif self.model_type == "openai":
            if not self.openai_api_key:
                raise ValueError(
                    "OPENAI_API_KEY is required for OpenAI embeddings"
                )

            self.model = OpenAIEmbeddings(
                model=self.model_name,
                openai_api_key=self.openai_api_key,
                max_retries=3,
            )

            try:
                self.tokenizer = tiktoken.encoding_for_model(
                    self.model_name
                )
            except KeyError:
                self.tokenizer = tiktoken.get_encoding(
                    "cl100k_base"
                )

            print(f"✅ Loaded {self.model_name}")

    def _extract_text(self, item: TextInput) -> str:
        if isinstance(item, Document):
            text = item.page_content
        elif isinstance(item, str):
            text = item
        else:
            raise TypeError(
                "Each input must be a string or LangChain Document"
            )

        text = text.strip()

        if not text:
            raise ValueError(
                "Embedding input cannot be empty or whitespace-only"
            )

        return text

    def _extract_texts(
        self,
        contents: List[TextInput],
    ) -> List[str]:
        if contents is None:
            raise ValueError("contents cannot be None")

        if not contents:
            raise ValueError("contents cannot be empty")

        return [
            self._extract_text(item)
            for item in contents
        ]

    def count_tokens(self, text: str) -> int:
        text = self._extract_text(text)

        if self.tokenizer is None:
            return 0

        if self.model_type == "sentence-transformers":
            return len(
                self.tokenizer.encode(
                    text,
                    add_special_tokens=False,
                )
            )

        return len(self.tokenizer.encode(text))

    def _get_max_input_tokens(self) -> int | None:
        if self.configured_max_input_tokens is not None:
            return self.configured_max_input_tokens

        if self.model_type == "sentence-transformers":
            max_seq_length = getattr(
                self.model,
                "max_seq_length",
                None,
            )

            if max_seq_length is not None:
                return int(max_seq_length)

        return None

    def _validate_input_lengths(
        self,
        texts: List[str],
    ):
        max_input_tokens = self._get_max_input_tokens()

        if max_input_tokens is None:
            return

        oversized = []

        for index, text in enumerate(texts):
            token_count = self.count_tokens(text)

            if token_count > max_input_tokens:
                oversized.append(
                    {
                        "index": index,
                        "tokens": token_count,
                        "limit": max_input_tokens,
                    }
                )

        if oversized:
            raise ValueError(
                "One or more chunks exceed the embedding model "
                f"input limit of {max_input_tokens} tokens: "
                f"{oversized[:5]}"
            )

    def _normalize(self, embeddings: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(
            embeddings,
            axis=1,
            keepdims=True,
        )

        return embeddings / np.maximum(norms, 1e-12)

    def _validate_embeddings(
        self,
        embeddings: np.ndarray,
        expected_count: int,
    ) -> np.ndarray:
        embeddings = np.asarray(
            embeddings,
            dtype=np.float32,
        )

        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)

        if embeddings.ndim != 2:
            raise ValueError(
                "Embeddings must be a two-dimensional matrix"
            )

        if embeddings.shape[0] != expected_count:
            raise ValueError(
                "Embedding count does not match input count: "
                f"{embeddings.shape[0]} != {expected_count}"
            )

        if not np.isfinite(embeddings).all():
            raise ValueError(
                "Embeddings contain NaN or infinite values"
            )

        if embeddings.shape[1] == 0:
            raise ValueError(
                "Embedding dimension cannot be zero"
            )

        self.embedding_dimension = embeddings.shape[1]

        return embeddings

    def generate_embeddings(self, contents: List[TextInput]) -> np.ndarray:
        texts = self._extract_texts(contents)

        self._validate_input_lengths(texts)

        print(
            f"🔄 Generating embeddings for "
            f"{len(texts)} chunks..."
        )

        if self.model_type == "sentence-transformers":
            embeddings = self.model.encode(
                texts,
                batch_size=self.batch_size,
                normalize_embeddings=self.normalize_embeddings,
                convert_to_numpy=True,
                show_progress_bar=len(texts) > 1,
            )

        elif self.model_type == "openai":
            embeddings = self.model.embed_documents(texts)

            embeddings = np.asarray(
                embeddings,
                dtype=np.float32,
            )

            if self.normalize_embeddings:
                embeddings = self._normalize(embeddings)

        embeddings = self._validate_embeddings(
            embeddings,
            expected_count=len(texts),
        )

        print(
            f"✅ Generated embeddings. "
            f"Shape: {embeddings.shape}"
        )
        print("_" * 50)
        print("\n")

        return embeddings

    def embed_query(self, query: str) -> np.ndarray:
        query = self._extract_text(query)

        self._validate_input_lengths([query])

        if self.model_type == "sentence-transformers":
            embedding = self.model.encode(
                [query],
                normalize_embeddings=self.normalize_embeddings,
                convert_to_numpy=True,
                show_progress_bar=False,
            )[0]

        elif self.model_type == "openai":
            embedding = self.model.embed_query(query)

            embedding = np.asarray(
                embedding,
                dtype=np.float32,
            )

            if self.normalize_embeddings:
                norm = np.linalg.norm(embedding)
                embedding = embedding / max(norm, 1e-12)

        embedding = np.asarray(
            embedding,
            dtype=np.float32,
        )

        if not np.isfinite(embedding).all():
            raise ValueError(
                "Query embedding contains NaN or infinite values"
            )

        if self.embedding_dimension is None:
            self.embedding_dimension = embedding.shape[0]

        if embedding.shape[0] != self.embedding_dimension:
            raise ValueError(
                "Query embedding dimension does not match "
                "document embedding dimension"
            )

        return embedding