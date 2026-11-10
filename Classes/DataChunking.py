from config import CHUNKING_STRATEGY, CHUNK_SIZE, CHUNK_OVERLAP, SEMANTIC_THRESHOLD, MIN_CHUNK_SIZE, EMBEDDING_MODEL_NAME, EMBEDDING_MODEL_TYPE, LLM_MODEL_NAME, LLM_MODEL_TYPE, OPENAI_API_KEY, GEMINI_API_KEY, LLM_MODEL_TEMPERATURE
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import uuid
import re
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from tqdm.auto import tqdm
from pydantic import BaseModel, Field

class DataChunking:
    def __init__(
        self,
        chunking_strategy:str = CHUNKING_STRATEGY,
        chunk_size:int = CHUNK_SIZE,
        chunk_overlap:int = CHUNK_OVERLAP,
        min_chunk_size:int = MIN_CHUNK_SIZE,
        embedding_model_name:str = EMBEDDING_MODEL_NAME,
        embedding_model_type:str = EMBEDDING_MODEL_TYPE,
        semantic_threshold:float = SEMANTIC_THRESHOLD,
        llm_model_name:str = LLM_MODEL_NAME,
        llm_model_type:str = LLM_MODEL_TYPE,
        llm_model_temperature:float = LLM_MODEL_TEMPERATURE,
        openai_api_key:str = OPENAI_API_KEY,
        gemini_api_key:str = GEMINI_API_KEY,
    ):
        self.chunking_strategy = chunking_strategy
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.semantic_threshold = semantic_threshold
        self.min_chunk_size = min(
            max(0, min_chunk_size),
            chunk_size,
        )
        self.strategies = {
            "sentence-based": self.sentence_based_chunking,
            "agentic": self.agentic_chunking,
            "recursive": self.recursive_chunking,
            "fixed-size": self.fixed_size_chunking,
            "semantic": self.semantic_chunking,
            "structured": self.structured_chunking,
        }
        self.embedding_model_name = embedding_model_name
        self.embedding_model_type = embedding_model_type
        self.llm_model_name = llm_model_name
        self.llm_model_type = llm_model_type
        
        if self.llm_model_type == "openai":
            self.llm = ChatOpenAI(
                model=self.llm_model_name,
                temperature=llm_model_temperature,
                api_key=openai_api_key,
            )
        elif self.llm_model_type == "google":
            self.llm = ChatGoogleGenerativeAI(
                model=self.llm_model_name,
                google_api_key=gemini_api_key,
            )
        else:
            raise ValueError(
                f"Unsupported LLM model type: {self.llm_model_type}"
            )


        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than 0")

        if not 0 <= chunk_overlap < chunk_size:
            raise ValueError(
                "chunk_overlap must be >= 0 and < chunk_size"
            )

        if not -1 <= semantic_threshold <= 1:
            raise ValueError(
                "semantic_threshold must be between -1 and 1"
            )

    def split_documents_into_chunks(self, documents):
        print("_" * 50)
        print(
            f"🔄 Started Generating chunks using "
            f"{self.chunking_strategy} chunking strategy"
        )
        self.documents = documents
        chunking_method = self.strategies.get(self.chunking_strategy)
        if chunking_method is None:
            raise ValueError(
                f"Unsupported chunking strategy: {self.chunking_strategy}"
            )

        split_documents = chunking_method()

        print(
            f"✅ Generated {len(documents)} documents "
            f"into {len(split_documents)} chunks."
        )
        print("_" * 50)
        print("\n")
        return split_documents

    def _create_documents(self, source_doc, chunks, strategy):
        """
        Preserve source metadata and attach chunk metadata.
        """
        result = []

        for index, text in enumerate(chunks):
            if not text.strip():
                continue

            metadata = dict(source_doc.metadata)
            metadata.update({
                "chunk_id": str(uuid.uuid4()),
                "chunk_index": index,
                "chunk_strategy": strategy,
                "content_length": len(text),
            })

            result.append(
                Document(
                    page_content=text,
                    metadata=metadata,
                )
            )

        return result

    def _get_sentences(self, text):
        """
        Lightweight sentence segmentation.
        """
        return [
            sentence.strip()
            for sentence in re.split(
                r'(?<=[.!?])\s+|\n+',
                text.strip(),
            )
            if sentence.strip()
        ]

    def _pack_text(self, units):
        """
        Pack sentences or topic groups into chunks
        without exceeding chunk_size.

        Oversized units are recursively split.
        """
        chunks = []
        current = ""

        for unit in units:
            unit = unit.strip()

            if not unit:
                continue

            if len(unit) > self.chunk_size:
                if current:
                    chunks.append(current)
                    current = ""

                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=self.chunk_size,
                    chunk_overlap=0,
                    separators=["\n\n", "\n", " ", ""],
                )

                chunks.extend(splitter.split_text(unit))
                continue

            candidate = (
                f"{current} {unit}" if current else unit
            )

            if len(candidate) <= self.chunk_size:
                current = candidate
            else:
                if current:
                    chunks.append(current)

                current = unit

        if current:
            chunks.append(current)

        return chunks

    def _apply_overlap(self, chunks):
        """
        Add word-boundary overlap without
        exceeding chunk_size.
        """

        if self.chunk_overlap <= 0:
            return chunks

        result = []
        for index, chunk in enumerate(chunks):
            if index == 0:
                result.append(chunk)
                continue

            available = self.chunk_size - len(chunk)

            if available <= 1:
                result.append(chunk)
                continue

            max_overlap = min(
                self.chunk_overlap,
                available - 1
            )

            previous_words = chunks[index - 1].split()
            overlap_words = []
            overlap_length = 0

            for word in reversed(previous_words):
                added_length = len(word) + (
                    1 if overlap_words else 0
                )

                if overlap_length + added_length > max_overlap:
                    break

                overlap_words.insert(0, word)
                overlap_length += added_length

            if overlap_words:
                prefix = " ".join(overlap_words)
                result.append(f"{prefix} {chunk}")
            else:
                result.append(chunk)

        return result

    def _get_embedding_model(self):
        """
        Lazy-load the embedding model.
        """
        if self.embedding_model_type == "sentence-transformers":
            self.embedding_model = SentenceTransformer(
                self.embedding_model_name
            )
        else:
            raise ValueError(
                f"Unknown embedding model type: {self.embedding_model_type}"
            )

        return self.embedding_model
    
    def fixed_size_chunking(self):
        """
        Exact character-window chunking.

        Uses fixed-size windows with configurable
        character overlap.
        """
        all_chunks = []        
        step = self.chunk_size - self.chunk_overlap

        for doc in self.documents:
            text = doc.page_content
            chunks = []
            for start in range(0, len(text), step):
                chunk = text[start:start + self.chunk_size]
                if chunk:
                    chunks.append(chunk)

                if start + self.chunk_size >= len(text):
                    break

            all_chunks.extend(
                self._create_documents(
                    doc,
                    chunks,
                    "fixed-size",
                )
            )

        return all_chunks

    def recursive_chunking(self):
        """
        Hierarchical character splitting:

        Paragraph -> Line -> Word -> Character
        """
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""],
        )

        all_chunks = []

        for doc in self.documents:
            chunks = splitter.split_text(
                doc.page_content
            )

            all_chunks.extend(
                self._create_documents(
                    doc,
                    chunks,
                    "recursive",
                )
            )

        return all_chunks

    def sentence_based_chunking(self):
        """
        Split into sentences and group them into
        dynamically sized chunks.
        """
        all_chunks = []
        for doc in self.documents:
            sentences = self._get_sentences(
                doc.page_content
            )
            chunks = self._pack_text(sentences)
            chunks = self._apply_overlap(chunks)
            all_chunks.extend(
                self._create_documents(
                    doc,
                    chunks,
                    "sentence-based",
                )
            )

        return all_chunks

    def structured_chunking(self, documents):
        """
        Split based on detected headings and sections.

        Supports:
        - Markdown headings
        - Chapter headings
        - Section headings
        - Numbered headings

        This operates on extracted plain text.
        """
        all_chunks = []

        heading_pattern = re.compile(
            r"^(?:"
            r"#{1,6}\s+.+"
            r"|(?:Chapter|Section)\s+\d+.*"
            r"|\d+(?:\.\d+)*[.)]?\s+.+"
            r")$",
            re.IGNORECASE,
        )

        for doc in documents:
            sections = []
            current_lines = []

            for line in doc.page_content.splitlines():
                stripped = line.strip()

                if (
                    stripped
                    and heading_pattern.match(stripped)
                    and current_lines
                ):
                    sections.append(
                        "\n".join(current_lines).strip()
                    )
                    current_lines = []

                current_lines.append(line)

            if current_lines:
                sections.append(
                    "\n".join(current_lines).strip()
                )

            chunks = []

            for section in sections:
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=self.chunk_size,
                    chunk_overlap=self.chunk_overlap,
                    separators=["\n\n", "\n", " ", ""],
                )

                chunks.extend(
                    splitter.split_text(section)
                )

            all_chunks.extend(
                self._create_documents(
                    doc,
                    chunks,
                    "structured",
                )
            )

        return all_chunks
        
    def agentic_chunking(self):
        """
        LLM-assisted topic boundary detection
        with progress tracking.
        """

        if self.llm is None:
            print(
                "⚠️ No LLM configured. "
                "Falling back to recursive chunking."
            )
            return self.recursive_chunking(self.documents)

        class TopicBoundaries(BaseModel):
            break_before: list[int] = Field(
                description=(
                    "Zero-based sentence indices where "
                    "a new topic begins. Exclude 0."
                )
            )

        structured_llm = self.llm.with_structured_output(
            TopicBoundaries
        )

        all_chunks = []
        max_window_chars = 6000

        # STEP 1: Prepare windows for all documents
        prepared_documents = []

        for doc in self.documents:

            sentences = self._get_sentences(
                doc.page_content
            )

            if not sentences:
                continue

            windows = []
            current_window = []
            current_length = 0

            for sentence in sentences:

                if (
                    current_window
                    and current_length + len(sentence) + 1
                    > max_window_chars
                ):
                    windows.append(current_window)
                    current_window = []
                    current_length = 0

                current_window.append(sentence)
                current_length += len(sentence) + 1

            if current_window:
                windows.append(current_window)

            prepared_documents.append((doc, windows))

        # STEP 2: Count total API calls
        total_windows = sum(
            len(windows)
            for _, windows in prepared_documents
        )

        print(
            f"🚀 Processing {len(prepared_documents)} documents "
            f"across {total_windows} windows..."
        )

        # STEP 3: Start progress bar
        with tqdm(
            total=total_windows,
            desc="🤖 Agentic Chunking",
            unit="window",
            dynamic_ncols=True,
            colour="green",
        ) as progress:

            for doc, windows in prepared_documents:

                chunks = []

                for window in windows:

                    numbered_text = "\n".join(
                        f"{i}: {sentence}"
                        for i, sentence in enumerate(window)
                    )

                    prompt = f"""
                        You are analyzing a document for RAG chunking.

                        Identify the sentence indices where a new
                        logical topic begins.

                        Rules:
                        - Return only boundary indices.
                        - Do not rewrite the document.
                        - Do not include index 0.
                        - Prefer meaningful topic changes.
                        - Avoid unnecessary fragmentation.

                        Document sentences:
                        {numbered_text}
                    """

                    # Gemini API call
                    response = structured_llm.invoke(prompt)

                    boundaries = sorted({
                        i for i in response.break_before
                        if type(i) is int
                        and 0 < i < len(window)
                    })

                    starts = [0] + boundaries
                    ends = boundaries + [len(window)]

                    for start, end in zip(starts, ends):

                        topic_text = " ".join(
                            window[start:end]
                        )

                        chunks.extend(
                            self._pack_text([topic_text])
                        )

                    # Update after successful API call
                    progress.update(1)

                    progress.set_postfix(
                        chunks=len(all_chunks) + len(chunks)
                    )

                chunks = self._apply_overlap(chunks)

                all_chunks.extend(
                    self._create_documents(
                        doc,
                        chunks,
                        "agentic",
                    )
                )

        print(
            f"✅ Agentic chunking completed: "
            f"{len(all_chunks)} chunks generated."
        )

        return all_chunks
    
    def semantic_chunking(self):
        """
        Uses sentence embeddings and cosine similarity.

        Creates a new topic group when adjacent
        sentence similarity falls below threshold.
        """
        all_chunks = []
        model = self._get_embedding_model()

        for doc in self.documents:
            sentences = self._get_sentences(
                doc.page_content
            )

            if not sentences:
                continue

            if len(sentences) == 1:
                chunks = self._pack_text(sentences)

            else:
                embeddings = model.encode(
                    sentences,
                    batch_size=32,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )

                similarities = np.sum(
                    embeddings[:-1] * embeddings[1:],
                    axis=1,
                )

                groups = []
                current = [sentences[0]]

                for index, similarity in enumerate(similarities):
                    next_sentence = sentences[index + 1]
                    current_text = " ".join(current)

                    topic_changed = (
                        float(similarity)
                        < self.semantic_threshold
                    )

                    size_exceeded = (
                        len(current_text)
                        + len(next_sentence)
                        + 1
                        > self.chunk_size
                    )

                    if (
                        topic_changed
                        and len(current_text) >= self.min_chunk_size
                    ) or size_exceeded:
                        groups.append(" ".join(current))
                        current = [next_sentence]
                    else:
                        current.append(next_sentence)

                if current:
                    groups.append(" ".join(current))

                chunks = []

                for group in groups:
                    chunks.extend(
                        self._pack_text([group])
                    )

            chunks = self._apply_overlap(chunks)

            all_chunks.extend(
                self._create_documents(
                    doc,
                    chunks,
                    "semantic",
                )
            )

        return all_chunks