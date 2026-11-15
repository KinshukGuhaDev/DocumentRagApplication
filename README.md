# 🧠 Enterprise Document RAG Application

> An end-to-end, production-grade **Retrieval-Augmented Generation (RAG)** system featuring multi-format document parsing, intelligent file-aware chunking (7 strategies including Hybrid and Agentic), dual local/cloud embedding engines, dense+lexical hybrid reranking, and hallucination-resistant LLM synthesis.

[![Python Version](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/LangChain-0.3.30-darkgreen.svg?logo=chainlink&logoColor=white)](https://python.langchain.com/)
[![ChromaDB](https://img.shields.io/badge/VectorDB-ChromaDB-orange.svg)](https://www.trychroma.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📌 Table of Contents

- [Overview & Outcomes](#-overview--outcomes)
- [End-to-End System Architecture](#-end-to-end-system-architecture)
- [Key Features](#-key-features)
- [Installation Guide](#-installation-guide)
- [Configuration Settings](#-configuration-settings)
- [Behind the Scenes: Deep-Dive Diagrams](#-behind-the-scenes-deep-dive-diagrams)
  - [1. The Ingestion Pipeline](#1-the-ingestion-pipeline)
  - [2. The Retrieval & Synthesis Pipeline](#2-the-retrieval--synthesis-pipeline)
- [Usage & Running the Project](#-usage--running-the-project)
  - [Running the Ingestion Pipeline](#step-1-running-the-ingestion-pipeline)
  - [Running the Retrieval Pipeline](#step-2-running-the-retrieval-pipeline)
  - [Programmatic Python API](#step-3-programmatic-python-api)
- [Project Directory Structure](#-project-directory-structure)
- [Troubleshooting & FAQs](#-troubleshooting--faqs)

---

## 🎯 Overview & Outcomes

Many traditional RAG systems suffer from two fatal flaws:
1. **Context Fragmentation**: Tabular rows (CSV, Excel) and structured data (JSON, SQL) get split by arbitrary character delimiters, corrupting data integrity.
2. **Retrieval Blindspots & Hallucinations**: Pure semantic vector search often misses specific entity names, keywords, or exact ID numbers, and standard LLM prompts invent facts when relevant context is absent.

This application is engineered specifically to eliminate these weaknesses.

### What is the Outcome of This Project?
- **Universal Multi-Format Ingestion**: Ingests unstructured documents (PDF, DOCX, TXT), tabular sheets (CSV, Excel), hierarchical data (JSON), and relational databases (SQL) while preserving structural integrity.
- **Intelligent Chunking with Zero Context Loss**: Automatically applies atomic row extraction for spreadsheets/tables and semantic/sentence packing for natural prose.
- **High-Recall Hybrid Reranking**: Combines **70% Dense Semantic Similarity** (vector embeddings) with **30% Lexical Keyword Overlap** (exact keyword & phrase matching) over an oversampled candidate pool.
- **Strict Anti-Hallucination Answers**: Synthesizes verified answers using **Google Gemini** or **OpenAI GPT** with zero temperature and mandatory provenance citations (`[Source 1] file.pdf (page 4)`).
- **Persistent Vector Store**: Stores vectors locally in **ChromaDB** using cosine similarity with automatic dimension checking and idempotent SHA-256 upserting.

---

## 🏗️ End-to-End System Architecture

The following animated flow diagram illustrates the end-to-end lifecycle of a document from raw bytes to vector storage, and how a user query traverses the hybrid search and generation stages:

<p align="center">
  <img src="assets/images/rag-system-overview.svg" alt="Document RAG System End-to-End Architecture" width="100%" />
</p>

The system operates across two decoupled, asynchronous phases:
1. **Document Ingestion Phase**: Raw documents are validated, parsed with format-specialized loaders, chunked using token-safe strategies, normalized into vector embeddings, and committed to ChromaDB.
2. **Retrieval & Synthesis Phase**: User queries are embedded, oversampled across the vector index, reranked via dense+lexical weights, injected into a strict grounding prompt, and answered by the LLM with exact source citations.

---

## 🚀 Key Features

| Capability | Implementation Details |
| :--- | :--- |
| **Multi-Format Parsers** | Specialized loaders for **PDF** (`PyMuPDF`), **DOCX** (`docx2txt`), **TXT** (UTF-8), **CSV** (`csv.DictReader`), **Excel** (`openpyxl`), **JSON** (recursive dictionary traversal), and **SQL** (statement boundary parser). |
| **7 Chunking Strategies** | `hybrid` (file-type aware, atomic records vs prose), `semantic` (cosine boundary drops), `agentic` (LLM-driven boundary detection), `sentence-based` (`spaCy`), `structured` (Markdown headers), `recursive`, and `fixed-size`. |
| **Chunk Quality Controls** | `_healthy_chunks()` merges tiny orphan fragments (< 20% chunk size), `_apply_overlap()` enforces 15% whole-word sliding overlap, and token length is tracked with `tiktoken` (`cl100k_base`). |
| **Dual Embedding Backends** | Run locally offline with **Sentence-Transformers** (`all-MiniLM-L6-v2`, 384 dims) or cloud-scale with **OpenAI** (`text-embedding-3-small`, 1536 dims). Includes pre-flight token limit checks and L2 normalization. |
| **ChromaDB Store & Guardrails** | Local persistent store (`Database/vector_store`) configured with HNSW Cosine distance. Employs a dimension guardrail to prevent collection corruption and SHA-256 idempotent upserts. |
| **Hybrid Search & Reranking** | Pre-fetches an oversampled candidate pool ($K \times 4$), reranks by `0.70 × Cosine Similarity + 0.30 × Lexical Overlap`, adds a +0.20 phrase bonus, and cuts off low scores via `SCORE_THRESHOLD`. |
| **Grounded LLM Generation** | Zero-temperature synthesis via **Google Gemini** (`gemini-3.5-flash-lite`) or **OpenAI** (`gpt-4.1-mini`) with automatic fallback to *"The documents do not contain enough information to answer this."* |

---

## 💻 Installation Guide

### Prerequisites
- **Python 3.10+** (Python 3.10, 3.11, or 3.12 recommended)
- **Git**
- A **Google Gemini API Key** or **OpenAI API Key**

### 1. Clone the Repository
```bash
git clone <repository-url>
cd RagApplication
```

### 2. Create and Activate a Virtual Environment

**On Linux / macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

**On Windows (PowerShell):**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**On Windows (Command Prompt):**
```cmd
python -m venv .venv
.venv\Scripts\activate.bat
```

### 3. Install Dependencies
```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 4. Install the spaCy Language Model
The `sentence-based` and `hybrid` chunking strategies utilize spaCy's `en_core_web_sm` model for grammatical sentence segmentation:
```bash
python -m spacy download en_core_web_sm
```

---

## ⚙️ Configuration Settings

All pipeline configurations are centralized in [`config.py`](config.py) and environment variables are loaded from `.env`.

### 1. Environment Variables (`.env`)
Create a `.env` file in the project root:
```env
# Optional if using Google Gemini
GEMINI_API_KEY="your-google-gemini-api-key"

# Optional if using OpenAI
OPENAI_API_KEY="your-openai-api-key"
```

### 2. Configuration Parameters in `config.py`

| Parameter | Default Value | Options / Description |
| :--- | :--- | :--- |
| `DIRECTORY_PATH` | `"Datasets/pdf_files"` | Path to the directory holding documents to ingest. |
| `FILE_TYPE` | `"pdf"` | `"pdf"`, `"docx"`, `"txt"`, `"csv"`, `"json"`, `"xlsx"`, `"xls"`, `"sql"`, or `"all"`. |
| `IS_SINGLE_FILE` | `True` | `True` to ingest one specific file; `False` to scan `DIRECTORY_PATH` recursively. |
| `SINGLE_FILE_PATH`| `"Datasets/pdf_files/01_the_future_of_vertical_forests.pdf"` | Path to target file when `IS_SINGLE_FILE = True`. |
| `CHUNKING_STRATEGY` | `"hybrid"` | `"hybrid"`, `"semantic"`, `"agentic"`, `"sentence-based"`, `"structured"`, `"recursive"`, `"fixed-size"`. |
| `CHUNK_SIZE` | `None` (defaults to 220 tokens) | Target maximum tokens per chunk. Set to `None` for strategy default. |
| `SEMANTIC_THRESHOLDS` | `{"sentence-transformers": 0.65, "openai": 0.80}` | Cosine similarity threshold for semantic topic transitions. |
| `EMBEDDING_MODEL_TYPE` | `"sentence-transformers"` | `"sentence-transformers"` (local) or `"openai"` (cloud API). |
| `EMBEDDING_MODEL_NAME` | `"all-MiniLM-L6-v2"` | `"all-MiniLM-L6-v2"` (384-dim) or `"text-embedding-3-small"` (1536-dim). |
| `LLM_MODEL_TYPE` | `"google"` | `"google"` (Gemini) or `"openai"` (GPT). |
| `LLM_MODEL_NAME` | `"gemini-3.5-flash-lite"` | `"gemini-3.5-flash-lite"`, `"gpt-4.1-mini"`, or another supported model. |
| `LLM_MODEL_TEMPERATURE` | `0` | Controls randomness (0 = deterministic, factual). |
| `VECTOR_DATABASE` | `"chromadb"` | Vector database provider. |
| `VECTOR_STORE_PATH` | `"Database/vector_store"` | Local disk directory where ChromaDB stores indexes and SQLite files. |
| `VECTOR_STORE_COLLECTION_NAME` | `"pdf_documents_cosine_v2"` | ChromaDB collection name. |
| `TOP_K` | `3` | Number of top retrieved chunks passed to the LLM. |
| `SCORE_THRESHOLD` | `0.5` | Minimum combined hybrid score required for a chunk to be admitted. |

---

## 🔍 Behind the Scenes: Deep-Dive Diagrams

### 1. The Ingestion Pipeline

The ingestion pipeline converts disparate file formats into vector representations without data corruption:

<p align="center">
  <img src="assets/images/injestion-pipeline-diagram.svg" alt="Document Ingestion Pipeline Deep Dive" width="100%" />
</p>

#### How It Works Step-by-Step:
1. **Dynamic Loading (`LoadFiles.py`)**:
   - Discovers files and validates extensions against allowed types.
   - Emits standardized `Document` objects stamped with `source_file`, `source_path`, `file_type`, and a cryptographic `document_id = SHA256(canonical_path)`.
   - Special handling converts tabular rows into descriptive key-value records (`Column: Value`), JSON into structural nodes with JSONPath pointers, and SQL into intact statements.
2. **Structural Chunking Engine (`DataChunking.py`)**:
   - Routes documents through the chosen strategy. In `hybrid` mode, structured rows (CSV, JSON, Excel) remain atomic chunks to preserve tabular meaning.
   - Prose documents (PDF, DOCX, TXT) undergo sentence segmentation, packing into 220-token windows, and 15% sliding whole-word overlap.
   - `_healthy_chunks()` merges under-sized text fragments into predecessor chunks.
   - Generates deterministic `chunk_id = SHA256(source_file|page|strategy|text)`.
3. **Batched Embedding Generation (`EmbeddingManager.py`)**:
   - Pre-flight check verifies that chunk token lengths do not exceed model sequence limits.
   - Encodes text into vectors in batches of 32 chunks.
   - Applies L2 vector normalization ($\|v\|_2 = 1$) so that cosine similarity reduces to a fast dot product.
   - Asserts non-zero dimensions and checks for any `NaN` or infinite values.
4. **ChromaDB Upsert (`VectorStore.py`)**:
   - Queries `_existing_dimension()`. If the collection already contains vectors of a different dimension (e.g. 384 vs 1536), an explicit error is raised before writing.
   - Flattens and sanitizes metadata types for ChromaDB compatibility.
   - Executes `collection.upsert()`. Re-running ingestion updates existing IDs without creating duplicate records.

---

### 2. The Retrieval & Synthesis Pipeline

The retrieval pipeline finds relevant context and ensures that generated responses are strictly grounded in document facts:

<p align="center">
  <img src="assets/images/retrieval-pipeline-diagram.svg" alt="Retrieval and Synthesis Pipeline Deep Dive" width="100%" />
</p>

#### How It Works Step-by-Step:
1. **Query Vectorization**:
   - Validates that the user input is a non-empty string and normalizes it.
   - Encodes the query into a unit-normalized vector matching the collection's embedding space.
   - Verifies that the active collection's stored model metadata matches the current query model.
2. **Oversampled Candidate Retrieval**:
   - Instead of retrieving only `TOP_K` (3) chunks, it queries ChromaDB for a larger candidate pool of $\min(\max(K \times 4, 12), \text{total})$.
   - This prevents dense vector search from omitting specific entity names or exact numbers that rank lower on pure cosine distance.
3. **Hybrid Dense + Lexical Reranker**:
   - Calculates **Dense Similarity**: $\text{similarity\_score} = 1.0 - \text{cosine\_distance}$.
   - Calculates **Lexical Overlap**: Ratio of query keywords present in the chunk, plus a $+0.20$ bonus if the verbatim query string appears.
   - Computes the combined score:
     $$\text{Rerank Score} = 0.70 \times \text{Similarity Score} + 0.30 \times \text{Lexical Score}$$
   - Filters out chunks below `SCORE_THRESHOLD` (0.50) and selects the top `TOP_K` candidates.
4. **Anti-Hallucination Prompting & Synthesis**:
   - Constructs structured context blocks with file and page numbers.
   - Instructs the LLM with strict negative constraints (*"Do not use outside knowledge. Do not invent facts. If the context does not contain the answer, say: 'The documents do not contain enough information to answer this.' "*).
   - Generates answers at zero temperature and appends bibliography citations.
   - Appends the turn to `self.history` for multi-turn conversational tracking.

---

## 💻 Usage & Running the Project

### Step 1: Running the Ingestion Pipeline

To load your documents, split them into chunks, generate embeddings, and populate ChromaDB:

```bash
python injestion_pipeline.py
```
*(Alternatively: `python main.py`)*

#### Example Ingestion Console Output:
```text
__________________________________________________

Loading files...
--------------------------------------------------
🔃 Processing file: 01_the_future_of_vertical_forests.pdf from: Datasets/pdf_files
✅ Loaded 4 documents from 01_the_future_of_vertical_forests.pdf.
--------------------------------------------------
✅ Loaded 4 documents from 1 files.
__________________________________________________


Creating chunks...
__________________________________________________
🔄 Started Generating chunks using hybrid chunking strategy
✅ Processed 4 source documents and generated 14 chunks.
__________________________________________________


Embedding chunks...
__________________________________________________
🔄 Loading embedding model: all-MiniLM-L6-v2
✅ Loaded all-MiniLM-L6-v2. Embedding dimension: 384
🔄 Generating embeddings for 14 chunks...
Batches: 100%|█████████████████████████████████████████| 1/1 [00:00<00:00,  8.21it/s]
✅ Generated embeddings. Shape: (14, 384)
__________________________________________________


Vector Storing...
__________________________________________________
✅ ChromaDB initialized with collection 'pdf_documents_cosine_v2'
📁 Storage path: /workspace/RagApplication/Database/vector_store
📊 Existing records: 0
✅ Upserted 14 chunks. Collection now contains 14 records.
__________________________________________________
```

---

### Step 2: Running the Retrieval Pipeline

To ask questions against your ingested documents via the interactive CLI:

```bash
python retrieval_pipeline.py
```

#### Example Interactive Session:
```text
Enter your question: What are the primary ecological benefits of vertical forests?
```

**Output:**
```json
{
  "question": "What are the primary ecological benefits of vertical forests?",
  "answer": "Vertical forests absorb significant amounts of carbon dioxide, produce oxygen, filter airborne particulate matter, and reduce the urban heat island effect by providing natural shade and transpiration [Source 1]. They also foster urban biodiversity by providing habitats for birds and pollinating insects [Source 2].\n\nSources:\n[Source 1] 01_the_future_of_vertical_forests.pdf (page 2)\n[Source 2] 01_the_future_of_vertical_forests.pdf (page 3)",
  "status": "success",
  "sources": [
    {
      "index": 1,
      "source": "01_the_future_of_vertical_forests.pdf",
      "page": 2,
      "similarity_score": 0.8412,
      "preview": "Vertical forests act as green lungs in dense urban environments. A single tower can absorb up to 30 tons of CO2..."
    },
    {
      "index": 2,
      "source": "01_the_future_of_vertical_forests.pdf",
      "page": 3,
      "similarity_score": 0.7935,
      "preview": "Biodiversity monitoring revealed over 20 species of birds and pollinators nesting within the facade vegetation..."
    }
  ]
}
```

---

### Step 3: Programmatic Python API

You can integrate the ingestion and retrieval components directly into web services (such as FastAPI, Flask, or Streamlit):

```python
from Classes.RAGRetriever import RAGRetriever

# Initialize retriever (automatically connects to configured ChromaDB & LLM)
retriever = RAGRetriever(top_k=3, score_threshold=0.5)

# Query the knowledge base
response = retriever.retrieve("What was the net profit for Q3?")

# Access structured fields
print("Answer:", response["answer"])
print("Status:", response["status"])
for source in response["sources"]:
    print(f"- {source['source']} (Page {source['page']}): Score {source['similarity_score']}")
```

---

## 📂 Project Directory Structure

```text
RagApplication/
├── assets/
│   └── images/
│       ├── rag-system-overview.svg          # Animated End-to-End Architecture diagram
│       ├── injestion-pipeline-diagram.svg   # Animated Ingestion Pipeline diagram
│       └── retrieval-pipeline-diagram.svg   # Animated Retrieval & Reranker diagram
├── Classes/
│   ├── LoadFiles.py                         # Multi-format document parser & metadata stamper
│   ├── DataChunking.py                      # 7 Chunking strategies, token packing & overlap
│   ├── EmbeddingManager.py                  # Dual embedding provider (SentenceTransformers/OpenAI)
│   ├── VectorStore.py                       # ChromaDB persistent storage & dimension guardrails
│   └── RAGRetriever.py                      # Hybrid dense+lexical reranker & LLM synthesis
├── config.py                                # Central pipeline configuration & hyperparameters
├── commands.txt                             # Quick reference setup commands
├── requirements.txt                         # Pinned Python package dependencies
├── main.py                                  # Primary script executing the ingestion pipeline
├── injestion_pipeline.py                    # Dedicated ingestion execution entry point
├── retrieval_pipeline.py                    # Interactive query CLI interface
├── Datasets/                                # Sample datasets for benchmarking
│   ├── pdf_files/                           # PDF documents
│   ├── docx_files/                          # Microsoft Word documents
│   ├── excel_files/                         # Excel spreadsheets (.xlsx, .xls)
│   ├── text_files/                          # Plain text documents (.txt)
│   └── sql_files/                           # Relational SQL dump files (.sql)
├── Database/
│   └── vector_store/                        # ChromaDB SQLite & HNSW index files
└── Notebooks/
    ├── Document.ipynb                       # Exploratory analysis notebook
    └── Ingestion_Pipelines/
        └── pdf_loader.ipynb                 # Interactive PDF loading notebook
```

---

## ❓ Troubleshooting & FAQs

### Q1: `ValueError: Query embedding dimension does not match the Chroma collection`
- **Cause**: You switched embedding models in `config.py` (e.g. from `all-MiniLM-L6-v2` with 384 dimensions to `text-embedding-3-small` with 1536 dimensions) while keeping the same collection name.
- **Fix**: Either update `VECTOR_STORE_COLLECTION_NAME` in `config.py` to a new name (e.g. `"pdf_documents_openai_v1"`) or delete the old index in `Database/vector_store` and re-run ingestion.

### Q2: `GEMINI_API_KEY is required for Google embeddings` or `OPENAI_API_KEY is required`
- **Cause**: The API key is missing from your `.env` file or environment.
- **Fix**: Create a `.env` file in the root folder with `GEMINI_API_KEY="your_key"` or `OPENAI_API_KEY="your_key"`.

### Q3: `Can't find model 'en_core_web_sm'`
- **Cause**: The spaCy English language model has not been downloaded into your virtual environment.
- **Fix**: Run:
  ```bash
  python -m spacy download en_core_web_sm
  ```

### Q4: How do I ingest an entire directory instead of a single file?
- **Fix**: In [`config.py`](config.py), set:
  ```python
  IS_SINGLE_FILE = False
  DIRECTORY_PATH = "Datasets/pdf_files"  # or any folder path
  FILE_TYPE = "all"                     # or "pdf", "docx", etc.
  ```
  Then run `python injestion_pipeline.py`.

---

<p align="center">
  Built with ❤️ for enterprise-grade, hallucination-free document retrieval.
</p>
