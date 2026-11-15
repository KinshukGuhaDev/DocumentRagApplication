import os
from dotenv import load_dotenv

load_dotenv()

# File Configurations
DIRECTORY_PATH = "Datasets/pdf_files"
FILE_TYPE = "pdf" # Options: "pdf", "docx", "txt", "csv", "json", "sql"
IS_SINGLE_FILE = True
SINGLE_FILE_PATH = "Datasets/pdf_files/01_the_future_of_vertical_forests.pdf"

#Chunking Configurations
CHUNKING_STRATEGY = "hybrid" # Automatically selects a file-type-aware strategy
CHUNK_SIZE = None # Leave None to dynamically adjust chunk size based on the strategy
SEMANTIC_THRESHOLDS = {
    "sentence-transformers": 0.65,
    "openai": 0.80,
}

#Model Configurations
EMBEDDING_MODEL_TYPE = "sentence-transformers" # Options: "sentence-transformers", "openai"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"  # Options: "all-MiniLM-L6-v2", "text-embedding-3-small"
LLM_MODEL_TYPE="nvidia" # Options: "google", "openai", "nvidia"
LLM_MODEL_NAME="nemotron-3.5-super-vl-120b-a12b" # Options: "gemini-3.5-flash-lite", "gpt-4.1-mini", "nemotron-3.5-super-vl-120b-a12b"
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
LLM_MODEL_TEMPERATURE = 0                     
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
HF_HUB_DISABLE_SYMLINKS_WARNING=1
HF_HUB_DISABLE_PROGRESS_BARS=1

# Vector Store Configurations
VECTOR_DATABASE = "chromadb"
VECTOR_STORE_PATH = "Database/vector_store"
VECTOR_STORE_COLLECTION_NAME = "pdf_documents_cosine_v2"

# Retrieval Configurations
TOP_K = 3
SCORE_THRESHOLD = 0.5


