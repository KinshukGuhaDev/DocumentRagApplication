import os
from dotenv import load_dotenv

load_dotenv()

# File Configurations
DIRECTORY_PATH = "Datasets/pdf_files"
FILE_TYPE = "pdf" # Options: "pdf", "docx", "txt", "csv", "json", "sql"
IS_SINGLE_FILE = True
SINGLE_FILE_PATH = "Datasets/pdf_files/gpt3_models.pdf"

#Chunking Configurations
CHUNKING_STRATEGY = "agentic" # Options: "sentence-based", "agentic", "recursive", "fixed-size", "semantic", "structured"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
MIN_CHUNK_SIZE = 200
SEMANTIC_THRESHOLD = 0.65

#Model Configurations
EMBEDDING_MODEL_TYPE = "sentence-transformers" # Options: "sentence-transformers", "openai"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"  # Options: "all-MiniLM-L6-v2", "langchain-openai"
LLM_MODEL_TYPE="google" # Options: "google", "openai"
LLM_MODEL_NAME="gemini-3.5-flash-lite" # Options: "gemini-3.5-flash-lite", "gpt-4.1-mini"
LLM_MODEL_TEMPERATURE = 0
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Vector Store Configurations
VECTOR_DATABASE = "chromadb"
VECTOR_STORE_PATH = "Database/vector_store"
VECTOR_STORE_COLLECTION_NAME = "pdf_documents_cosine_v1"

# Retrieval Configurations
TOP_K = 3
SCORE_THRESHOLD = 0.5


