### This Script is used to dynamically load any filetype of documents with dynamic configurations.
from Classes.LoadFiles import LoadFiles
from Classes.DataChunking import DataChunking
from Classes.EmbeddingManager import EmbeddingManager
from Classes.VectorStore import VectorStore

loaded_files = LoadFiles().load_files()
chunks = DataChunking().split_documents_into_chunks(loaded_files)
embeddings = EmbeddingManager().generate_embeddings(chunks)
VectorStore().add_documents(chunks, embeddings)