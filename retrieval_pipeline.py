from Classes.RAGRetriever import RAGRetriever

rag_retriever = RAGRetriever()
query = input("Enter your question: ")
retrieved = rag_retriever.retrieve(query)
print(retrieved)