# Vectore store service. 
# used chroma db for craeting vector store

from typing import List
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

class VectorstoreService:
    def __init__(self, embeddings):
        self.embeddings = embeddings
        self.vectorstore = None
    
    def create_vectorstore(self, documents: List[Document]) -> Chroma:
        self.vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings
        )
        
        print(f"[VectorStore] Vector store created with {len(documents)} chunks")
        return self.vectorstore