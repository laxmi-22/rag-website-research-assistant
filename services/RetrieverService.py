# Retriver Service
# used MMR serach -MMR improves coverage
# k=3 - number of dcouments to return

from typing import List
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

class RetrieverService:
    def __init__(self, vectorstore: Chroma, k: int = 3):
        self.retriever = vectorstore.as_retriever(
            search_type="mmr",  # Maximum Marginal Relevance
            search_kwargs={"k": k}
        )
    
    def get_relevant_documents(self, query: str) -> List[Document]:
        return self.retriever.get_relevant_documents(query)