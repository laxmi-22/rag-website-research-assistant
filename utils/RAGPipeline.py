from services.WebLoaderService import WebLoaderService
from services.ChunkingService import ChunkingService
from services.EmbeddingService import EmbeddingService
from services.VectorstoreService import VectorstoreService
from services.RetrieverService import RetrieverService
from services.ResponseGeneratorService import ResponseGeneratorService
from typing import List


class RAGPipeline:
    def __init__(self, urls: List[str]):
        self.loader = WebLoaderService(urls)
        self.chunker = ChunkingService()
        self.embeddings = EmbeddingService()
        self.vectorstore = None
        self.retriever = None
        self.generator = None

    # loading documents
    def load_documents(self):       
        self.documents = self.loader.load_content()

    # splitting docuemnts
    def split_documents(self):        
        self.chunks = self.chunker.create_chunks(self.documents)
        
    # create Embeddings
    def create_embeddings(self):        
        self.embeddings = EmbeddingService()

    # create vectorstore
    def build_vectorstore(self):        
        vector_store = VectorstoreService(self.embeddings.get_embeddings())       
        self.vectorstore = vector_store.create_vectorstore(self.chunks)

    # build RAG after vector store
    def build(self):   
        retriever_component = RetrieverService(self.vectorstore)
        self.retriever = retriever_component.retriever        
        self.generator = ResponseGeneratorService()
        self.qa_chain = self.generator.create_qa_chain(self.retriever)


    # procedure to get response for query
    def query(self, question):
        # Retrieve docs with scores
        docs_with_scores = self.vectorstore.similarity_search_with_score(question, k=3)
       
        # Run QA chain
        response = self.qa_chain.invoke({"question": question})

        # Extract sources from retrieved docs
        sources = list(set([doc.metadata.get("source", "") for doc, _ in docs_with_scores]))

        return {
            "answer": response["answer"],
            "sources": sources,
            "docs_with_scores": docs_with_scores
        }

