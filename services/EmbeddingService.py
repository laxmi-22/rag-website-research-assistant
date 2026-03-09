# Embedding service
# used HuggingFaceEndpointEmbeddings.
# Store HuggingFace token in .env file

import os
from getpass import getpass
from langchain_huggingface.embeddings import HuggingFaceEndpointEmbeddings


class EmbeddingService:
    def __init__(self, model_name: str = "BAAI/bge-base-en-v1.5"):
        # Get HuggingFace token if not already set
        if 'HUGGINGFACEHUB_API_TOKEN' not in os.environ:
            hf_token = getpass("Enter your HuggingFace API token: ")
            os.environ['HUGGINGFACEHUB_API_TOKEN'] = hf_token
            
        self.embeddings = HuggingFaceEndpointEmbeddings(            
            model=model_name,
            huggingfacehub_api_token=os.environ['HUGGINGFACEHUB_API_TOKEN'],
        )
    
    def get_embeddings(self):        
        print("[Embedding] Embedding model initialized")

        return self.embeddings
