# chunking service - splits loaded documents into chunks
# Used RecursiveCharacterTextSplitter -Recursive splitting preserves paragraphs and sentences and degrades gracefully

from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import time


class ChunkingService:
    def __init__(self,chunk_size:int =256,chunk_overlap:int=50):
        self.splitter=RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
    
    def create_chunks(self,documents:List[Document]):
        start=time.time()
        chunks=self.splitter.split_documents(documents)
        end=time.time()
        chunktime=round(end-start,2)

        if len(chunks)==0:
            raise ValueError("No chunks created.")            
        else:           
            print(f"[Chunking] Created {len(chunks)} chunks from {len(documents)} documents in {chunktime} seconds")
            return chunks

