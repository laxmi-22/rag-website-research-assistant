# Loader Service. Loads data from websites
# WebBaseLoader - For static HTML pages,Simple, lightweight, fast
# UnstructuredURLLoader- For pages with structured sections & tables,Better semantic extraction

from langchain_community.document_loaders import WebBaseLoader,UnstructuredURLLoader

class WebLoaderService:
    def __init__(self,urls):
        self.urls=urls

    def load_content(self):
        loader=UnstructuredURLLoader(
            urls=self.urls,
             headers={
                "User-Agent": "Mozilla/5.0"
    })
        documents=loader.load()
        
        if not documents:
            raise ValueError("Could not extract content from provided URLs.")            
        else:            
            print(f"[WebLoader] Loaded content from {len(self.urls)} URLs")
            return documents
