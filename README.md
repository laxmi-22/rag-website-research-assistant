# AI Website Research Assistant (RAG)

Part of my learning journey in Retrieval-Augmented Generation (RAG) systems.
A Retrieval-Augmented Generation (RAG) application that allows users to ask questions about the content of websites.

The application ingests website URLs, processes the content, stores embeddings in a vector database, and uses an LLM to generate answers grounded in the retrieved context.

## Features

- Load content from multiple website URLs
- Automatically split content into chunks
- Generate embeddings for semantic search
- Store embeddings in a vector database (Chroma)
- Retrieve relevant content based on user queries
- Generate answers using a Large Language Model
- Display source links for transparency
- Simple interactive UI built with Streamlit

## Architecture

The application follows a typical RAG pipeline:

1. URL Loading  
   Load website content using LangChain URL loaders

2. Text Chunking  
   Split documents into smaller chunks for embedding

3. Embedding Generation  
   Convert text chunks into vector embeddings

4. Vector Storage  
   Store embeddings in a Chroma vector database

5. Retrieval  
   Retrieve the most relevant chunks for a user query

6. Answer Generation  
   Generate an answer using the retrieved context

## Project Structure



## Installation

### Clone the repository
git clone https://github.com/laxmi-22/rag-website-research-assistant.git
cd rag-website-research-assistant

### Create virtual environment
python -m venv venv

### Activate environment
Windows
venv\Scripts\activate

### Install dependencies
pip install -r requirements.txt

### Add your API key
Create `.env` file:
GROQ_API_KEY=your_api_key_here
HUGGINGFACEHUB_API_TOKEN= "your_huggingface_token"


## Run the Application
streamlit run app.py

The application will open in your browser.

## Example

Enter website URLs in the sidebar and ask questions related to their content.

Example query:
What is FastAPI?


The system retrieves relevant content from the provided websites and generates an answer along with the source links.

## Screenshot

![App Screenshot](screenshot.png)

## Tech Stack

- Python
- LangChain -Orchestration & Prompt Management
- Groq cloud- High-speed inference
- ChromaDB - Vector store
- HuggingFace Embedding
- Streamlit-Simple & interactive user interface
- llama-3.1-8b-instant – Large Language Model

## Future Improvements

- Retrieval score visualization
- Chat history
- Multi-query retrieval
- Better chunk visualization
- Logging and monitoring



