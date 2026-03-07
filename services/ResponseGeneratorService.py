# Response Generator service
# Used Groq API key for chat with model llama-3.1-8b-instant
# used RetrievalQAWithSourcesChain so that response should return answer as well as source links 

from langchain_classic.chains.qa_with_sources.retrieval import RetrievalQAWithSourcesChain
import os
from dotenv import load_dotenv,find_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq

# 1. Force find and load the .env file
load_dotenv(find_dotenv())

# 2. Verify it loaded (debugging step)
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("GROQ_API_KEY not found. Check your .env file!")


class ResponseGeneratorService:
    def __init__(self, model_id: str = "llama-3.1-8b-instant"):       
        self.model = ChatGroq(
                model=model_id,
                temperature=0.1, # Lower is better for factual RAG
                groq_api_key=api_key
)
    def create_qa_chain(self, retriever): 

         self.template = """
                    You are a strict document-based assistant.

                                Rules:
                                - Use ONLY the information present in the context
                                - Do NOT add explanations
                                - Do NOT use external knowledge
                                - You may list multiple items if they appear in different parts of the context
                                - If the answer is not explicitly stated, say "I don't know"

                                Context:
                                {context}

                                Question:
                                {question}

                                Answer: (copy or lightly rephrase from context only):                    
                    """

         self.prompt = PromptTemplate(
            template=self.template,
            input_variables=["context", "question"]
        )


        # Standard initialization
         return RetrievalQAWithSourcesChain.from_chain_type(
                llm=self.model,
                chain_type="stuff",
                retriever=retriever                
            )


       