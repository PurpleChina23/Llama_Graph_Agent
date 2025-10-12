"""

Embedding utilities for the LLM Agent

This module provides embedding functionality for text processing and similarity search.
"""

import os
from typing import List, Optional
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
import numpy as np


class EmbeddingManager:
    """Manages text embeddings and vector storage."""
    
    def __init__(self, model_name: str = "text-embedding-3-small"):
        """Initialize the embedding manager.
        
        Args:
            model_name: The embedding model to use
        """
        self.model_name = model_name
        self.embeddings = None
        self.vectorstore = None
        self._initialize_embeddings()
    
    def _initialize_embeddings(self):
        """Initialize the embedding model."""
        try:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                print("⚠️ OPENAI_API_KEY not found. Embeddings will not work.")
                return
            
            self.embeddings = OpenAIEmbeddings(
                model=self.model_name,
                openai_api_key=api_key
            )
            print(f"✅ Embeddings initialized with model: {self.model_name}")
        except Exception as e:
            print(f"❌ Failed to initialize embeddings: {e}")
    
    def create_embeddings(self, texts: List[str]) -> Optional[List[List[float]]]:
        """Create embeddings for a list of texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors, or None if embeddings not initialized
        """
        if not self.embeddings:
            print("❌ Embeddings not initialized")
            return None
        
        try:
            return self.embeddings.embed_documents(texts)
        except Exception as e:
            print(f"❌ Failed to create embeddings: {e}")
            return None
    
    def create_vectorstore(self, documents: List[Document]) -> bool:
        """Create a FAISS vector store from documents.
        
        Args:
            documents: List of Document objects
            
        Returns:
            True if successful, False otherwise
        """
        if not self.embeddings:
            print("❌ Embeddings not initialized")
            return False
        
        try:
            self.vectorstore = FAISS.from_documents(documents, self.embeddings)
            print(f"✅ Vector store created with {len(documents)} documents")
            return True
        except Exception as e:
            print(f"❌ Failed to create vector store: {e}")
            return False
    
    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """Perform similarity search in the vector store.
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of similar documents
        """
        if not self.vectorstore:
            print("❌ Vector store not initialized")
            return []
        
        try:
            return self.vectorstore.similarity_search(query, k=k)
        except Exception as e:
            print(f"❌ Failed to perform similarity search: {e}")
            return []
    
    def add_documents(self, documents: List[Document]) -> bool:
        """Add documents to the existing vector store.
        
        Args:
            documents: List of Document objects to add
            
        Returns:
            True if successful, False otherwise
        """
        if not self.vectorstore:
            print("❌ Vector store not initialized")
            return False
        
        try:
            self.vectorstore.add_documents(documents)
            print(f"✅ Added {len(documents)} documents to vector store")
            return True
        except Exception as e:
            print(f"❌ Failed to add documents: {e}")
            return False


# Global embedding manager instance
embedding_manager = EmbeddingManager()


def get_embedding_manager() -> EmbeddingManager:
    """Get the global embedding manager instance.
    
    Returns:
        The global EmbeddingManager instance
    """
    return embedding_manager
