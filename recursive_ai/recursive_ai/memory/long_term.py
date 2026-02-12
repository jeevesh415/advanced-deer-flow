import os
from typing import List, Dict, Optional
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

class CognitiveMemory:
    """Long-term memory using vector store."""
    def __init__(self, collection_name: str = "recursive_ai_memory", persist_directory: str = "./db"):
        self.collection_name = collection_name
        self.persist_directory = persist_directory

        # Initialize embeddings (assuming OPENAI_API_KEY is set)
        if not os.getenv("OPENAI_API_KEY"):
            # Provide a warning or fallback
            print("WARNING: OPENAI_API_KEY not found. Memory will not function correctly without it.")
            self.embeddings = None
        else:
            self.embeddings = OpenAIEmbeddings()

        if self.embeddings:
            self.vector_store = Chroma(
                collection_name=collection_name,
                embedding_function=self.embeddings,
                persist_directory=persist_directory
            )
        else:
            self.vector_store = None

    def add_memory(self, text: str, metadata: Dict[str, str] = None) -> str:
        """Stores a piece of information."""
        if not self.vector_store:
            return "Memory Disabled"

        doc = Document(page_content=text, metadata=metadata or {})
        ids = self.vector_store.add_documents([doc])
        return ids[0] if ids else ""

    def retrieve_relevant(self, query: str, k: int = 5) -> List[Document]:
        """Retrieves most relevant memories."""
        if not self.vector_store:
            return []
        return self.vector_store.similarity_search(query, k=k)

    def search_by_metadata(self, filter_dict: Dict[str, str]) -> List[Document]:
        """Find memories matching specific metadata."""
        if not self.vector_store:
            return []
        # Chroma supports where filter
        return self.vector_store.similarity_search(" ", k=100, filter=filter_dict)

    def clear(self):
        """Wipes memory (Use with caution)."""
        if self.vector_store:
            self.vector_store.delete_collection()
