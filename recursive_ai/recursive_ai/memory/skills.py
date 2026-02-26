from typing import List, Dict, Optional
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

class SkillLibrary:
    """Specialized memory for executable code patterns and strategies."""
    def __init__(self, collection_name: str = "recursive_ai_skills", persist_directory: str = "./db"):
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        try:
            self.embeddings = OpenAIEmbeddings()
            self.vector_store = Chroma(
                collection_name=collection_name,
                embedding_function=self.embeddings,
                persist_directory=persist_directory
            )
        except Exception:
            self.vector_store = None

    def store_skill(self, name: str, code: str, description: str, usage_count: int = 0) -> str:
        """Stores a successful code pattern or function."""
        if not self.vector_store:
            return "Skill Library Disabled"

        doc = Document(
            page_content=f"{description}\n\n```python\n{code}\n```",
            metadata={"name": name, "type": "code", "usage_count": usage_count}
        )
        ids = self.vector_store.add_documents([doc])
        return ids[0] if ids else ""

    def retrieve_skill(self, query: str, k: int = 3) -> List[Document]:
        """Finds code snippets relevant to the problem."""
        if not self.vector_store:
            return []
        return self.vector_store.similarity_search(query, k=k)

    def update_usage(self, skill_id: str):
        """Increments usage count (simulated via metadata update - requires re-indexing in Chroma)."""
        # For simplicity in this iteration, we skip complex update logic
        pass
