from typing import List

from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from langchain_core.embeddings import Embeddings

from config.settings import VECTOR_STORE_DIR


def create_or_load_vectorstore(documents: List[Document], embeddings: Embeddings, user_role: str):
    """
    Creates a new ChromaDB vector store or loads an existing one
    based on the user's role.
    """
    # Each role gets a separate vector store to maintain data separation.
    persist_directory = f"{VECTOR_STORE_DIR}_{user_role}"
    
    try:
        vector_store = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory=persist_directory
        )
        return vector_store
    except Exception as e:
        print(f"Error creating/loading vector store: {e}")
        return None
