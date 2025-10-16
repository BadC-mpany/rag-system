from langchain_community.embeddings import SentenceTransformerEmbeddings

def get_embedding_model():
    """
    Initializes and returns the sentence-transformer embedding model.
    """
    # Using a popular, lightweight model.
    # You can choose other models from sentence-transformers.
    model_name = "all-MiniLM-L6-v2"
    
    try:
        embeddings = SentenceTransformerEmbeddings(model_name=model_name)
        return embeddings
    except Exception as e:
        print(f"Error initializing embedding model: {e}")
        print("Please ensure you have an internet connection and the required libraries are installed.")
        return None
