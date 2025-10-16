from langchain_openai import ChatOpenAI

from config.settings import OPENROUTER_API_KEY, BASE_URL, MODEL_NAME


def get_llm():
    """
    Initializes and returns the ChatOpenRouter LLM.
    """
    try:
        llm = ChatOpenAI(
            api_key=OPENROUTER_API_KEY,
            base_url=BASE_URL,
            model=MODEL_NAME,
            streaming=True
        )
        return llm
    except Exception as e:
        print(f"Error initializing LLM: {e}")
        return None
