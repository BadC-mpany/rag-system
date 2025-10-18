import os
from dotenv import load_dotenv, find_dotenv, set_key

def _get_or_set_env_var(key_name: str, prompt_message: str, optional: bool = True):
    """
    Loads an environment variable from .env file.
    If not found, prompts the user for it and saves it to the .env file.
    If 'optional' is True, it will not prompt if the key is missing.
    """
    load_dotenv()
    
    api_key = os.getenv(key_name)
    
    if not api_key and not optional:
        api_key = input(prompt_message)
        
        # Find the .env file, creating it if it doesn't exist
        dotenv_path = find_dotenv()
        if not dotenv_path:
            dotenv_path = os.path.join(os.getcwd(), '.env')
            with open(dotenv_path, 'w') as f:
                pass # Create an empty .env file
        
        set_key(dotenv_path, key_name, api_key)
        print(f"{key_name} has been saved to your .env file.")
        
    return api_key

# --- API Keys ---
OPENROUTER_API_KEY = _get_or_set_env_var(
    "OPENROUTER_API_KEY", 
    "Please enter your OpenRouter API key: "
)
TAVILY_API_KEY = _get_or_set_env_var(
    "TAVILY_API_KEY",
    "Please enter your Tavily API key: ",
    optional=True
)

# --- Basic Settings ---
BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
MODEL_NAME = os.getenv("OPENROUTER_MODEL_NAME", "meta-llama/llama-3.3-70b-instruct")
CREDENTIALS_FILE = os.path.join(os.path.dirname(__file__), 'credentials.json')

# --- Data Directories ---
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
PUBLIC_DATA_DIR = os.path.join(DATA_DIR, 'public')
WORKER_DATA_DIR = os.path.join(DATA_DIR, 'worker')
ADMIN_DATA_DIR = os.path.join(DATA_DIR, 'admin')

# --- Vector Store Settings ---
VECTOR_STORE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'vector_store')

# --- Logging ---
LOGS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs')

# --- Embedding backend settings ---
# Use the Hugging Face Inference API instead of local sentence-transformers when
# USE_HF_EMBEDDINGS is true (set in .env). HF token should be placed in .env
# as HF_TOKEN. HF_LOGGING enables debug logs for HF embedding calls.
load_dotenv()

# If set to true (1/true/yes) the repo will call HF Inference API for embeddings.
USE_HF_EMBEDDINGS = os.getenv("USE_HF_EMBEDDINGS", "false").lower() in ("1", "true", "yes")
# HF API token will be read from .env (HF_TOKEN)
HF_TOKEN = os.getenv("HF_TOKEN")
# HF pipeline URL (default to all-MiniLM-L6-v2 feature-extraction pipeline)
HF_API_URL = os.getenv("HF_API_URL", "https://api-inference.huggingface.co/models/sentence-transformers/all-MiniLM-L6-v2")
# Enable/disable verbose logging for HF embedding calls
HF_LOGGING = os.getenv("HF_LOGGING", "true").lower() in ("1", "true", "yes")
