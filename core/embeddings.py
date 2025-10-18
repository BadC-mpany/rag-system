import os
import requests
from typing import List

from config.settings import USE_HF_EMBEDDINGS, HF_API_URL, HF_TOKEN, HF_LOGGING

try:
    from langchain_community.embeddings import SentenceTransformerEmbeddings
except Exception:
    SentenceTransformerEmbeddings = None


class HFEmbeddings:
    """Simple wrapper to call Hugging Face Inference API for embeddings.

    Implements the minimal interface expected (embed_documents / embed_query)
    used by downstream vectorstore builders.
    """
    def __init__(self, api_url: str = HF_API_URL, token: str = HF_TOKEN, logging: bool = HF_LOGGING):
        self.api_url = api_url
        self.token = token
        self.headers = {"Authorization": f"Bearer {self.token}"} if self.token else {}
        self.logging = logging
        self._working_payload_format = None  # Cache discovered payload format

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # Try batch processing first if we haven't determined the format yet
        # or if we know batch works
        if len(texts) > 1 and self._working_payload_format != str:
            try:
                return self._embed_batch(texts)
            except Exception as e:
                if self.logging:
                    print(f"HFEmbeddings: batch embedding failed ({e}), falling back to individual calls")
        
        # Fall back to individual embedding calls
        return [self._embed(text) for text in texts]
    
    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Attempt to embed multiple texts in a single API call."""
        if self.logging:
            print(f"HFEmbeddings: attempting batch embedding for {len(texts)} texts")
        
        if not self.token:
            raise RuntimeError("HF token (HF_TOKEN) is not set.")
        
        # Trim long texts
        max_len = 2000
        trimmed_texts = []
        for text in texts:
            if len(text) > max_len:
                trimmed_texts.append(text[:max_len])
            else:
                trimmed_texts.append(text)
        
        payload = {
            "inputs": trimmed_texts,
            "options": {"wait_for_model": True}
        }
        
        try:
            resp = requests.post(self.api_url, headers=self.headers, json=payload, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            
            if isinstance(data, list) and len(data) == len(texts):
                embeddings = []
                for item in data:
                    if isinstance(item, list):
                        if item and isinstance(item[0], (int, float)):
                            embeddings.append(item)
                        else:
                            raise RuntimeError(f"Unexpected batch item format: {type(item)}")
                    else:
                        raise RuntimeError(f"Unexpected batch item type: {type(item)}")
                
                if self.logging:
                    print(f"HFEmbeddings: batch embedding succeeded, got {len(embeddings)} vectors of dim {len(embeddings[0]) if embeddings else 0}")
                return embeddings
            else:
                raise RuntimeError(f"Batch response length mismatch: expected {len(texts)}, got {len(data) if isinstance(data, list) else 'not a list'}")
                
        except Exception as e:
            if self.logging:
                print(f"HFEmbeddings: batch request failed: {e}")
            raise

    def embed_query(self, text: str) -> List[float]:
        return self._embed(text)

    def _embed(self, text: str) -> List[float]:
        if self.logging:
            print(f"HFEmbeddings: requesting embedding for text of length {len(text)}")
        if not self.token:
            raise RuntimeError("HF token (HF_TOKEN) is not set. Please add it to .env or disable USE_HF_EMBEDDINGS.")
        
        max_len = 2000
        if len(text) > max_len:
            if self.logging:
                print(f"HFEmbeddings: input too long ({len(text)}), trimming to {max_len} characters")
            text = text[:max_len]

        payload = {
            "inputs": text,
            "options": {"wait_for_model": True}
        }

        try:
            if self.logging:
                print(f"HFEmbeddings: calling {self.api_url}")
            resp = requests.post(self.api_url, headers=self.headers, json=payload, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            
            if self.logging:
                print(f"HFEmbeddings: received response type: {type(data)}")
            
            # Handle error responses
            if isinstance(data, dict) and "error" in data:
                err = data.get("error")
                raise RuntimeError(f"HF inference error: {err}")

            # Handle successful responses - HF returns flat array for single input
            if isinstance(data, list):
                if data and isinstance(data[0], (int, float)):
                    # Flat array: [0.1, 0.2, ...]
                    if self.logging:
                        print(f"HFEmbeddings: got embedding of dim {len(data)}")
                    return data
                elif data and isinstance(data[0], list):
                    # Nested: [[0.1, 0.2, ...]]
                    if self.logging:
                        print(f"HFEmbeddings: got nested embedding of dim {len(data[0])}")
                    return data[0]

            raise RuntimeError(f"Unexpected HF response format: {type(data)}, first element: {type(data[0]) if isinstance(data, list) and data else 'N/A'}")
            
        except requests.HTTPError as http_err:
            body = resp.text if resp is not None else None
            if self.logging:
                print(f"HFEmbeddings: HTTP {resp.status_code}: {body[:500] if body else 'no body'}")
            raise RuntimeError(f"HF API error {resp.status_code}: {body[:200] if body else str(http_err)}")
        except Exception as exc:
            if self.logging:
                print(f"HFEmbeddings: Exception: {exc}")
            raise


def get_embedding_model():
    """Return either a Hugging Face-backed embeddings wrapper or local SentenceTransformerEmbeddings."""
    if USE_HF_EMBEDDINGS:
        if HF_LOGGING:
            print("Embeddings: configured to use Hugging Face Inference API")
        if not HF_TOKEN:
            print("HF embeddings enabled but HF_TOKEN is not set.")
            return None
        return HFEmbeddings()

    # Fallback to local sentence-transformers
    model_name = "all-MiniLM-L6-v2"
    if SentenceTransformerEmbeddings is None:
        print("Local SentenceTransformerEmbeddings not available (missing package).")
        return None
    try:
        embeddings = SentenceTransformerEmbeddings(model_name=model_name)
        return embeddings
    except Exception as e:
        print(f"Error initializing local embedding model: {e}")
        return None
