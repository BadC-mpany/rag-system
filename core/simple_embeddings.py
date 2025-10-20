"""
Simple offline embeddings using basic text features.
This is a fallback when network-based embeddings fail.
"""
import hashlib
import re
from typing import List
from collections import Counter
import math


class SimpleOfflineEmbeddings:
    """
    A simple offline embedding model that creates embeddings based on text features.
    This doesn't require any network access or pre-trained models.
    """
    
    def __init__(self, dimension: int = 384):
        self.dimension = dimension
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple documents."""
        return [self.embed_query(text) for text in texts]
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single text using simple features."""
        # Normalize text
        text = text.lower().strip()
        
        # Create feature vector
        features = []
        
        # 1. Character-based features (first 100 dimensions)
        char_counts = Counter(text)
        for i in range(26):  # a-z
            char = chr(ord('a') + i)
            features.append(char_counts.get(char, 0) / max(len(text), 1))
        
        # 2. Word-based features (next 100 dimensions)
        words = re.findall(r'\b\w+\b', text)
        word_counts = Counter(words)
        
        # Common words features
        common_words = ['the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 
                       'with', 'by', 'from', 'up', 'about', 'into', 'through', 'during',
                       'before', 'after', 'above', 'below', 'between', 'among', 'is', 'are',
                       'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does',
                       'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can',
                       'company', 'business', 'technology', 'system', 'data', 'information',
                       'project', 'team', 'work', 'service', 'product', 'customer', 'user',
                       'development', 'management', 'solution', 'platform', 'software',
                       'application', 'network', 'security', 'performance', 'quality',
                       'innovation', 'achievement', 'success', 'growth', 'revenue', 'budget',
                       'financial', 'report', 'analysis', 'strategy', 'planning', 'goal']
        
        for word in common_words[:74]:  # Use 74 to make total 100
            features.append(word_counts.get(word, 0) / max(len(words), 1))
        
        # 3. Text statistics (next 50 dimensions)
        features.extend([
            len(text) / 1000.0,  # Text length (normalized)
            len(words) / max(len(text), 1),  # Word density
            len(set(words)) / max(len(words), 1),  # Vocabulary diversity
            sum(1 for c in text if c.isupper()) / max(len(text), 1),  # Uppercase ratio
            sum(1 for c in text if c.isdigit()) / max(len(text), 1),  # Digit ratio
            text.count('.') / max(len(text), 1),  # Sentence density
            text.count(',') / max(len(text), 1),  # Comma density
            text.count('!') / max(len(text), 1),  # Exclamation density
            text.count('?') / max(len(text), 1),  # Question density
            text.count('\n') / max(len(text), 1),  # Line break density
        ])
        
        # 4. Hash-based features for remaining dimensions
        remaining_dims = self.dimension - len(features)
        if remaining_dims > 0:
            # Create hash-based features for semantic similarity
            hash_input = text.encode('utf-8')
            for i in range(remaining_dims):
                # Create different hash seeds
                seed_hash = hashlib.md5(f"{i}_{hash_input.hex()}".encode()).hexdigest()
                # Convert to float between -1 and 1
                hash_val = int(seed_hash[:8], 16) / (16**8) * 2 - 1
                features.append(hash_val)
        
        # Ensure we have exactly the right number of dimensions
        features = features[:self.dimension]
        while len(features) < self.dimension:
            features.append(0.0)
        
        # Normalize the vector
        magnitude = math.sqrt(sum(x*x for x in features))
        if magnitude > 0:
            features = [x / magnitude for x in features]
        
        return features
