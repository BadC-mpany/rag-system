import os
from typing import List, Dict, Callable

from langchain_community.document_loaders import (
    TextLoader, CSVLoader, Docx2txtLoader
)

# Try to import additional loaders, fall back to TextLoader if not available
try:
    from langchain_community.document_loaders import PyPDFLoader
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

try:
    from langchain_community.document_loaders import UnstructuredMarkdownLoader
    MARKDOWN_AVAILABLE = True
except ImportError:
    MARKDOWN_AVAILABLE = False

try:
    from langchain_community.document_loaders import JSONLoader
    JSON_AVAILABLE = True
except ImportError:
    JSON_AVAILABLE = False
from langchain.schema import Document

# Build the loader mapping dynamically based on available imports
LOADER_MAPPING: Dict[str, Callable] = {
    # Text formats
    ".txt": TextLoader,
    ".md": UnstructuredMarkdownLoader if MARKDOWN_AVAILABLE else TextLoader,
    ".markdown": UnstructuredMarkdownLoader if MARKDOWN_AVAILABLE else TextLoader,
    
    # Document formats
    ".pdf": PyPDFLoader if PDF_AVAILABLE else TextLoader,
    ".docx": Docx2txtLoader,
    
    # Data formats
    ".csv": CSVLoader,
    ".json": JSONLoader if JSON_AVAILABLE else TextLoader,
    
    # Code formats - all use TextLoader for now
    ".py": TextLoader,    # Python files
    ".cpp": TextLoader,   # C++ files
    ".c": TextLoader,     # C files
    ".h": TextLoader,     # Header files
    ".hpp": TextLoader,   # C++ header files
    ".js": TextLoader,    # JavaScript
    ".ts": TextLoader,    # TypeScript
    ".java": TextLoader,  # Java
    ".go": TextLoader,    # Go
    ".rs": TextLoader,    # Rust
    ".php": TextLoader,   # PHP
    ".rb": TextLoader,    # Ruby
    ".swift": TextLoader, # Swift
    ".kt": TextLoader,    # Kotlin
    ".scala": TextLoader, # Scala
    ".r": TextLoader,     # R
    ".sql": TextLoader,   # SQL
    ".sh": TextLoader,    # Shell scripts
    ".bash": TextLoader,  # Bash scripts
    ".ps1": TextLoader,   # PowerShell
    ".yaml": TextLoader,  # YAML
    ".yml": TextLoader,   # YAML
    ".toml": TextLoader,  # TOML
    ".ini": TextLoader,   # INI files
    ".cfg": TextLoader,   # Config files
    ".conf": TextLoader,  # Config files
    ".css": TextLoader,   # CSS files
    ".html": TextLoader,  # HTML files
    ".htm": TextLoader,   # HTML files
    ".xml": TextLoader,   # XML files
    
    # Other text-based formats
    ".log": TextLoader,   # Log files
    ".rtf": TextLoader,   # Rich Text Format (basic support)
}

def get_supported_file_types() -> List[str]:
    """
    Returns a list of supported file extensions.
    """
    return list(LOADER_MAPPING.keys())

def load_documents_from_directories(dir_paths: List[str]) -> List[Document]:
    """
    Loads documents from a list of directories, supporting multiple file types.
    
    Supported formats:
    - Text: .txt, .md, .markdown
    - Documents: .docx, .pdf (if pypdf installed)
    - Data: .csv, .json (if available)
    - Code: .py, .cpp, .c, .h, .hpp, .js, .ts, .java, .go, .rs, .php, .rb, .swift, .kt, .scala, .r, .sql
    - Config: .yaml, .yml, .toml, .ini, .cfg, .conf
    - Web: .html, .htm, .css, .xml
    - Scripts: .sh, .bash, .ps1
    - Other: .log, .rtf
    """
    documents = []
    loaded_files = []
    skipped_files = []
    
    for dir_path in dir_paths:
        if not os.path.exists(dir_path):
            print(f"Directory not found: {dir_path}")
            continue
            
        for filename in os.listdir(dir_path):
            filepath = os.path.join(dir_path, filename)
            
            # Skip directories
            if os.path.isdir(filepath):
                continue
                
            ext = "." + filename.rsplit(".", 1)[-1].lower() if '.' in filename else None
            
            if ext in LOADER_MAPPING:
                try:
                    loader_class = LOADER_MAPPING[ext]
                    loader = loader_class(filepath)
                    docs = loader.load()
                    documents.extend(docs)
                    loaded_files.append(f"{filename} ({ext})")
                except Exception as e:
                    print(f"Error loading file {filepath}: {e}")
                    skipped_files.append(f"{filename} ({ext}) - {str(e)}")
            else:
                skipped_files.append(f"{filename} - unsupported format")

    # Print summary
    if loaded_files:
        print(f"Successfully loaded {len(loaded_files)} files:")
        for file in loaded_files:
            print(f"  ✓ {file}")
    
    if skipped_files:
        print(f"Skipped {len(skipped_files)} files:")
        for file in skipped_files:
            print(f"  ✗ {file}")

    return documents
