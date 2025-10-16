from langchain_community.document_loaders import WebBaseLoader
from langchain_core.tools import tool

@tool
def url_parser_tool(url: str) -> str:
    """
    MUST be used to access and read the content of any external website URL provided by the user.
    The input to this tool must be a single, valid URL string (e.g., 'https://example.com').
    This is your primary tool for browsing or accessing links on the web.
    """
    try:
        loader = WebBaseLoader(url)
        docs = loader.load()
        content = " ".join([doc.page_content for doc in docs])
        return f"Successfully scraped content from {url}:\n\n{content}"
    except Exception as e:
        return f"Error scraping URL {url}: {e}. Please ensure the URL is valid and accessible."

def get_url_parser_tool():
    """
    Returns the URL parser tool.
    """
    return url_parser_tool
