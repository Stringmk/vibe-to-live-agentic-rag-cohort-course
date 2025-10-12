from src.tools.vector_search import search_knowledge_base
from src.core.config import get_settings

settings = get_settings()

def test_search_knowledge_base():
    query = "climate change"
    results = search_knowledge_base(query)
    print(results)

def test_collection_exists():
    from qdrant_client import QdrantClient
    client = QdrantClient(url=settings.QDRANT_URL, api_key=settings.QDRANT_API_KEY)
    collections = client.get_collections()
    print(collections)


if __name__ == "__main__":
    test_collection_exists()
    test_search_knowledge_base()