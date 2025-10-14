"""
Vector search tool using Qdrant with FastEmbed for document retrieval.

STUDENT TODO: Complete the implementation of this vector search tool.
Follow the hints and complete the sections marked with TODO.

Learning Objectives:
- Understand how to connect to Qdrant
- Implement semantic search with FastEmbed
- Format and return search results
"""
import os
from qdrant_client import QdrantClient, models
from pydantic_settings import BaseSettings



class VectorSearchTool:
    """
    Tool for performing semantic search using Qdrant vector database with FastEmbed.

    This tool:
    1. Accepts a user query
    2. Uses FastEmbed (via Qdrant) to automatically generate embeddings
    3. Searches Qdrant for similar documents
    4. Returns relevant chunks with metadata
    
    Note: This implementation uses FastEmbed through Qdrant's query_points method,
    which automatically handles embedding generation on both ingestion and query time.
    """

    def __init__(
        self, 
        #settings: BaseSettings = get_settings(),
        qdrant_url: str = None,
        qdrant_api_key: str = None,
        collection_name: str = None,
        model_name: str = "BAAI/bge-small-en"
    ):
        """
        Initialize Qdrant client with FastEmbed support.
        
        Args:
            settings: Application settings (default loads from environment)
            qdrant_url: Qdrant server URL (defaults to QDRANT_URL env var)
            qdrant_api_key: Qdrant API key (defaults to QDRANT_API_KEY env var)
            collection_name: Name of the collection (defaults to 'fed_speeches')
            model_name: FastEmbed model name (defaults to 'BAAI/bge-small-en')
        """
        #self.settings = settings
        # TODO 1: Load Qdrant URL and API key from parameters or environment variables
        self.qdrant_url = qdrant_url or os.getenv('QDRANT_URL')
        self.qdrant_api_key = qdrant_api_key or os.getenv('QDRANT_API_KEY')

        # TODO 2: Validate that URL and API key are provided
        if not self.qdrant_url or not self.qdrant_api_key:
            raise ValueError("QDRANT_URL and QDRANT_API_KEY must be provided")


        # TODO 3: Initialize the Qdrant client
        self.qdrant_client = QdrantClient(url=self.qdrant_url, 
                                          api_key=self.qdrant_api_key)
       
        # TODO 4: Set collection name and model name with default
        self.collection_name = collection_name or "fed_speeches"
        self.model_name = model_name

    def search(self, query: str, limit: int = 5) -> list[dict]:
        """
        Search for relevant documents in Qdrant using FastEmbed.

        Args:
            query: User's search query
            limit: Maximum number of results to return

        Returns:
            List of dictionaries containing:
            - content: Document content
            - metadata: Document metadata (title, speaker, date, etc.)
            - score: Similarity score
        """
        # TODO 5: Perform search using Qdrant's query_points method
        results = self.qdrant_client.query_points(
            collection_name=self.collection_name,
            query=models.Document(text=query, model=self.model_name),
            limit=limit
        ).points

        search_results = results

        #if not search_results:
        #    raise ValueError(f"No results found for query: '{query}'")

        # TODO 6: Format results into a list of dictionaries
        formatted_results = []
        for result in search_results:
            formatted_results.append({
                "id": result.id,
                "content": result.payload.get("document", ""),
                "metadata": {
                    "title": result.payload.get("title", ""),
                    "speaker": result.payload.get("speaker", ""),
                    "url": result.payload.get("url", ""),
                    "description": result.payload.get("description", ""),
                    "pub_date": result.payload.get("pub_date", ""),
                    "category": result.payload.get("category", ""),
                    "content_length": result.payload.get("content_length", 0),
                    "scraped_at": result.payload.get("scraped_at", "")
                },
                "score": result.score
            })

        return formatted_results

    def verify_collection(self) -> dict:
        """
        Verify that the collection exists and return its info.
        
        Returns:
            Dictionary with collection information
        """
        # TODO 7: Try to get collection info and handle errors
        
        try:
            # TODO: Call get_collection and extract info
            collection_info = self.qdrant_client.get_collection(self.collection_name)
            return {
                "exists": True,
                "points_count": collection_info.points_count,
                "vector_size": collection_info.config.params.vectors.size,
                "distance": collection_info.config.params.vectors.distance
            }   
        except Exception as e:
            # TODO: Return error dict
            return {
                "exists": False,
                "error": str(e)
            }


# Tool function for OpenAI Agents SDK
def search_knowledge_base(query: str, limit: int = 5) -> str:
    """
    Search the knowledge base for relevant information.

    This function is designed to be used as a tool with OpenAI Agents SDK.

    Args:
        query: User's search query
        limit: Maximum number of results to return (default: 5)

    Returns:
        Formatted string with search results
    """
    # TODO 8: Implement the search_knowledge_base function
    # Hint: This function wraps VectorSearchTool for use with OpenAI Agents
    # Hint: Use a try/except block to handle errors gracefully
    
    try:
        # TODO: Create VectorSearchTool instance
        tool = VectorSearchTool()
        
        # TODO: Perform search
        results = tool.search(query, limit=limit)
        
        # TODO: Check if results are empty
        if not results:
            return f"No results found for query: '{query}'"
        
        # TODO: Format results as a readable string
        # Include: number of documents, and for each result:
        #   - Result number and score
        #   - Title, Speaker, Date, Category
        #   - Content snippet (first 300 characters)
        formatted_results = []  
        for i, res in enumerate(results):
            
            formatted_results.append(
                f"Result: {i+1} \n"
                f"Id: {res['id']}\n"
                f"Score: {res['score']:.4f}:\n"
                f"Title: {res['metadata']['title']}\n"
                f"Speaker: {res['metadata']['speaker']}\n"
                f"Date: {res['metadata']['pub_date']}\n"
                f"Category: {res['metadata']['category']}\n"
                f"Content Snippet: {res['content'][:300]}...\n"
            )
            
        return f"Found {len(results)} relevant documents:\n\n" + "\n".join(formatted_results)

    except Exception as e:
        return f"Error searching knowledge base: {str(e)}"

