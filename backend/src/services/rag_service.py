import logging
from functools import lru_cache
from typing import Any, Dict

from config import get_settings, Settings
from agents.rag_agent import create_rag_agent
from models.models import AgentResponse

logger = logging.getLogger(__name__)


class RagService:
    """Service layer that wraps the RAG agent for use in FastAPI endpoints.

    Keeps a single RAG agent instance and normalizes responses to `ChatResponse`.
    """

    def __init__(self, settings: Settings = None) -> None:
        # Settings comes from get_settings(); allow tests to inject mocks
        self.settings = settings or get_settings()
        self.rag_agent = create_rag_agent()

    async def chat(self, 
                   query: str, 
                   session_id: str) -> AgentResponse:
        """Call the underlying RAG agent and return a ChatResponse.

        Args:
            query: user query
            session_id: session identifier

        Returns:
            ChatResponse: normalized response
        """
        logger.info("Received chat request", extra={"query": query, "session_id": session_id})

        try:

            if not self.rag_agent:
                raise RuntimeError("RAG agent is not initialized.")

            result = await self.rag_agent.chat(query=query, 
                                            session_id=session_id)

            # result may already be a pydantic model or a simple object/dict
            if isinstance(result, AgentResponse):
                return result
        except Exception as exc:
            logger.error("Error during RAG agent chat", exc_info=exc)
            raise


@lru_cache(maxsize=1)
def get_rag_service() -> RagService:
    """Dependency provider for FastAPI - returns a cached RagService instance."""
    return RagService()
