from typing import Optional
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException
import logging

from services.rag_service import get_rag_service, RagService
from models.models import ChatRequest, ChatResponse

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/chat", response_model=ChatResponse)
async def rag_controller(
    request: ChatRequest,
    service: RagService = Depends(get_rag_service),
    session_id: Optional[str] = None,
) -> ChatResponse:
    """POST /chat endpoint.

    - `request` is the JSON body validated as `ChatRequest`.
    - `session_id` may be provided as a query parameter; otherwise use request.session_id or generate one.
    """
    sid = request.session_id or session_id or str(uuid4())

    try:
        result = await service.chat(query=request.query, session_id=sid)

        # Ensure result is ChatResponse (service normalizes), but be defensive
        if isinstance(result, ChatResponse):
            return result

        # If service returned dict-like, convert
        try:
            return ChatResponse(
                answer=result.get("answer", ""),
                sources=result.get("sources"),
                session_id=sid,
            )
        except Exception:
            # Last-resort stringification
            return ChatResponse(answer=str(result), sources=None, session_id=sid)

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Unhandled error in /chat")
        raise HTTPException(status_code=500, detail="Internal server error")

