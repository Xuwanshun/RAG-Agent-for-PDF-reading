"""
Query endpoint — ask a question against the indexed corpus.

Every question and answer is saved to the database so users can find
their chat histories after logging out and back in.
"""

from __future__ import annotations

import logging
import time
import uuid
from datetime import UTC, datetime

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from sqlalchemy import insert, select

from api.dependencies import get_current_user
from config import user_scoped_settings
from db.models import conversations, messages
from rag.dispatch import answer_question

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/query", tags=["query"])


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, description="The question to ask.")
    top_k: int = Field(default=4, ge=1, le=20, description="Number of chunks to retrieve (1-20).")
    doc_filter: list[str] | None = Field(
        default=None, description="Limit search to these document IDs. Null searches all."
    )
    # If provided, saves messages to this existing conversation.
    # If None, a new conversation is created automatically.
    conversation_id: str | None = Field(default=None, description="Conversation to append to. Omit to start a new one.")


@router.post("")
async def query(request: Request, body: QueryRequest, user: dict = Depends(get_current_user)) -> JSONResponse:
    """
    Ask a question against the indexed PDF corpus.

    Every call saves two rows to the messages table:
      1. The user's question  (role="user")
      2. The assistant answer (role="assistant", sources attached)

    Pass conversation_id to continue an existing chat.
    Omit it to start a new conversation (ID is returned in the response).
    """
    settings = request.app.state.settings
    scoped = user_scoped_settings(settings, user["id"])
    engine = request.app.state.db_engine

    if not scoped.openai_api_key:
        raise HTTPException(
            status_code=422,
            detail="OPENAI_API_KEY is required for queries. Set it in your environment.",
        )

    now = datetime.now(UTC)
    conversation_id = body.conversation_id

    with engine.begin() as conn:
        if not conversation_id:
            # Start a new conversation, titled from the first question.
            conversation_id = str(uuid.uuid4())
            conn.execute(
                insert(conversations).values(
                    id=conversation_id,
                    user_id=user["id"],
                    title=body.question[:120],
                    created_at=now,
                )
            )
        else:
            # Verify this conversation belongs to the requesting user.
            row = conn.execute(
                select(conversations).where(
                    conversations.c.id == conversation_id,
                    conversations.c.user_id == user["id"],
                )
            ).first()
            if not row:
                raise HTTPException(status_code=404, detail="Conversation not found")

        # Save the user's question immediately — even if the RAG pipeline
        # fails the question is recorded so history is never silently lost.
        conn.execute(
            insert(messages).values(
                id=str(uuid.uuid4()),
                conversation_id=conversation_id,
                role="user",
                content=body.question,
                sources=None,
                created_at=now,
            )
        )

    # ── RAG pipeline ───────────────────────────────────────────────────────────
    logger.info("Query received: %r (top_k=%d, conversation=%s)", body.question, body.top_k, conversation_id)
    started = time.time()
    try:
        response = answer_question(
            body.question,
            settings=scoped,
            top_k=body.top_k,
            doc_filter=body.doc_filter,
        )
    except Exception as exc:
        logger.exception("Query failed: %r", body.question)
        raise HTTPException(status_code=500, detail=f"Query failed: {exc}") from exc

    sources_payload = response.sources

    # Save the assistant answer with its source citations.
    with engine.begin() as conn:
        conn.execute(
            insert(messages).values(
                id=str(uuid.uuid4()),
                conversation_id=conversation_id,
                role="assistant",
                content=response.answer,
                sources=sources_payload,
                created_at=datetime.now(UTC),
            )
        )

    return JSONResponse(
        {
            "question": response.question,
            "answer": response.answer,
            "sources": sources_payload,
            "router": response.router,
            "specialists": [
                {"agent_name": s.agent_name, "output": s.output, "region_ids": s.region_ids}
                for s in response.specialists
            ],
            "latency_ms": int((time.time() - started) * 1000),
            "top_k": body.top_k,
            "conversation_id": conversation_id,
        }
    )
