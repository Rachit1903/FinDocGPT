"""FastAPI entrypoint: POST /api/chat runs the agent; GET /health reports configuration."""

from __future__ import annotations

import os
from functools import lru_cache

from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware

from backend.schemas import ChatRequest, ChatResponse, HealthResponse
from src.config import settings
from src.llm import LLMError

app = FastAPI(title="FinDocGPT API", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in os.getenv("CORS_ORIGINS", "http://localhost:5173").split(",") if o.strip()],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@lru_cache(maxsize=1)
def get_agent():
    from src.agent.graph import FinDocAgent

    return FinDocAgent()


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", vector_store=settings.qdrant_url or f"embedded:{settings.qdrant_path.name}",
                          generator_model=settings.generator_model, agent_model=settings.agent_model,
                          embedding_model=settings.embedding_model)


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    try:
        result = await run_in_threadpool(get_agent().run, request.query)
    except LLMError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return ChatResponse(**result)
