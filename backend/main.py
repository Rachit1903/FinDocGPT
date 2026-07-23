"""FastAPI entrypoint for FinDocGPT."""

from __future__ import annotations

import os

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from backend.schemas import ChatRequest, ChatResponse, HealthResponse
from backend.service import FinDocService


load_dotenv()


def _allowed_origins() -> list[str]:
    origins = os.getenv("CORS_ORIGINS", "http://localhost:5173").split(",")
    return [origin.strip() for origin in origins if origin.strip()]


app = FastAPI(title="FinDocGPT API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
_service: FinDocService | None = None


def get_service() -> FinDocService:
    global _service
    if _service is None:
        _service = FinDocService()
    return _service


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        qdrant_url=os.getenv("QDRANT_URL", "http://localhost:6333"),
        generation_model=os.getenv("GROQ_GENERATION_MODEL"),
        judge_model=os.getenv("GROQ_JUDGE_MODEL"),
    )


@app.post("/api/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    try:
        payload = get_service().answer(request.query, k=request.k)
    except Exception as exc:  # pragma: no cover - HTTP wrapper around runtime failures
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return ChatResponse(**payload)
