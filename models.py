from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum


class RegulationType(str, Enum):
    GDPR = "gdpr"
    AI_ACT = "ai_act"
    DSA = "dsa"
    NIS2 = "nis2"


class Citation(BaseModel):
    regulation: str
    article: str
    title: str
    excerpt: str
    url: str
    relevance_score: float = 0.0


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=2000)
    regulations: list[str] = Field(default_factory=list)
    k: int = Field(default=5, ge=1, le=20)
    language: str = Field(default="en")
    mode: str = Field(default="detailed")  # "short" or "detailed"


class AskResponse(BaseModel):
    answer: str
    citations: list[Citation]
    confidence: float
    sources: list[dict] = Field(default_factory=list)  # Backwards compatibility


class IngestRequest(BaseModel):
    regulation: RegulationType
    url: str
    article_no: Optional[int] = None
    title: Optional[str] = None
    content: Optional[str] = None


class IngestResponse(BaseModel):
    success: bool
    message: str
    chunks_created: int = 0


class HealthResponse(BaseModel):
    status: str
    qdrant: str
    version: str
    embedding_model: str
    llm_model: str
