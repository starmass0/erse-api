from openai import OpenAI
from groq import Groq
from typing import Optional
import logging

from config import get_settings
from models import Citation

settings = get_settings()
logger = logging.getLogger(__name__)

_openai_client: Optional[OpenAI] = None
_groq_client: Optional[Groq] = None


def get_openai_client() -> OpenAI:
    """Get or create OpenAI client."""
    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAI(api_key=settings.openai_api_key)
    return _openai_client


def get_groq_client() -> Groq:
    """Get or create Groq client."""
    global _groq_client
    if _groq_client is None:
        _groq_client = Groq(api_key=settings.groq_api_key)
    return _groq_client


def synthesize_answer(
    question: str,
    chunks: list[dict],
    language: str = "en",
    mode: str = "detailed",
) -> tuple[str, list[Citation], float]:
    """Generate an answer using retrieved chunks."""

    if not chunks:
        return "I couldn't find relevant information to answer your question.", [], 0.0

    # Build context from chunks
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        article_info = f"Article {chunk['article_no']}" if chunk.get('article_no') else "Section"
        context_parts.append(
            f"[Source {i}] {chunk['regulation'].upper()} - {article_info}: {chunk.get('title', '')}\n"
            f"{chunk['content']}\n"
        )

    context = "\n---\n".join(context_parts)

    # Build prompt
    length_instruction = "Be concise (2-3 paragraphs max)." if mode == "short" else "Provide a comprehensive answer."

    system_prompt = f"""You are ERSE (European Regulatory Source Engine), an expert assistant on EU regulations.
Your task is to answer questions based ONLY on the provided regulatory sources.

Guidelines:
- Answer in {language}
- {length_instruction}
- Always cite specific articles when referencing information (e.g., "According to GDPR Article 7...")
- If the sources don't contain enough information, say so clearly
- Be precise and factual - this is legal/regulatory information
- Do not make up information not in the sources"""

    user_prompt = f"""Question: {question}

Relevant Regulatory Sources:
{context}

Please provide an accurate answer based on these sources."""

    try:
        if settings.llm_provider == "openai":
            client = get_openai_client()
            response = client.chat.completions.create(
                model=settings.llm_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.3,
                max_tokens=1500 if mode == "detailed" else 500,
            )
        else:
            client = get_groq_client()
            response = client.chat.completions.create(
                model=settings.llm_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.3,
                max_tokens=1500 if mode == "detailed" else 500,
            )

        answer = response.choices[0].message.content

    except Exception as e:
        logger.error(f"LLM API error: {e}")
        return f"Error generating answer: {str(e)}", [], 0.0

    # Build citations
    citations = []
    for chunk in chunks:
        citations.append(Citation(
            regulation=chunk["regulation"].upper(),
            article=str(chunk.get("article_no", "N/A")),
            title=chunk.get("title", ""),
            excerpt=chunk["content"][:300] + "..." if len(chunk["content"]) > 300 else chunk["content"],
            url=chunk.get("url", ""),
            relevance_score=chunk.get("score", 0.0),
        ))

    # Calculate confidence based on average relevance score
    avg_score = sum(c.relevance_score for c in citations) / len(citations) if citations else 0
    confidence = min(avg_score * 1.2, 1.0)  # Scale up slightly, cap at 1.0

    return answer, citations, confidence
