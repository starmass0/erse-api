from qdrant_client.models import PointStruct
from typing import Optional
import hashlib
import logging
import requests
import uuid
from bs4 import BeautifulSoup
import re

from config import get_settings
from services.embeddings import get_embedding
from services.retrieval import get_qdrant_client, ensure_collection_exists

settings = get_settings()
logger = logging.getLogger(__name__)


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> list[str]:
    """Split text into overlapping chunks."""
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size

        # Try to break at sentence boundary
        if end < len(text):
            # Look for sentence ending in last 100 chars
            last_period = text.rfind('. ', end - 100, end)
            if last_period > start:
                end = last_period + 1

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        start = end - overlap
        if start < 0:
            start = 0

    return chunks


def generate_point_id(content: str, regulation: str, article_no: Optional[int]) -> str:
    """Generate a deterministic UUID for a point."""
    key = f"{regulation}_{article_no}_{content[:100]}"
    # Create a UUID from the MD5 hash (UUID v3/v5 style)
    hash_bytes = hashlib.md5(key.encode()).digest()
    return str(uuid.UUID(bytes=hash_bytes))


def scrape_gdpr_article(url: str) -> dict:
    """Scrape a GDPR article from gdpr-info.eu."""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        # Extract title
        title_elem = soup.find('h1', class_='entry-title')
        title = title_elem.get_text(strip=True) if title_elem else ""

        # Extract article number from URL or title
        article_no = None
        match = re.search(r'art-(\d+)', url)
        if match:
            article_no = int(match.group(1))

        # Extract content
        content_elem = soup.find('div', class_='entry-content')
        if content_elem:
            # Remove scripts, styles, and navigation elements
            for tag in content_elem.find_all(['script', 'style', 'nav']):
                tag.decompose()
            # Remove sidebar and footer elements
            for tag in content_elem.find_all(class_=['nav-links', 'entry-meta', 'toc', 'post-navigation']):
                tag.decompose()
            content = content_elem.get_text(separator='\n', strip=True)

            # Clean up common footer/navigation text
            # First, remove recital sections entirely
            content = re.sub(r'Suitable Recitals.*$', '', content, flags=re.DOTALL)

            lines = content.split('\n')
            cleaned_lines = []
            skip_patterns = [
                'GDPR', 'Table of contents', 'Report error',
                '←', '→', 'Suitable Recitals', 'Recitals'
            ]
            skip_next = False
            for line in lines:
                line = line.strip()
                # Skip empty lines
                if not line:
                    continue
                # Skip navigation lines
                if any(line.startswith(p) or line == p for p in skip_patterns):
                    continue
                # Skip article navigation (Art. X GDPR)
                if re.match(r'^Art\.\s*\d+\s*GDPR$', line):
                    continue
                # Skip standalone parentheses with numbers (recital refs)
                if re.match(r'^\(?\d+\)?$', line):
                    continue
                # Skip recital titles
                if re.match(r'^\(\s*\d+\s*\)', line):
                    continue
                # Skip lines that are just recital names
                if any(x in line.lower() for x in ['recital', 'conditions for consent', 'burden of proof', 'freely given']):
                    if len(line) < 60:
                        continue
                cleaned_lines.append(line)
            content = '\n'.join(cleaned_lines)
        else:
            content = ""

        return {
            "title": title,
            "article_no": article_no,
            "content": content,
            "url": url,
        }
    except Exception as e:
        logger.error(f"Error scraping {url}: {e}")
        return {}


def ingest_document(
    regulation: str,
    content: str,
    article_no: Optional[int] = None,
    title: str = "",
    url: str = "",
) -> int:
    """Ingest a document into Qdrant."""
    ensure_collection_exists()
    client = get_qdrant_client()

    # Chunk the content
    chunks = chunk_text(content)

    # Generate embeddings for all chunks
    embeddings = get_embedding(chunks)

    # Create points
    points = []
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        point_id = generate_point_id(chunk, regulation, article_no)

        points.append(PointStruct(
            id=point_id,
            vector=embedding.tolist(),
            payload={
                "content": chunk,
                "regulation": regulation.lower(),
                "article_no": article_no,
                "title": title,
                "url": url,
                "chunk_index": i,
            }
        ))

    # Upsert to Qdrant
    if points:
        client.upsert(
            collection_name=settings.qdrant_collection,
            points=points,
        )
        logger.info(f"Ingested {len(points)} chunks for {regulation} Article {article_no}")

    return len(points)


def ingest_from_url(regulation: str, url: str, article_no: Optional[int] = None) -> int:
    """Scrape and ingest content from a URL."""
    if "gdpr-info.eu" in url:
        data = scrape_gdpr_article(url)
        if data:
            return ingest_document(
                regulation=regulation,
                content=data.get("content", ""),
                article_no=data.get("article_no") or article_no,
                title=data.get("title", ""),
                url=url,
            )
    return 0


def ingest_gdpr_batch(articles: list[int] = None):
    """Ingest multiple GDPR articles."""
    if articles is None:
        articles = list(range(1, 100))  # Articles 1-99

    total_chunks = 0
    for art_no in articles:
        url = f"https://gdpr-info.eu/art-{art_no}-gdpr/"
        try:
            chunks = ingest_from_url("gdpr", url, art_no)
            total_chunks += chunks
            logger.info(f"Ingested GDPR Article {art_no}: {chunks} chunks")
        except Exception as e:
            logger.error(f"Failed to ingest GDPR Article {art_no}: {e}")

    return total_chunks


def scrape_eurlex_article(url: str, regulation: str) -> dict:
    """Scrape an article from EUR-Lex or similar sources."""
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (compatible; ERSE/2.0)'}
        response = requests.get(url, timeout=15, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        # Extract title
        title = ""
        title_elem = soup.find('h1') or soup.find('title')
        if title_elem:
            title = title_elem.get_text(strip=True)

        # Extract main content
        content = ""
        # Try various content containers
        content_selectors = [
            'div.eli-main-body',
            'div#TexteOnly',
            'div.texte',
            'article',
            'main',
            'div.content'
        ]
        for selector in content_selectors:
            content_elem = soup.select_one(selector)
            if content_elem:
                for tag in content_elem.find_all(['script', 'style', 'nav', 'header', 'footer']):
                    tag.decompose()
                content = content_elem.get_text(separator='\n', strip=True)
                break

        if not content:
            # Fallback: get body content
            body = soup.find('body')
            if body:
                for tag in body.find_all(['script', 'style', 'nav', 'header', 'footer']):
                    tag.decompose()
                content = body.get_text(separator='\n', strip=True)

        return {
            "title": title,
            "content": content,
            "url": url,
        }
    except Exception as e:
        logger.error(f"Error scraping {url}: {e}")
        return {}


def ingest_dsa_batch():
    """Ingest Digital Services Act articles."""
    # DSA key articles and sections
    dsa_sources = [
        ("https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:32022R2065", "Digital Services Act - Full Text"),
    ]

    total_chunks = 0
    for url, title in dsa_sources:
        try:
            data = scrape_eurlex_article(url, "dsa")
            if data and data.get("content"):
                chunks = ingest_document(
                    regulation="dsa",
                    content=data["content"],
                    title=title,
                    url=url,
                )
                total_chunks += chunks
                logger.info(f"Ingested DSA: {chunks} chunks")
        except Exception as e:
            logger.error(f"Failed to ingest DSA: {e}")

    return total_chunks


def ingest_nis2_batch():
    """Ingest NIS2 Directive articles."""
    nis2_sources = [
        ("https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:32022L2555", "NIS2 Directive - Full Text"),
    ]

    total_chunks = 0
    for url, title in nis2_sources:
        try:
            data = scrape_eurlex_article(url, "nis2")
            if data and data.get("content"):
                chunks = ingest_document(
                    regulation="nis2",
                    content=data["content"],
                    title=title,
                    url=url,
                )
                total_chunks += chunks
                logger.info(f"Ingested NIS2: {chunks} chunks")
        except Exception as e:
            logger.error(f"Failed to ingest NIS2: {e}")

    return total_chunks


def ingest_aiact_batch():
    """Ingest AI Act articles."""
    aiact_sources = [
        ("https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=OJ:L_202401689", "AI Act - Full Text"),
    ]

    total_chunks = 0
    for url, title in aiact_sources:
        try:
            data = scrape_eurlex_article(url, "aiact")
            if data and data.get("content"):
                chunks = ingest_document(
                    regulation="aiact",
                    content=data["content"],
                    title=title,
                    url=url,
                )
                total_chunks += chunks
                logger.info(f"Ingested AI Act: {chunks} chunks")
        except Exception as e:
            logger.error(f"Failed to ingest AI Act: {e}")

    return total_chunks
