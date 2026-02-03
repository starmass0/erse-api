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


def clean_regulation_text(text: str) -> str:
    """Clean scraped regulation text by removing common junk patterns."""
    # Lines to skip
    skip_patterns = [
        'cookie', 'privacy policy', 'terms of', 'subscribe', 'newsletter',
        'follow us', 'share this', 'tweet', 'facebook', 'linkedin',
        'hiring', 'job', 'apply', 'salary', 'career',
        'advertisement', 'sponsored', 'related articles',
        'read more', 'see also', 'click here', 'learn more',
        'sign up', 'register', 'login', 'account',
    ]

    lines = text.split('\n')
    cleaned = []
    for line in lines:
        line = line.strip()
        if not line or len(line) < 10:
            continue
        # Skip junk lines
        if any(pattern in line.lower() for pattern in skip_patterns):
            continue
        # Skip lines that are just navigation/links
        if line.startswith('→') or line.startswith('←') or line.startswith('|'):
            continue
        cleaned.append(line)

    return '\n'.join(cleaned)


def ingest_dsa_batch():
    """Ingest Digital Services Act - clean official content."""
    # DSA key content - official EU sources
    dsa_content = """
Digital Services Act (DSA) - Regulation (EU) 2022/2065

The Digital Services Act (DSA) is a European Union regulation that aims to create a safer digital space where the fundamental rights of users are protected and to establish a level playing field for businesses.

CHAPTER I - GENERAL PROVISIONS

Article 1 - Subject matter
This Regulation lays down harmonised rules on the provision of intermediary services in the internal market. In particular, it establishes:
(a) a framework for the conditional exemption from liability of providers of intermediary services;
(b) rules on specific due diligence obligations tailored to certain specific categories of providers of intermediary services;
(c) rules on the implementation and enforcement of this Regulation, including as regards the cooperation of and coordination between the competent authorities.

Article 2 - Scope
This Regulation applies to intermediary services offered to recipients of the service that have their place of establishment or residence in the Union, irrespective of where the providers of those intermediary services have their place of establishment.

Article 3 - Definitions
For the purpose of this Regulation, the following definitions apply:
(a) 'information society service' means a service within the meaning of Article 1(1), point (b), of Directive (EU) 2015/1535;
(b) 'recipient of the service' means any natural or legal person who uses an intermediary service, in particular for the purposes of seeking information or making it accessible;
(c) 'consumer' means any natural person who is acting for purposes which are outside his or her trade, business, craft, or profession;
(d) 'intermediary service' means one of the following information society services: mere conduit, caching, hosting;
(e) 'illegal content' means any information that, in itself or in relation to an activity, including the sale of products or the provision of services, is not in compliance with Union law or the law of any Member State;
(f) 'online platform' means a hosting service that, at the request of a recipient of the service, stores and disseminates information to the public;
(g) 'online search engine' means an intermediary service that allows users to input queries in order to perform searches of, in principle, all websites;
(h) 'very large online platform' means an online platform which has a number of average monthly active recipients of the service in the Union equal to or higher than 45 million;
(i) 'very large online search engine' means an online search engine which has a number of average monthly active recipients of the service in the Union equal to or higher than 45 million.

CHAPTER II - LIABILITY OF PROVIDERS OF INTERMEDIARY SERVICES

Article 4 - Mere conduit
Providers of mere conduit services shall not be liable for information transmitted.

Article 5 - Caching
Providers of caching services shall not be liable for automatic, intermediate and temporary storage of information.

Article 6 - Hosting
Providers of hosting services shall not be liable for information stored at the request of a recipient of the service.

CHAPTER III - DUE DILIGENCE OBLIGATIONS

Article 11 - Points of contact
Providers of intermediary services shall designate a single point of contact.

Article 12 - Legal representatives
Providers of intermediary services which do not have an establishment in the Union but offer services in the Union shall designate a legal representative in one of the Member States.

Article 14 - Terms and conditions
Providers of intermediary services shall include information on any restrictions that they impose in relation to the use of their service.

Article 16 - Notice and action mechanisms
Providers of hosting services shall put mechanisms in place to allow any individual or entity to notify them of illegal content.

Article 17 - Statement of reasons
Providers of hosting services shall provide a clear and specific statement of reasons to any affected recipient of the service when they restrict the service.

CHAPTER IV - ADDITIONAL PROVISIONS FOR ONLINE PLATFORMS

Article 20 - Internal complaint-handling system
Online platforms shall provide recipients of the service with access to an internal complaint-handling system.

Article 22 - Trusted flaggers
Online platforms shall take the necessary technical and organisational measures to ensure that notices submitted by trusted flaggers are given priority.

Article 24 - Online interface design and organisation
Online platforms shall not design, organise or operate their online interfaces in a way that deceives or manipulates the recipients of their service or in a way that otherwise materially distorts or impairs the ability of the recipients of their service to make free and informed decisions.

Article 26 - Advertising on online platforms
Online platforms that display advertising shall ensure that the recipients of the service can identify clearly and unambiguously the advertising as such.

Article 27 - Recommender system transparency
Online platforms that use recommender systems shall set out in their terms and conditions the main parameters used in their recommender systems.

CHAPTER V - ADDITIONAL OBLIGATIONS FOR VERY LARGE ONLINE PLATFORMS AND SEARCH ENGINES

Article 34 - Risk assessment
Very large online platforms and very large online search engines shall diligently identify, analyse and assess any systemic risks in the Union.

Article 35 - Mitigation of risks
Very large online platforms and very large online search engines shall put in place reasonable, proportionate and effective mitigation measures.

Article 37 - Independent audit
Very large online platforms and very large online search engines shall be subject, at their own expense and at least once a year, to independent audits.

Article 40 - Data access and scrutiny
Very large online platforms and very large online search engines shall provide access to data to vetted researchers.
"""

    total_chunks = 0
    try:
        chunks = ingest_document(
            regulation="dsa",
            content=dsa_content.strip(),
            title="Digital Services Act (DSA) - Regulation (EU) 2022/2065",
            url="https://eur-lex.europa.eu/eli/reg/2022/2065/oj",
        )
        total_chunks += chunks
        logger.info(f"Ingested DSA: {chunks} chunks")
    except Exception as e:
        logger.error(f"Failed to ingest DSA: {e}")

    return total_chunks


def ingest_nis2_batch():
    """Ingest NIS2 Directive - clean official content."""
    nis2_content = """
NIS2 Directive - Directive (EU) 2022/2555

The NIS2 Directive (Network and Information Security Directive) is European Union legislation on cybersecurity that aims to achieve a high common level of cybersecurity across the Union.

CHAPTER I - GENERAL PROVISIONS

Article 1 - Subject matter
This Directive lays down measures that aim to achieve a high common level of cybersecurity across the Union, with a view to improving the functioning of the internal market.

Article 2 - Scope
This Directive applies to public and private entities which qualify as medium-sized enterprises or exceed the ceilings for medium-sized enterprises, and which provide their services or carry out their activities within the Union.

The Directive covers entities in the following sectors:
- Energy (electricity, oil, gas, hydrogen)
- Transport (air, rail, water, road)
- Banking and financial market infrastructures
- Health sector
- Drinking water and waste water
- Digital infrastructure
- ICT service management
- Public administration
- Space

Article 3 - Definitions
Essential entities include:
- Large enterprises in high-criticality sectors
- Qualified trust service providers
- Top-level domain name registries
- DNS service providers
- Providers of public electronic communications networks

Important entities include:
- Medium-sized enterprises in high-criticality sectors
- Entities in other critical sectors

CHAPTER II - COORDINATED CYBERSECURITY FRAMEWORKS

Article 7 - National cybersecurity strategy
Each Member State shall adopt a national cybersecurity strategy providing strategic objectives, policies, and regulatory measures.

Article 8 - Competent authorities and single points of contact
Each Member State shall designate one or more competent authorities responsible for cybersecurity and the supervision of the application of this Directive.

Article 9 - National cyber crisis management frameworks
Each Member State shall designate or establish one or more competent authorities responsible for the management of large-scale cybersecurity incidents and crises.

Article 10 - Computer security incident response teams (CSIRTs)
Each Member State shall designate or establish one or more CSIRTs covering at least the sectors and services covered by this Directive.

CHAPTER III - COOPERATION

Article 13 - Cooperation Group
A Cooperation Group is established to support and facilitate strategic cooperation and the exchange of information among Member States.

Article 14 - CSIRTs network
A network of national CSIRTs is established to contribute to the development of confidence and trust and to promote swift and effective operational cooperation.

Article 15 - European cyber crisis liaison organisation network (EU-CyCLONe)
EU-CyCLONe is established to support the coordinated management of large-scale cybersecurity incidents and crises at operational level.

CHAPTER IV - CYBERSECURITY RISK-MANAGEMENT AND REPORTING OBLIGATIONS

Article 21 - Cybersecurity risk-management measures
Member States shall ensure that essential and important entities take appropriate and proportionate technical, operational and organisational measures to manage the risks posed to the security of network and information systems.

These measures shall include:
(a) policies on risk analysis and information system security;
(b) incident handling;
(c) business continuity and crisis management;
(d) supply chain security;
(e) security in network and information systems acquisition, development and maintenance;
(f) policies and procedures to assess the effectiveness of cybersecurity risk-management measures;
(g) basic cyber hygiene practices and cybersecurity training;
(h) policies and procedures regarding the use of cryptography and encryption;
(i) human resources security, access control policies and asset management;
(j) the use of multi-factor authentication or continuous authentication solutions.

Article 23 - Reporting obligations
Member States shall ensure that essential and important entities notify, without undue delay, significant incidents to the competent authority or the CSIRT.

An early warning shall be submitted within 24 hours of becoming aware of the significant incident.
An incident notification shall be submitted within 72 hours of becoming aware of the significant incident.
A final report shall be submitted not later than one month after the submission of the incident notification.

CHAPTER V - JURISDICTION AND REGISTRATION

Article 26 - Jurisdiction and territoriality
Entities shall be deemed to be under the jurisdiction of the Member State in which they have their main establishment.

Article 27 - Registry of entities
ENISA shall create and maintain a registry of DNS service providers, TLD name registries, and cloud computing service providers.

CHAPTER VI - INFORMATION SHARING

Article 29 - Cybersecurity information-sharing arrangements
Member States shall ensure that entities may exchange relevant cybersecurity information among themselves.

CHAPTER VII - SUPERVISION AND ENFORCEMENT

Article 32 - Supervisory measures in respect of essential entities
Competent authorities shall have the power to subject essential entities to on-site inspections and off-site supervision.

Article 33 - Supervisory measures in respect of important entities
Competent authorities shall have the power to subject important entities to supervisory measures when provided with evidence that an important entity does not comply with this Directive.

Article 34 - General conditions for imposing administrative fines
Member States shall ensure that administrative fines imposed are effective, proportionate and dissuasive.

For essential entities: maximum fine of at least EUR 10,000,000 or 2% of total worldwide annual turnover.
For important entities: maximum fine of at least EUR 7,000,000 or 1.4% of total worldwide annual turnover.
"""

    total_chunks = 0
    try:
        chunks = ingest_document(
            regulation="nis2",
            content=nis2_content.strip(),
            title="NIS2 Directive - Directive (EU) 2022/2555",
            url="https://eur-lex.europa.eu/eli/dir/2022/2555/oj",
        )
        total_chunks += chunks
        logger.info(f"Ingested NIS2: {chunks} chunks")
    except Exception as e:
        logger.error(f"Failed to ingest NIS2: {e}")

    return total_chunks


def ingest_aiact_batch():
    """Ingest AI Act - clean official content."""
    aiact_content = """
AI Act - Regulation (EU) 2024/1689

The AI Act is the European Union's comprehensive regulatory framework for artificial intelligence. It establishes harmonised rules for the placing on the market, putting into service, and use of AI systems in the Union.

CHAPTER I - GENERAL PROVISIONS

Article 1 - Subject matter
This Regulation lays down harmonised rules for the placing on the market, the putting into service and the use of artificial intelligence systems (AI systems) in the Union.

This Regulation pursues the following objectives:
(a) improve the functioning of the internal market by laying down a uniform legal framework;
(b) promote the uptake of human-centric and trustworthy artificial intelligence;
(c) ensure a high level of protection of health, safety, fundamental rights, democracy and rule of law.

Article 2 - Scope
This Regulation applies to providers, deployers, importers, distributors, and operators of AI systems placed on the market or put into service in the Union.

Article 3 - Definitions
Key definitions:
- 'AI system' means a machine-based system designed to operate with varying levels of autonomy, that may exhibit adaptiveness after deployment and that infers, from the input it receives, how to generate outputs such as predictions, content, recommendations, or decisions.
- 'provider' means a natural or legal person that develops an AI system or a general-purpose AI model and places it on the market or puts the AI system into service under its own name or trademark.
- 'deployer' means a natural or legal person that uses an AI system under its authority.
- 'high-risk AI system' means an AI system that poses significant risks to health, safety, or fundamental rights.

CHAPTER II - PROHIBITED AI PRACTICES

Article 5 - Prohibited AI practices
The following AI practices are prohibited:
(a) AI systems that deploy subliminal techniques or purposefully manipulative or deceptive techniques to distort behaviour causing significant harm;
(b) AI systems that exploit vulnerabilities of persons due to their age, disability or specific social or economic situation;
(c) AI systems for social scoring by public authorities;
(d) AI systems for risk assessment of natural persons to predict criminal offences based solely on profiling;
(e) AI systems that create facial recognition databases through untargeted scraping;
(f) AI systems that infer emotions in workplaces and educational institutions, except for safety or medical reasons;
(g) Biometric categorisation systems that categorise persons based on sensitive characteristics;
(h) Real-time remote biometric identification systems in publicly accessible spaces for law enforcement, except in certain limited cases.

CHAPTER III - HIGH-RISK AI SYSTEMS

Article 6 - Classification rules for high-risk AI systems
An AI system is considered high-risk if it:
(a) is intended to be used as a safety component of a product covered by Union harmonisation legislation; or
(b) falls within the areas listed in Annex III.

Annex III high-risk areas include:
1. Biometrics (remote biometric identification, biometric categorisation, emotion recognition)
2. Critical infrastructure (management and operation of road traffic, water, gas, heating, electricity)
3. Education and vocational training (determining access, assessing learning outcomes)
4. Employment (recruitment, work-related decisions, task allocation, monitoring)
5. Access to essential services (creditworthiness assessment, risk assessment for insurance)
6. Law enforcement (risk assessment, polygraphs, evidence reliability assessment)
7. Migration and border control (risk assessment, document authenticity verification)
8. Administration of justice (researching and interpreting facts and law)

Article 9 - Risk management system
Providers of high-risk AI systems shall establish, implement, document and maintain a risk management system.

Article 10 - Data and data governance
Training, validation and testing data sets shall be subject to appropriate data governance and management practices.

Article 11 - Technical documentation
The technical documentation shall be drawn up before the AI system is placed on the market and shall be kept up to date.

Article 13 - Transparency and provision of information to deployers
High-risk AI systems shall be designed and developed to ensure that their operation is sufficiently transparent to enable deployers to interpret output and use it appropriately.

Article 14 - Human oversight
High-risk AI systems shall be designed and developed in such a way that they can be effectively overseen by natural persons.

Article 15 - Accuracy, robustness and cybersecurity
High-risk AI systems shall be designed and developed to achieve an appropriate level of accuracy, robustness, and cybersecurity.

CHAPTER IV - TRANSPARENCY OBLIGATIONS

Article 50 - Transparency obligations for certain AI systems
Providers shall ensure that AI systems intended to interact with natural persons are designed to inform persons that they are interacting with an AI system.

AI systems that generate synthetic audio, image, video or text content shall ensure that the outputs are marked in a machine-readable format and detectable as artificially generated.

Deployers of emotion recognition or biometric categorisation systems shall inform natural persons exposed to such systems.

CHAPTER V - GENERAL-PURPOSE AI MODELS

Article 51 - Classification of general-purpose AI models
A general-purpose AI model shall be classified as a general-purpose AI model with systemic risk if:
(a) it has high impact capabilities; or
(b) the cumulative amount of compute used for its training exceeds 10^25 floating point operations.

Article 53 - Obligations for providers of general-purpose AI models
Providers of general-purpose AI models shall:
(a) draw up and keep up-to-date technical documentation;
(b) draw up and keep up-to-date information and documentation for downstream providers;
(c) put in place a policy to respect Union copyright law;
(d) draw up and make publicly available a sufficiently detailed summary about the content used for training.

Article 55 - Obligations for providers of general-purpose AI models with systemic risk
In addition to the obligations in Article 53, providers shall:
(a) perform model evaluation;
(b) assess and mitigate possible systemic risks;
(c) keep track of, document, and report serious incidents;
(d) ensure an adequate level of cybersecurity protection.

CHAPTER VII - GOVERNANCE

Article 64 - AI Office
The AI Office is established within the Commission to implement and enforce this Regulation with respect to general-purpose AI models.

Article 65 - European Artificial Intelligence Board
A European Artificial Intelligence Board is established as a body of the Union.

CHAPTER IX - PENALTIES

Article 99 - Penalties
Member States shall lay down the rules on penalties applicable to infringements of this Regulation.

For prohibited AI practices: administrative fines up to EUR 35,000,000 or 7% of worldwide annual turnover.
For non-compliance with high-risk requirements: administrative fines up to EUR 15,000,000 or 3% of worldwide annual turnover.
For supplying incorrect information: administrative fines up to EUR 7,500,000 or 1% of worldwide annual turnover.

CHAPTER XII - FINAL PROVISIONS

Article 113 - Entry into force and application
This Regulation enters into force on the twentieth day following publication in the Official Journal.

It shall apply from 2 August 2026, with the following exceptions:
- Prohibited practices: apply from 2 February 2025
- General-purpose AI models: apply from 2 August 2025
- High-risk AI systems (Annex III): apply from 2 August 2027 for certain categories
"""

    total_chunks = 0
    try:
        chunks = ingest_document(
            regulation="aiact",
            content=aiact_content.strip(),
            title="AI Act - Regulation (EU) 2024/1689",
            url="https://eur-lex.europa.eu/eli/reg/2024/1689/oj",
        )
        total_chunks += chunks
        logger.info(f"Ingested AI Act: {chunks} chunks")
    except Exception as e:
        logger.error(f"Failed to ingest AI Act: {e}")

    return total_chunks
