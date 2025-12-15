import os
import json
import logging
import re
from typing import List

from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq
from pypdf import PdfReader
import trafilatura
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# >>> NOWE IMPORTY DLA YOUTUBE: Dodano RequestBlocked
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.formatters import TextFormatter
from youtube_transcript_api._errors import NoTranscriptFound, TranscriptsDisabled, YouTubeRequestFailed, RequestBlocked
from urllib.parse import urlparse, parse_qs
# <<<

# ---- Logger configuration ----
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("flashcards-backend")

# ---- Settings ----
MAX_CHARS_PER_CHUNK = 4000
MAX_CHUNKS = 10
MAX_REQUESTED_CARDS = 200

# TF-IDF settings
MAX_SENTENCES = 100
MAX_WORDS = 15000
LANGUAGE = "english" # Language for stop words removal (TF-IDF)
YOUTUBE_SUMMARY_THRESHOLD = 1000 # Word count threshold for transcript summarization

# ---- Groq client init ----
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    logger.warning("GROQ_API_KEY not set in environment — requests to AI will fail.")
client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

# ---- App initialization ----
app = FastAPI(title="Flashcards API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- Models ----
class FlashcardRequest(BaseModel):
    text: str
    count: int = 10

class URLFlashcardRequest(BaseModel):
    url: str
    count: int = 10

# ---- Utils ----
def split_text(text: str, max_chars: int = MAX_CHARS_PER_CHUNK) -> List[str]:
    chunks: List[str] = []
    current = ""
    for line in text.split("\n"):
        added = ("\n" + line) if current else line
        if len(current) + len(added) > max_chars:
            if current.strip():
                chunks.append(current.strip())
            current = line
        else:
            current += added
    if current.strip():
        chunks.append(current.strip())
    return chunks[:MAX_CHUNKS]

def extract_text_from_pdf(file: UploadFile) -> str:
    reader = PdfReader(file.file)
    text_parts: List[str] = []
    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            text_parts.append(extracted)
    return "\n\n".join(text_parts)

# ZMODYFIKOWANA FUNKCJA: Dodano User-Agent i angielskie błędy
def fetch_clean_text(url: str) -> str:
    # Adding User-Agent header to pretend to be a browser and prevent 403 block
    user_agent_header = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    
    downloaded = trafilatura.fetch_url(url, user_agent=user_agent_header)
    
    if not downloaded:
        logger.error(f"Failed to fetch URL content for {url} (downloaded is None)")
        # English translation of "Serwer docelowy zablokował dostęp (403 Forbidden) lub URL jest nieosiągalny."
        raise RuntimeError("Target server blocked access (403 Forbidden) or URL is unreachable.")
        
    text = trafilatura.extract(
        downloaded,
        include_comments=False,
        include_tables=False,
        include_formatting=False
    )
    if not text:
        # English translation of "Nie udało się wyodrębnić treści ze strony..."
        raise RuntimeError("Could not extract content from the page (page may be empty or blocked).")

    text = re.sub(r'\[[^\]]*\]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()

    words = text.split()
    if len(words) > MAX_WORDS:
        text = " ".join(words[:MAX_WORDS])
    return text

def split_into_sentences(text: str):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if len(s.split()) > 5]

def summarize_tfidf(text: str, max_sentences: int = MAX_SENTENCES) -> str:
    sentences = split_into_sentences(text)
    if len(sentences) <= max_sentences:
        return " ".join(sentences)

    vectorizer = TfidfVectorizer(
        stop_words=LANGUAGE,
        ngram_range=(1, 2)
    )
    tfidf = vectorizer.fit_transform(sentences)
    sentence_scores = np.asarray(tfidf.sum(axis=1)).ravel()
    top_indices = np.argsort(sentence_scores)[-max_sentences:]
    top_indices = sorted(top_indices)
    summary = " ".join(sentences[i] for i in top_indices)
    return summary


# >>> NEW YOUTUBE UTILS <<<

# Full list of language codes (Priority: EN > PL > Globally common)
WSZYSTKIE_KODY_PRIORYTET = [
    # User priorities
    'en',  # English (Global priority)
    'pl',  # Polish (High user priority)
    
    # Globally common languages and YouTube (performance optimization)
    'es',  # Spanish
    'pt',  # Portuguese
    'ru',  # Russian
    'zh',  # Chinese (simplified and traditional)
    'de',  # German
    'fr',  # French
    'ja',  # Japanese
    'ko',  # Korean
    'it',  # Italian
    'hi',  # Hindi
    'id',  # Indonesian
    'tr',  # Turkish
    'ar',  # Arabic
    'th',  # Thai
    'vi',  # Vietnamese
    'nl',  # Dutch
    'sv',  # Swedish

    # Remaining languages (for maximum coverage, sorted by code)
    'af', 'sq', 'am', 'as', 'ay', 'az', 'eu', 'bn', 'bs', 'bg', 'my', 
    'ca', 'ceb', 'hr', 'cs', 'da', 'et', 'fil', 'fi', 'gl', 'ka', 'el', 
    'gu', 'ha', 'iw', 'hu', 'is', 'ig', 'ga', 'jv', 'kn', 'kk', 'km', 
    'ky', 'lo', 'lv', 'lt', 'mk', 'mg', 'ms', 'ml', 'mr', 'mn', 'ne', 
    'no', 'or', 'om', 'ps', 'fa', 'pa', 'ro', 'sm', 'sa', 'sr', 'sn', 
    'sd', 'si', 'sk', 'sl', 'so', 'su', 'sw', 'tl', 'ta', 'te', 'ti', 
    'uk', 'ur', 'uz', 'cy', 'xh', 'yi', 'yo', 'zu'
]

def wyodrebnij_video_id(url: str) -> str or None:
    """
    Extracts video ID from various YouTube link formats (watch, youtu.be, playlists).
    """
    query = urlparse(url)
    if query.hostname in ('www.youtube.com', 'youtube.com', 'm.youtube.com'):
        if query.path == '/watch':
            return parse_qs(query.query).get('v', [None])[0]
        if query.path.startswith('/embed/'):
            return query.path.split('/')[2]
    if query.hostname in ('youtu.be', 'www.youtu.be'):
        return query.path[1:]
    match = re.search(r'(?:v=|/v/|embed/|youtu\.be/)([a-zA-Z0-9_-]{11})', url)
    if match:
        return match.group(1)
    return None

def pobierz_transkrypt_w_pierwszym_jezyku(video_id: str) -> str:
    """
    Downloads transcript in the first found language from the priority list.
    """
    ytt_api = YouTubeTranscriptApi()
    formatter = TextFormatter()
    
    # API finds the FIRST available language from the 'WSZYSTKIE_KODY_PRIORYTET' list
    fetched_dane = ytt_api.fetch(video_id, languages=WSZYSTKIE_KODY_PRIORYTET)
    
    # Formatting using TextFormatter
    return formatter.format_transcript(fetched_dane)

# <<<

def call_model_generate(text: str, count: int) -> str:
    if not client:
        raise RuntimeError("Groq client not configured (GROQ_API_KEY missing).")

    system_prompt = """You are an expert educational AI designed to create high-retention Anki-style flashcards.

### CORE LOGIC (TOPIC vs. CONTENT):
Analyze the content inside the <user_input> tags and classify it into one of two modes:

**MODE A: TOPIC EXPANSION** (Triggered when input is short, abstract, or a title)
- Generate NEW examples, facts, vocabulary internally.

**MODE B: CONTENT EXTRACTION** (Triggered when input is long, detailed text)
- Extract facts ONLY from the provided text.

### LANGUAGE RULES:
- Target Language = input language
- For language learning, use Native -> Target

### JSON OUTPUT FORMAT:
Return strictly a JSON object with a "flashcards" array.
{
  "flashcards": [
    {"question": "Clear, specific question", "answer": "Concise answer"}
  ]
}
"""

    user_prompt = f"""Generate exactly {count} flashcards based on the text below.
Apply the TOPIC vs CONTENT logic strictly.

<user_input>
{text}
</user_input>
"""
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.6,
        response_format={"type": "json_object"}
    )
    return response.choices[0].message.content

def parse_model_chunk(raw: str) -> List[dict]:
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict) and "flashcards" in parsed:
            items = parsed["flashcards"]
            if isinstance(items, list):
                return [p for p in items if isinstance(p, dict) and "question" in p and "answer" in p]
        if isinstance(parsed, list):
            return [p for p in parsed if isinstance(p, dict)]
    except json.JSONDecodeError:
        logger.error("JSON Decode Error: %s", raw[:200])
    return []

def distribute_counts_across_chunks(total_count: int, chunks_count: int) -> List[int]:
    if chunks_count <= 0: return []
    base = total_count // chunks_count
    remainder = total_count % chunks_count
    distribution = []
    for i in range(chunks_count):
        per = base + (1 if i < remainder else 0)
        if per <= 0: per = 1
        distribution.append(per)
    while sum(distribution) > total_count:
        for i in range(len(distribution)-1, -1, -1):
            if distribution[i] > 0 and sum(distribution) > total_count:
                distribution[i] -= 1
    return [d for d in distribution if d > 0]


# ---- routes ----
@app.get("/")
def root():
    return {"status": "ok"}

@app.post("/flashcards")
def flashcards_from_text(data: FlashcardRequest):
    if not data.text or not data.text.strip():
        raise HTTPException(status_code=400, detail="Text is empty")
    if data.count <= 0:
        raise HTTPException(status_code=400, detail="Count must be positive")

    chunks = split_text(data.text)
    if len(chunks) == 0:
        raise HTTPException(status_code=400, detail="Text is empty after processing")

    distribution = distribute_counts_across_chunks(data.count, len(chunks))
    all_cards: List[dict] = []

    try:
        for chunk, per_count in zip(chunks, distribution):
            raw = call_model_generate(chunk, per_count)
            parsed = parse_model_chunk(raw)
            if len(parsed) > per_count:
                parsed = parsed[:per_count]
            all_cards.extend(parsed)

        return {"flashcards": all_cards}
    except Exception as e:
        logger.exception("Error generating flashcards from text")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/flashcards/pdf")
async def flashcards_from_pdf(file: UploadFile = File(...), count: int = Form(10)):
    if count <= 0:
        raise HTTPException(status_code=400, detail="Count must be positive")

    text = extract_text_from_pdf(file)
    if not text or not text.strip():
        raise HTTPException(status_code=400, detail="PDF does not contain text")

    chunks = split_text(text)
    if len(chunks) == 0:
        raise HTTPException(status_code=400, detail="PDF text too short")

    distribution = distribute_counts_across_chunks(count, len(chunks))
    all_cards: List[dict] = []

    try:
        for chunk, per_count in zip(chunks, distribution):
            raw = call_model_generate(chunk, per_count)
            parsed = parse_model_chunk(raw)
            if len(parsed) > per_count:
                parsed = parsed[:per_count]
            all_cards.extend(parsed)

        return {"chunks": len(chunks), "flashcards": all_cards}
    except Exception as e:
        logger.exception("Error generating flashcards from pdf")
        raise HTTPException(status_code=500, detail=str(e))

# ZMODYFIKOWANY ENDPOINT flashcards_from_url
@app.post("/flashcards/url")
def flashcards_from_url(data: URLFlashcardRequest):
    if not data.url or not data.url.strip():
        raise HTTPException(status_code=400, detail="URL is empty")
    if data.count <= 0:
        raise HTTPException(status_code=400, detail="Count must be positive")

    try:
        # 1️⃣ Fetch and clean text (with added User-Agent)
        text = fetch_clean_text(data.url)
        
        # Sprawdzanie, czy fetch_clean_text zwróciło pusty tekst po parsowaniu
        if not text:
            raise RuntimeError("Could not extract content from the page or it is empty.") 

        # 2️⃣ TF-IDF summarization
        summary = summarize_tfidf(text)

        # 3️⃣ Splitting into chunks
        chunks = split_text(summary)
        if len(chunks) == 0:
            raise HTTPException(status_code=400, detail="Summary too short")

        # 4️⃣ Distributing card counts
        distribution = distribute_counts_across_chunks(data.count, len(chunks))
        all_cards: List[dict] = []

        # 5️⃣ Sending to Groq AI
        for chunk, per_count in zip(chunks, distribution):
            raw = call_model_generate(chunk, per_count)
            parsed = parse_model_chunk(raw)
            if len(parsed) > per_count:
                parsed = parsed[:per_count]
            all_cards.extend(parsed)

        return {"url": data.url, "flashcards": all_cards}

    except RuntimeError as e:
        # Catching RuntimeError (e.g., 403 Forbidden from trafilatura)
        logger.error(f"Runtime error fetching URL content: {e}")
        # Returning 400 Bad Request with a clear message
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Error generating flashcards from URL")
        raise HTTPException(status_code=500, detail=str(e))

# ==============================================================================
#                 YOUTUBE ENDPOINT
# ==============================================================================

@app.post("/flashcards/youtube")
def flashcards_from_youtube(data: URLFlashcardRequest):
    """
    Generates flashcards based on a YouTube video transcript. 
    Summarizes the transcript if it exceeds 1000 words.
    """
    url = data.url
    
    if not url or not url.strip():
        raise HTTPException(status_code=400, detail="URL is empty")
    if data.count <= 0:
        raise HTTPException(status_code=400, detail="Count must be positive")

    # 1. Extract Video ID
    video_id = wyodrebnij_video_id(url)
    if not video_id:
        raise HTTPException(
            status_code=400, 
            detail="Bad URL: Could not extract video ID."
        )

    # 2. Download transcript (YouTube error handling)
    try:
        transkrypt = pobierz_transkrypt_w_pierwszym_jezyku(video_id)
        
    except NoTranscriptFound:
        raise HTTPException(
            status_code=404, 
            detail="There is no transcript available for this video."
        )
    except TranscriptsDisabled:
        raise HTTPException(
            status_code=403, 
            detail="Transcription is disabled for this video."
        )
    except RequestBlocked: # Handling IP/Cloud Block
        logger.error("YouTube IP Blocked/Rate Limited on cloud provider IP.")
        raise HTTPException(
            status_code=503, 
            detail="ERROR: Server IP is blocked by YouTube (Cloud Provider IP). Try again later or use a different source."
        )
    except YouTubeRequestFailed as e:
        logger.error(f"YouTube API Request Failed: {e}")
        raise HTTPException(
            status_code=503, 
            detail="API connection problem. Please try again."
        )
    except Exception as e:
        logger.exception("Unknown error during YouTube transcript download")
        raise HTTPException(status_code=500, detail="Unknown server error.")
    
    # 3. Length check and summarization (1000 words logic)
    words = transkrypt.split()
    word_count = len(words)
    content_to_process = transkrypt

    if word_count > YOUTUBE_SUMMARY_THRESHOLD:
        # English translation of logger message
        logger.info(f"Transcript is too long ({word_count} words). Summarizing TF-IDF...")
        content_to_process = summarize_tfidf(transkrypt)
    else:
        # English translation of logger message
        logger.info(f"Transcript is short ({word_count} words). Processing full text.")

    # 4. Splitting into chunks (for Groq API)
    chunks = split_text(content_to_process)
    if len(chunks) == 0:
        raise HTTPException(status_code=400, detail="Transcript is too short after processing.")

    # 5. Generating flashcards
    distribution = distribute_counts_across_chunks(data.count, len(chunks))
    all_cards: List[dict] = []

    try:
        for chunk, per_count in zip(chunks, distribution):
            raw = call_model_generate(chunk, per_count)
            parsed = parse_model_chunk(raw)
            if len(parsed) > per_count:
                parsed = parsed[:per_count]
            all_cards.extend(parsed)

        return {"source_url": url, "source_id": video_id, "flashcards": all_cards}
    
    except Exception as e:
        logger.exception("Flashcard generation failed")
        raise HTTPException(status_code=500, detail=str(e))
