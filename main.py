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
from urllib.parse import urlparse, parse_qs
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.formatters import TextFormatter

# Obsługa błędów YouTube (kompatybilność wersji)
try:
    from youtube_transcript_api._errors import NoTranscriptFound, TranscriptsDisabled, YouTubeRequestFailed
except ImportError:
    NoTranscriptFound = Exception
    TranscriptsDisabled = Exception
    YouTubeRequestFailed = Exception

# ---- konfiguracja loggera ----
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("flashcards-backend")

# ---- ustawienia ----
MAX_CHARS_PER_CHUNK = 4000
MAX_CHUNKS = 10

MAX_REQUESTED_CARDS = 200

# TF-IDF ustawienia
MAX_SENTENCES = 100
MAX_WORDS = 15000  
YOUTUBE_SUMMARY_THRESHOLD = 1000 

# ---- init klienta Groq ----
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

app = FastAPI(title="Ironclad Flashcards API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- modele ----
class FlashcardRequest(BaseModel):
    text: str
    count: int = 10

class URLFlashcardRequest(BaseModel):
    url: str
    count: int = 10

# ---- utils ----
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
    full_text = "\n\n".join(text_parts)
    return full_text.strip()

def fetch_clean_text(url: str) -> str:
    downloaded = trafilatura.fetch_url(url)
    if not downloaded:
        raise RuntimeError("Cannot get the URL")
    text = trafilatura.extract(
        downloaded,
        include_comments=False,
        include_tables=False,
        include_formatting=False
    )
    if not text:
        raise RuntimeError("Cannot fetch the Text from URL")
    text = re.sub(r'\[[^\]]*\]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def split_into_sentences(text: str):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if len(s.split()) > 3]

def summarize_tfidf(text: str, max_sentences: int = MAX_SENTENCES) -> str:
    sentences = split_into_sentences(text)
    if len(sentences) <= max_sentences:
        return " ".join(sentences)

    # WAŻNE: stop_words=None, żeby działało dla każdego języka (PL, FR, DE, EN itd.)
    vectorizer = TfidfVectorizer(stop_words=None, ngram_range=(1, 2))
    try:
        tfidf = vectorizer.fit_transform(sentences)
        sentence_scores = np.asarray(tfidf.sum(axis=1)).ravel()
        top_indices = np.argsort(sentence_scores)[-max_sentences:]
        top_indices = sorted(top_indices)
        return " ".join(sentences[i] for i in top_indices)
    except Exception as e:
        logger.error(f"Summarization failed: {e}")
        return " ".join(sentences[:max_sentences])

# ---- YOUTUBE PRIORITY LIST ----
# Lista priorytetów pobierania napisów
WSZYSTKIE_KODY_PRIORYTET = ['pl', 'en', 'es', 'de', 'fr', 'it', 'pt', 'ru', 'ja', 'zh', 'ko']

def wyodrebnij_video_id(url: str) -> str or None:
    query = urlparse(url)
    if query.hostname in ('www.youtube.com', 'youtube.com', 'm.youtube.com'):
        if query.path == '/watch': return parse_qs(query.query).get('v', [None])[0]
        if query.path.startswith('/embed/'): return query.path.split('/')[2]
    if query.hostname in ('youtu.be', 'www.youtu.be'): return query.path[1:]
    match = re.search(r'(?:v=|/v/|embed/|youtu\.be/)([a-zA-Z0-9_-]{11})', url)
    return match.group(1) if match else None

def pobierz_transkrypt_w_pierwszym_jezyku(video_id: str) -> str:
    ytt_api = YouTubeTranscriptApi()
    formatter = TextFormatter()
    fetched_dane = ytt_api.fetch(video_id, languages=WSZYSTKIE_KODY_PRIORYTET)
    return formatter.format_transcript(fetched_dane)

def call_model_generate(text: str, count: int) -> str:
    if not client: raise RuntimeError("Groq client not configured.")
    
    # =========================================================================
    # ŻELAZNY SYSTEM PROMPT - NIE DO ZMORDOWANIA
    # =========================================================================
    system_prompt = """You are an expert educational AI designed to create high-retention Anki-style flashcards.

### CORE LOGIC (TOPIC vs. CONTENT):
Analyze the content inside the <user_input> tags and classify it into one of two modes:

**MODE A: TOPIC EXPANSION** (Triggered when input is short, abstract, or a title, e.g., "Spanish B1", "Photosynthesis", "History of Rome")
- **Action:** You act as a Subject Matter Expert. You must generating NEW examples, facts, and vocabulary from your internal knowledge base.
- **Language Topics:** If the topic is a language (e.g., "Spanish Vocabulary"), generate word/phrase pairs.
  - GOOD: Q: "House (in Spanish)" -> A: "Casa"
  - BAD: Q: "What level is this?" -> A: "B1" (NEVER DO THIS)

**MODE B: CONTENT EXTRACTION** (Triggered when input is long, detailed text, notes, or articles)
- **Action:** You act as an Analyst. You must extract facts ONLY from the provided text.
- Do not hallucinate outside info.

### LANGUAGE RULES:
1. **Target Language:** The flashcards must be in the language of the INPUT text.
   - Exception: For Language Learning topics, use Bilingual format (Native -> Target).

### BAN LIST (NEVER DO THIS):
- Do NOT ask questions about the text itself (e.g., "What does the text say about X?").
- Do NOT ask about metadata (e.g., "Who is the author?").
- Do NOT create "orphan" questions like "He went there." (Who is he?).

### JSON OUTPUT FORMAT:
Return strictly a JSON object with a "flashcards" array.
{
  "flashcards": [
    {
      "question": "Clear, specific question",
      "answer": "Concise answer"
    }
  ]
}
"""

Detect the language of the text and produce the flashcards in the same language.
    # --- BEZPIECZNE FORMATOWANIE INPUTU ---
    # Używamy XML tags, żeby Llama wiedziała dokładnie, gdzie zaczyna się i kończy tekst użytkownika.
    user_prompt = f"""Generate exactly {count} flashcards based on the text below.
Apply the TOPIC vs CONTENT logic strictly.

Text:
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
        temperature=0.4, # Niski, żeby nie gwiazdorzył
        response_format={"type": "json_object"}
    )
    return response.choices[0].message.content

def parse_model_chunk(raw: str) -> List[dict]:
    try:
        parsed = json.loads(raw)
        return parsed.get("flashcards", [])
    except json.JSONDecodeError:
        logger.error(f"JSON Decode Error. Raw: {raw[:50]}")
        return []
    except Exception as e:
        logger.error(f"General Parse Error: {e}")
        return []

def distribute_counts_across_chunks(total_count: int, chunks_count: int) -> List[int]:
    if chunks_count <= 0: return []
    base = total_count // chunks_count
    remainder = total_count % chunks_count
    return [base + (1 if i < remainder else 0) for i in range(chunks_count)]

# ---- routes ----

@app.post("/flashcards")
def flashcards_from_text(data: FlashcardRequest):
    if not data.text.strip(): raise HTTPException(status_code=400, detail="Text is empty")
    
    processed_text = data.text
    if len(processed_text.split()) > MAX_WORDS:
        processed_text = summarize_tfidf(processed_text)

    chunks = split_text(processed_text)
    distribution = distribute_counts_across_chunks(data.count, len(chunks))
    all_cards = []

    try:
        for chunk, per_count in zip(chunks, distribution):
            if per_count == 0: continue
            raw = call_model_generate(chunk, per_count)
            all_cards.extend(parse_model_chunk(raw)[:per_count])
        return {"flashcards": all_cards}
    except Exception as e:
        logger.exception("Text processing error")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/flashcards/pdf")
async def flashcards_from_pdf(file: UploadFile = File(...), count: int = Form(10)):
    try:
        text = extract_text_from_pdf(file)
        if not text: raise HTTPException(status_code=400, detail="PDF is empty")

        processed_content = text
        if len(text.split()) > MAX_WORDS:
            logger.info("PDF too long. Summarizing...")
            processed_content = summarize_tfidf(text)

        chunks = split_text(processed_content)
        distribution = distribute_counts_across_chunks(count, len(chunks))
        all_cards = []

        for chunk, per_count in zip(chunks, distribution):
            if per_count == 0: continue
            raw = call_model_generate(chunk, per_count)
            all_cards.extend(parse_model_chunk(raw)[:per_count])

        return {"flashcards": all_cards}
    except Exception as e:
        logger.exception("PDF processing error")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/flashcards/url")
def flashcards_from_url(data: URLFlashcardRequest):
    try:
        text = fetch_clean_text(data.url)
        processed_content = text
        if len(text.split()) > MAX_WORDS:
            processed_content = summarize_tfidf(text)

        chunks = split_text(processed_content)
        distribution = distribute_counts_across_chunks(data.count, len(chunks))
        all_cards = []

        for chunk, per_count in zip(chunks, distribution):
            if per_count == 0: continue
            raw = call_model_generate(chunk, per_count)
            all_cards.extend(parse_model_chunk(raw)[:per_count])

        return {"url": data.url, "flashcards": all_cards}
    except Exception as e:
        logger.exception("URL processing error")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/flashcards/youtube")
def flashcards_from_youtube(data: URLFlashcardRequest):
    video_id = wyodrebnij_video_id(data.url)
    if not video_id: raise HTTPException(status_code=400, detail="Invalid YouTube URL")

    try:
        transkrypt = pobierz_transkrypt_w_pierwszym_jezyku(video_id)
        processed_content = transkrypt
        if len(transkrypt.split()) > YOUTUBE_SUMMARY_THRESHOLD:
            processed_content = summarize_tfidf(transkrypt)

        chunks = split_text(processed_content)
        distribution = distribute_counts_across_chunks(data.count, len(chunks))
        all_cards = []

        for chunk, per_count in zip(chunks, distribution):
            if per_count == 0: continue
            raw = call_model_generate(chunk, per_count)
            all_cards.extend(parse_model_chunk(raw)[:per_count])

        return {"source_url": data.url, "flashcards": all_cards}
    except Exception as e:
        logger.exception("YouTube processing error")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


