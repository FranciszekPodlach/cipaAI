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
from youtube_transcript_api._errors import NoTranscriptFound, TranscriptsDisabled, YouTubeRequestFailed

# ---- konfiguracja loggera ----
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("flashcards-backend")

# ---- ustawienia ----
MAX_CHARS_PER_CHUNK = 4000
MAX_CHUNKS = 10
MAX_REQUESTED_CARDS = 200

# TF-IDF ustawienia
MAX_SENTENCES = 100
MAX_WORDS = 15000  # Maksymalna ilość słów wejściowych przed streszczeniem
LANGUAGE = "english" 
YOUTUBE_SUMMARY_THRESHOLD = 1000 # Próg słów do streszczania transkryptu

# ---- init klienta Groq ----
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

app = FastAPI(title="Flashcards API")

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
    # Czyszczenie nadmiarowych białych znaków (podobnie jak w URL)
    full_text = re.sub(r'\s+', ' ', full_text).strip()
    return full_text

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
    return [s.strip() for s in sentences if len(s.split()) > 5]

def summarize_tfidf(text: str, max_sentences: int = MAX_SENTENCES) -> str:
    """Streszcza tekst używając TF-IDF do wybrania najważniejszych zdań."""
    sentences = split_into_sentences(text)
    if len(sentences) <= max_sentences:
        return " ".join(sentences)

    vectorizer = TfidfVectorizer(stop_words=LANGUAGE, ngram_range=(1, 2))
    try:
        tfidf = vectorizer.fit_transform(sentences)
        sentence_scores = np.asarray(tfidf.sum(axis=1)).ravel()
        top_indices = np.argsort(sentence_scores)[-max_sentences:]
        top_indices = sorted(top_indices)
        summary = " ".join(sentences[i] for i in top_indices)
        return summary
    except Exception as e:
        logger.error(f"Summarization failed: {e}")
        # W razie błędu vectorizera (np. same stop-words), zwróć ucięty początek
        return " ".join(sentences[:max_sentences])

# ---- YOUTUBE PRIORITY LIST ----
WSZYSTKIE_KODY_PRIORYTET = ['en', 'es', 'pt', 'ru', 'zh', 'de', 'fr', 'ja', 'ko', 'it', 'pl']

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
    system_prompt = """You are an expert educational AI designed to create high-retention Anki-style flashcards. Return strictly a JSON object with a "flashcards" array containing "question" and "answer" fields."""
    user_prompt = f"Generate exactly {count} flashcards based on the text below.\n<user_input>\n{text}\n</user_input>"
    
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
        temperature=0.6,
        response_format={"type": "json_object"}
    )
    return response.choices[0].message.content

def parse_model_chunk(raw: str) -> List[dict]:
    try:
        parsed = json.loads(raw)
        return parsed.get("flashcards", [])
    except: return []

def distribute_counts_across_chunks(total_count: int, chunks_count: int) -> List[int]:
    if chunks_count <= 0: return []
    base = total_count // chunks_count
    remainder = total_count % chunks_count
    return [base + (1 if i < remainder else 0) for i in range(chunks_count)]

# ---- routes ----

@app.post("/flashcards")
def flashcards_from_text(data: FlashcardRequest):
    if not data.text.strip(): raise HTTPException(status_code=400, detail="text is empty")
    
    # Dla czystego tekstu też dodajemy opcję streszczenia, jeśli wklejono gigantyczny tekst
    processed_text = data.text
    if len(processed_text.split()) > MAX_WORDS:
        processed_text = summarize_tfidf(processed_text)

    chunks = split_text(processed_text)
    distribution = distribute_counts_across_chunks(data.count, len(chunks))
    all_cards = []

    try:
        for chunk, per_count in zip(chunks, distribution):
            raw = call_model_generate(chunk, per_count)
            all_cards.extend(parse_model_chunk(raw)[:per_count])
        return {"flashcards": all_cards}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/flashcards/pdf")
async def flashcards_from_pdf(file: UploadFile = File(...), count: int = Form(10)):
    try:
        # 1️⃣ Wyciągnięcie całego tekstu z PDF
        text = extract_text_from_pdf(file)
        if not text: raise HTTPException(status_code=400, detail="PDF is empty")

        # 2️⃣ LOGIKA STRESZCZANIA (Taka sama jak w URL)
        # Sprawdzamy czy tekst przekracza limit słów
        processed_content = text
        if len(text.split()) > MAX_WORDS:
            logger.info("PDF too long. Summarizing...")
            processed_content = summarize_tfidf(text)

        # 3️⃣ Dzielenie na chunki i wysyłka do AI
        chunks = split_text(processed_content)
        distribution = distribute_counts_across_chunks(count, len(chunks))
        all_cards = []

        for chunk, per_count in zip(chunks, distribution):
            raw = call_model_generate(chunk, per_count)
            all_cards.extend(parse_model_chunk(raw)[:per_count])

        return {"flashcards": all_cards}
    except Exception as e:
        logger.exception("PDF processing failed")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/flashcards/url")
def flashcards_from_url(data: URLFlashcardRequest):
    try:
        text = fetch_clean_text(data.url)
        
        # LOGIKA STRESZCZANIA
        processed_content = text
        if len(text.split()) > MAX_WORDS:
            processed_content = summarize_tfidf(text)

        chunks = split_text(processed_content)
        distribution = distribute_counts_across_chunks(data.count, len(chunks))
        all_cards = []

        for chunk, per_count in zip(chunks, distribution):
            raw = call_model_generate(chunk, per_count)
            all_cards.extend(parse_model_chunk(raw)[:per_count])

        return {"url": data.url, "flashcards": all_cards}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/flashcards/youtube")
def flashcards_from_youtube(data: URLFlashcardRequest):
    video_id = wyodrebnij_video_id(data.url)
    if not video_id: raise HTTPException(status_code=400, detail="Bad URL")

    try:
        transkrypt = pobierz_transkrypt_w_pierwszym_jezyku(video_id)
        
        # LOGIKA STRESZCZANIA
        processed_content = transkrypt
        if len(transkrypt.split()) > YOUTUBE_SUMMARY_THRESHOLD:
            processed_content = summarize_tfidf(transkrypt)

        chunks = split_text(processed_content)
        distribution = distribute_counts_across_chunks(data.count, len(chunks))
        all_cards = []

        for chunk, per_count in zip(chunks, distribution):
            raw = call_model_generate(chunk, per_count)
            all_cards.extend(parse_model_chunk(raw)[:per_count])

        return {"source_url": data.url, "flashcards": all_cards}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
