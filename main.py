import os
import json
import logging
from typing import List

from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq
from pypdf import PdfReader
import trafilatura
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# ---- konfiguracja loggera ----
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("flashcards-backend")

# ---- ustawienia ----
MAX_CHARS_PER_CHUNK = 4000
MAX_CHUNKS = 10
MAX_REQUESTED_CARDS = 200
TFIDF_MAX_SENTENCES = 100
TFIDF_MAX_WORDS = 15000
TFIDF_LANGUAGE = "english"

# ---- inicjalizacja aplikacji ----
app = FastAPI(title="Flashcards API")

# Zezwól na CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- init klienta Groq ----
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    logger.warning("GROQ_API_KEY not set in environment — requests to AI will fail.")
client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

# ---- modele ----
class FlashcardRequest(BaseModel):
    text: str
    count: int = 10

class URLRequest(BaseModel):
    url: str
    max_sentences: int = TFIDF_MAX_SENTENCES

# ---- utilsy ----
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

def call_model_generate(text: str, count: int) -> str:
    """Call Groq model with the REFINED Topic vs Content logic."""
    if not client:
        raise RuntimeError("Groq client not configured (GROQ_API_KEY missing).")

    # --- PLATINUM SYSTEM PROMPT ---
    system_prompt = """You are an expert educational AI designed to create high-retention Anki-style flashcards.

### CORE LOGIC (TOPIC vs. CONTENT):
Analyze the content inside the <user_input> tags and classify it into one of two modes:

**MODE A: TOPIC EXPANSION** (Triggered when input is short, abstract, or a title, e.g., "Spanish B1", "Photosynthesis", "History of Rome")
- **Action:** You act as a Subject Matter Expert. You must generate NEW examples, facts, and vocabulary from your internal knowledge base.
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

    # --- BEZPIECZNE FORMATOWANIE INPUTU ---
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
        for value in parsed.values():
            if isinstance(value, list) and value and isinstance(value[0], dict) and "question" in value[0]:
                 return [p for p in value if isinstance(p, dict)]
    except json.JSONDecodeError:
        logger.error("JSON Decode Error: %s", raw[:200])
        pass
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

# ---- TF-IDF utilities ----
def fetch_clean_text(url: str) -> str:
    downloaded = trafilatura.fetch_url(url)
    if not downloaded:
        raise HTTPException(status_code=400, detail="Nie udało się pobrać strony")

    text = trafilatura.extract(
        downloaded,
        include_comments=False,
        include_tables=False,
        include_formatting=False
    )
    if not text:
        raise HTTPException(status_code=400, detail="Nie udało się wyodrębnić treści")

    text = re.sub(r'\[[^\]]*\]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    words = text.split()
    if len(words) > TFIDF_MAX_WORDS:
        text = " ".join(words[:TFIDF_MAX_WORDS])
    return text

def split_into_sentences(text: str):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if len(s.split()) > 5]

def summarize_tfidf(text: str, max_sentences: int) -> str:
    sentences = split_into_sentences(text)
    if len(sentences) <= max_sentences:
        return " ".join(sentences)

    vectorizer = TfidfVectorizer(stop_words=TFIDF_LANGUAGE, ngram_range=(1,2))
    tfidf = vectorizer.fit_transform(sentences)
    sentence_scores = np.asarray(tfidf.sum(axis=1)).ravel()
    top_indices = np.argsort(sentence_scores)[-max_sentences:]
    top_indices = sorted(top_indices)
    summary = " ".join(sentences[i] for i in top_indices)
    return summary

# ---- routes ----
@app.get("/")
def root():
    return {"status": "ok"}

@app.post("/flashcards")
def flashcards_from_text(data: FlashcardRequest):
    if not data.text or not data.text.strip():
        raise HTTPException(status_code=400, detail="text is empty")
    if data.count <= 0: 
        raise HTTPException(status_code=400, detail="count must be positive")

    chunks = split_text(data.text)
    if len(chunks) == 0:
        raise HTTPException(status_code=400, detail="text is empty after processing")

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
        raise HTTPException(status_code=400, detail="count must be positive")

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

# ---- NOWY ENDPOINT TF-IDF ----
@app.post("/summarize/url")
def summarize_url(data: URLRequest):
    if not data.url.strip():
        raise HTTPException(status_code=400, detail="URL is empty")

    text = fetch_clean_text(data.url)
    summary = summarize_tfidf(text, data.max_sentences)
    return {
        "url": data.url,
        "summary": summary,
        "words": len(summary.split())
    }
