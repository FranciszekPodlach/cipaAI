import os
import json
import logging
from typing import List

from fastapi import FastAPI, UploadFile, File, HTTPException, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq
from pypdf import PdfReader

# ---- konfiguracja loggera ----
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("flashcards-backend")

# ---- ustawienia ----
MAX_CHARS_PER_CHUNK = 2500
MAX_CHUNKS = 10
MAX_REQUESTED_CARDS = 200

# ---- inicjalizacja aplikacji ----
app = FastAPI(title="Flashcards API")

# Zezwól na CORS (zmień "*" na docelowe origine w produkcji)
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


# ---- utilsy ----
def split_text(text: str, max_chars: int = MAX_CHARS_PER_CHUNK) -> List[str]:
    """Split text into chunks not exceeding max_chars roughly by lines; limit to MAX_CHUNKS."""
    chunks: List[str] = []
    current = ""

    for line in text.split("\n"):
        # keep paragraphs intact as much as possible
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
    """Extract text from uploaded PDF (pypdf)."""
    reader = PdfReader(file.file)
    text_parts: List[str] = []
    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            text_parts.append(extracted)
    return "\n\n".join(text_parts)


def call_model_generate(text: str, count: int) -> str:
    """Call Groq model and return raw assistant content (string). Raises exceptions from client."""
    if not client:
        raise RuntimeError("Groq client not configured (GROQ_API_KEY missing).")

    prompt = f"""
You are an assistant that converts text into study flashcards.
Generate exactly {count} flashcards from the text below (if the text is too short, generate as many sensible flashcards as possible, up to {count}).
Return ONLY a single valid JSON array (no extra commentary) in the format:

[
  {{ "question": "short question string", "answer": "concise answer string" }},
  ...
]

Detect the language of the text and produce the flashcards in the same language.

Text:
{text}
"""
    # Use chat completions API
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
    )
    # response.choices[0].message.content expected
    return response.choices[0].message.content


def parse_model_chunk(raw: str) -> List[dict]:
    """Try to parse a single chunk returned by model into list of flashcard dicts."""
    # Model should return a JSON array string — be tolerant and try to recover
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            # ensure each element is a dict
            return [p for p in parsed if isinstance(p, dict)]
    except Exception:
        # attempt to extract the first JSON array inside the text
        try:
            start = raw.index('[')
            end = raw.rindex(']') + 1
            snippet = raw[start:end]
            parsed = json.loads(snippet)
            if isinstance(parsed, list):
                return [p for p in parsed if isinstance(p, dict)]
        except Exception:
            # give up
            logger.debug("Failed to parse model chunk as JSON array. Raw: %s", raw[:200])
            raise ValueError("Model returned non-JSON response or unparsable output.")
    raise ValueError("Model returned unexpected structure.")


def distribute_counts_across_chunks(total_count: int, chunks_count: int) -> List[int]:
    """Distribute total_count across chunks roughly evenly, at least 1 per chunk if possible."""
    if chunks_count <= 0:
        return []
    base = total_count // chunks_count
    remainder = total_count % chunks_count
    distribution = []
    for i in range(chunks_count):
        per = base + (1 if i < remainder else 0)
        if per <= 0:
            per = 1
        distribution.append(per)
    # if total_count < chunks_count, we may have produced numbers > total_count; normalize
    # ensure sum <= total_count by trimming some from the end if necessary
    while sum(distribution) > total_count:
        for i in range(len(distribution)-1, -1, -1):
            if distribution[i] > 0 and sum(distribution) > total_count:
                distribution[i] -= 1
    # remove zeros if any
    distribution = [d for d in distribution if d > 0]
    return distribution


# ---- routes ----
@app.get("/")
def root():
    return {"status": "ok"}


@app.post("/flashcards")
def flashcards_from_text(data: FlashcardRequest):
    if not data.text or not data.text.strip():
        raise HTTPException(status_code=400, detail="text is empty")
    if data.count <= 0 or data.count > MAX_REQUESTED_CARDS:
        raise HTTPException(status_code=400, detail=f"count must be between 1 and {MAX_REQUESTED_CARDS}")

    # split and distribute
    chunks = split_text(data.text)
    if len(chunks) == 0:
        raise HTTPException(status_code=400, detail="text is empty after processing")

    distribution = distribute_counts_across_chunks(data.count, len(chunks))

    all_cards: List[dict] = []
    try:
        for chunk, per_count in zip(chunks, distribution):
            raw = call_model_generate(chunk, per_count)
            parsed = parse_model_chunk(raw)
            # if parsed has more than per_count, take per_count; if less - accept what we got
            if len(parsed) > per_count:
                parsed = parsed[:per_count]
            all_cards.extend(parsed)
        return {"flashcards": all_cards}
    except Exception as e:
        logger.exception("Error generating flashcards from text")
        # if it's a Groq token error, surface it:
        msg = str(e)
        raise HTTPException(status_code=500, detail=msg)


@app.post("/flashcards/pdf")
async def flashcards_from_pdf(file: UploadFile = File(...), count: int = Form(10)):
    if count <= 0 or count > MAX_REQUESTED_CARDS:
        raise HTTPException(status_code=400, detail=f"count must be between 1 and {MAX_REQUESTED_CARDS}")

    text = extract_text_from_pdf(file)
    if not text or not text.strip():
        raise HTTPException(status_code=400, detail="PDF does not contain text")

    chunks = split_text(text)
    if len(chunks) == 0:
        raise HTTPException(status_code=400, detail="PDF text too short after processing")

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
