import os
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from pydantic import BaseModel
from groq import Groq
from pypdf import PdfReader

# ====================
# CONFIG
# ====================
MAX_CHARS_PER_CHUNK = 2500
MAX_CHUNKS = 10

# ====================
# APP
# ====================
app = FastAPI()

# ====================
# GROQ CLIENT
# ====================
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ====================
# MODELS
# ====================
class FlashcardRequest(BaseModel):
    text: str
    count: int = 10   # domyślnie 10

# ====================
# UTILS
# ====================
def split_text(text: str, max_chars: int = MAX_CHARS_PER_CHUNK):
    chunks = []
    current = ""

    for line in text.split("\n"):
        if len(current) + len(line) > max_chars:
            chunks.append(current)
            current = line
        else:
            current += "\n" + line

    if current.strip():
        chunks.append(current)

    return chunks[:MAX_CHUNKS]


def extract_text_from_pdf(file: UploadFile) -> str:
    reader = PdfReader(file.file)
    text = ""

    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted + "\n"

    return text


def generate_flashcards_from_text(text: str, count: int = 10):
    prompt = f"""
You are an assistant that converts text into study flashcards.
Generate exactly {count} flashcards from the text below (if the text is too short, generate as many sensible flashcards as possible, up to {count}).
Return ONLY a single valid JSON array (no extra commentary) in the format:

[
  {{ "question": "short question string", "answer": "concise answer string" }}
]

Detect the language of the text and produce the flashcards in the same language.

Text:
{text}
"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content


# ====================
# ROUTES
# ====================

@app.get("/")
def root():
    return {"status": "ok"}


@app.post("/flashcards")
def flashcards_from_text(data: FlashcardRequest):
    try:
        if not data.text.strip():
            raise HTTPException(status_code=400, detail="text is empty")
        if data.count <= 0 or data.count > 200:
            raise HTTPException(status_code=400, detail="count must be between 1 and 200")

        chunks = split_text(data.text)
        result = []

        remaining = data.count
        for chunk in chunks:
            per_chunk = max(1, remaining // len(chunks))
            per_chunk = min(per_chunk, remaining)

            content = generate_flashcards_from_text(chunk, per_chunk)
            result.append(content)

            remaining -= per_chunk
            if remaining <= 0:
                break

        return {"flashcards": result}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/flashcards/pdf")
async def flashcards_from_pdf(
    file: UploadFile = File(...),
    count: int = Form(10)
):
    try:
        if count <= 0 or count > 200:
            raise HTTPException(status_code=400, detail="count must be between 1 and 200")

        text = extract_text_from_pdf(file)
        if not text.strip():
            raise HTTPException(status_code=400, detail="PDF does not contain text")

        chunks = split_text(text)
        result = []

        remaining = count
        total_chunks = len(chunks)

        for i, chunk in enumerate(chunks):
            per_chunk = max(1, remaining // (total_chunks - i))
            per_chunk = min(per_chunk, remaining)

            content = generate_flashcards_from_text(chunk, per_chunk)
            result.append(content)

            remaining -= per_chunk
            if remaining <= 0:
                break

        return {"chunks": len(result), "flashcards": result}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
