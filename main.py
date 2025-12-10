import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from groq import Groq
from pypdf import PdfReader

# ====================
# CONFIG
# ====================
MAX_CHARS_PER_CHUNK = 2500
MAX_CHUNKS = 10  # zabezpieczenie (max ~25k znaków)

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

def generate_flashcards_from_text(text: str):
    prompt = f"""
make flashcards from this text in texts language.

return ONLY json:
[
  {{ "question": "...", "answer": "..." }}
]

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
        chunks = split_text(data.text)
        result = []

        for chunk in chunks:
            result.append(generate_flashcards_from_text(chunk))

        return {"flashcards": result}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/flashcards/pdf")
async def flashcards_from_pdf(file: UploadFile = File(...)):
    try:
        text = extract_text_from_pdf(file)

        if not text.strip():
            raise HTTPException(status_code=400, detail="PDF nie zawiera tekstu")

        chunks = split_text(text)
        result = []

        for chunk in chunks:
            result.append(generate_flashcards_from_text(chunk))

        return {
            "chunks": len(chunks),
            "flashcards": result
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

