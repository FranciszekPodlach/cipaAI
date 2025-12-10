import os
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from groq import Groq
from pypdf import PdfReader

# --------------------
# APP
# --------------------
app = FastAPI()

# --------------------
# GROQ CLIENT
# --------------------
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# --------------------
# MODELS
# --------------------
class FlashcardRequest(BaseModel):
    text: str

# --------------------
# HEALTH CHECK
# --------------------
@app.get("/")
def root():
    return {"status": "ok"}

# --------------------
# TEXT -> FLASHCARDS
# --------------------
@app.post("/flashcards")
def generate_flashcards(data: FlashcardRequest):
    prompt = f"""
Przerób poniższy tekst na fiszki do nauki.

Zwróć TYLKO czysty JSON w formacie:
[
  {{ "question": "...", "answer": "..." }},
  {{ "question": "...", "answer": "..." }}
]

Tekst:
{data.text}
"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}]
    )

    return {
        "flashcards": response.choices[0].message.content
    }

# --------------------
# PDF -> TEXT
# --------------------
def extract_text_from_pdf(file: UploadFile) -> str:
    reader = PdfReader(file.file)
    text = ""

    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted + "\n"

    return text

# --------------------
# PDF -> FLASHCARDS
# --------------------
@app.post("/flashcards/pdf")
async def generate_flashcards_from_pdf(file: UploadFile = File(...)):
    text = extract_text_from_pdf(file)

    if not text.strip():
        return {"error": "Nie udało się wyciągnąć tekstu z PDF"}

    prompt = f"""
Przerób poniższy tekst na fiszki do nauki.

Zwróć TYLKO czysty JSON w formacie:
[
  {{ "question": "...", "answer": "..." }},
  {{ "question": "...", "answer": "..." }}
]

Tekst:
{text}
"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}]
    )

    return {
        "flashcards": response.choices[0].message.content
    }
