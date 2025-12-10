import os
from fastapi import FastAPI
from pydantic import BaseModel
from groq import Groq

# --- APP ---
app = FastAPI()

# --- GROQ CLIENT ---
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# --- REQUEST BODY ---
class FlashcardRequest(BaseModel):
    text: str

# --- HEALTH CHECK ---
@app.get("/")
def root():
    return {"status": "ok"}

# --- API ENDPOINT ---
@app.post("/flashcards")
def generate_flashcards(data: FlashcardRequest):
    prompt = f"""
change text into flashcards
return ONLY json in format:

[
  {{ "question": "...", "answer": "..." }},
  {{ "question": "...", "answer": "..." }}
]

Tekst:
{data.text}
"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    return {
        "flashcards": response.choices[0].message.content
    }
