# na górze pliku dodaj:
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from pydantic import BaseModel
# ... reszta importów

# ------------------------------------------------
class FlashcardRequest(BaseModel):
    text: str
    count: int = 10   # domyślnie 10, można przesłać inną wartość

# ------------------------------------------------
def generate_flashcards_from_text(text: str, count: int = 10):
    # jasna instrukcja — dokładnie count fiszek, w tym samym języku co tekst
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
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

# ------------------------------------------------
@app.post("/flashcards")
def flashcards_from_text(data: FlashcardRequest):
    try:
        # validate input early
        if not data.text.strip():
            raise HTTPException(status_code=400, detail="text is empty")
        if data.count <= 0 or data.count > 200:
            raise HTTPException(status_code=400, detail="count must be between 1 and 200")

        chunks = split_text(data.text)
        result = []
        # pass count to each chunk but limit total requested flashcards:
        remaining = data.count
        for chunk in chunks:
            # compute per-chunk desired count: distribute remaining across chunks
            per_chunk = max(1, remaining // len(chunks))
            # ensure not to request more than remaining
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

# ------------------------------------------------
@app.post("/flashcards/pdf")
async def flashcards_from_pdf(file: UploadFile = File(...), count: int = Form(10)):
    try:
        if count <= 0 or count > 200:
            raise HTTPException(status_code=400, detail="count must be between 1 and 200")

        text = extract_text_from_pdf(file)
        if not text.strip():
            raise HTTPException(status_code=400, detail="PDF does not contain text")

        chunks = split_text(text)
        # distribute count across chunks (simple distribution)
        total_chunks = len(chunks)
        result = []
        remaining = count
        for i, chunk in enumerate(chunks):
            # distribute remaining proportionally or evenly
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
