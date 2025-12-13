def call_model_generate(text: str, count: int) -> str:
    """Call Groq model with the REFINED Topic vs Content logic."""
    if not client:
        raise RuntimeError("Groq client not configured (GROQ_API_KEY missing).")

    # --- PLATINUM SYSTEM PROMPT ---
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

    # --- BEZPIECZNE FORMATOWANIE INPUTU ---
    # Używamy XML tags, żeby Llama wiedziała dokładnie, gdzie zaczyna się i kończy tekst użytkownika.
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
