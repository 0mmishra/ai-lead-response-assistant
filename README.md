# AI Lead Response Assistant

A memory-based conversational assistant using:
- FastAPI backend
- Streamlit chat frontend
- OpenRouter LLM API
- Internal structured extraction + safety guardrails

## Architecture

- UI remains chatbot-only (no structured JSON shown)
- Structured extraction runs internally to improve response quality and safety
- Conversation history is preserved in Streamlit session state and sent to backend each turn

## Project Structure

```text
backend/
  main.py
  schemas.py
  llm_service.py
  guardrails.py
frontend/
  app.py
.env.example
requirements.txt
README.md
```

## Memory-Based Workflow

```text
[User Message in Streamlit]
        |
        v
[Streamlit Session Memory: st.session_state.messages]
        |
        v
[POST /respond with {history, message}]
        |
        v
[FastAPI]
  1) sanitize history + latest message
  2) internal structured extraction (not exposed)
  3) context-aware reply generation with continuity rules
  4) guardrails cleanup (no hallucinated actions / no guarantees)
        |
        v
[Return {reply}]
        |
        v
[Streamlit renders assistant reply and appends to memory]
```

## Safety Design

- No invented completed actions
- No guaranteed outcomes
- Professional, client-friendly wording
- Continuity instructions reduce repetitive fallback questions

## Setup

1. Create and activate a virtual environment.

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
```

2. Install dependencies.

```bash
pip install -r requirements.txt
```

3. Configure environment variables.

```bash
copy .env.example .env
```

Set `OPENROUTER_API_KEY` in `.env`.

## Run Backend

From project root:

```bash
uvicorn backend.main:app --reload
```

From `backend/` folder:

```bash
uvicorn main:app --reload
```

## Run Frontend

In a second terminal:

```bash
streamlit run frontend/app.py
```

## API Contract

### Request

```json
{
  "history": [
    {"role": "user", "content": "Damp patches appear on my wall during rainy season."},
    {"role": "assistant", "content": "Thanks for sharing that. Could you tell me how long this has been happening?"}
  ],
  "message": "They are from past 1 year."
}
```

### Response

```json
{
  "reply": "Thanks for confirming. Since this has continued for about a year..."
}
```

## Testing

1. Start backend and frontend.
2. Send:
   - `Damp patches appear on my wall during rainy season.`
3. Then send:
   - `They are from past 1 year.`
4. Verify assistant continues the same topic and does not ask what issue user means.
