"""FastAPI entrypoint for the AI Lead Response Assistant."""

from typing import Dict, List

from fastapi import FastAPI, HTTPException

try:
    # Works when launched from project root: uvicorn backend.main:app --reload
    from backend.guardrails import apply_guardrails
    from backend.llm_service import LLMServiceError, extract_structured_data, generate_reply
    from backend.schemas import ChatRequest, ChatResponse, StructuredExtraction
except ModuleNotFoundError:
    # Works when launched from backend dir: uvicorn main:app --reload
    from guardrails import apply_guardrails
    from llm_service import LLMServiceError, extract_structured_data, generate_reply
    from schemas import ChatRequest, ChatResponse, StructuredExtraction

app = FastAPI(title="AI Lead Response Assistant", version="2.0.0")


@app.get("/health")
def health_check() -> Dict[str, str]:
    """Basic health endpoint for local service checks."""
    return {"status": "ok"}


def _sanitize_history(history: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Filter history into valid role/content message objects."""
    sanitized: List[Dict[str, str]] = []
    for item in history:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role", "")).strip().lower()
        content = str(item.get("content", "")).strip()
        if role not in {"user", "assistant"} or not content:
            continue
        sanitized.append({"role": role, "content": content})
    return sanitized


@app.post("/respond", response_model=ChatResponse)
def respond(request: ChatRequest) -> ChatResponse:
    """Generate a context-aware assistant reply using prior conversation memory."""
    latest_message = request.message.strip()
    if not latest_message:
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    try:
        history = _sanitize_history(request.history or [])
        context_messages = history + [{"role": "user", "content": latest_message}]
        context_blob = "\n".join(
            f"{msg['role']}: {msg['content']}" for msg in context_messages
        )

        extracted_raw = extract_structured_data(context_blob)
        structured = StructuredExtraction.model_validate(extracted_raw)
        generated_reply = generate_reply(
            history=history,
            latest_message=latest_message,
            extracted_data=structured.model_dump(),
        )
        cleaned_reply = apply_guardrails(
            reply_text=generated_reply,
            context_blob=context_blob,
            extracted_data=structured.model_dump(),
        )

        return ChatResponse(reply=cleaned_reply)
    except LLMServiceError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Unexpected server error: {exc}") from exc
