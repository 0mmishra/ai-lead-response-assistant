"""FastAPI entrypoint for the AI Lead Response Assistant."""

from typing import Dict, List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

try:
    # Works when launched from project root
    from backend.guardrails import apply_guardrails
    from backend.llm_service import LLMServiceError, extract_structured_data, generate_reply
    from backend.schemas import ChatRequest, ChatResponse, StructuredExtraction
except ModuleNotFoundError:
    # Works when launched from backend directory
    from guardrails import apply_guardrails
    from llm_service import LLMServiceError, extract_structured_data, generate_reply
    from schemas import ChatRequest, ChatResponse, StructuredExtraction


# -----------------------------------------------------------------------------
# FastAPI App Setup
# -----------------------------------------------------------------------------
app = FastAPI(title="AI Lead Response Assistant", version="2.1.0")

# Enable CORS for frontend communication (Streamlit / Web clients)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Safe for assignment demo. Restrict in production.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------------------------------------------------------------
# Health & Root Endpoints
# -----------------------------------------------------------------------------
@app.get("/")
def root() -> Dict[str, str]:
    """Root endpoint used by deployment platforms for health verification."""
    return {"message": "AI Lead Response Assistant is running"}


@app.get("/health")
def health_check() -> Dict[str, str]:
    """Basic health endpoint."""
    return {"status": "ok"}


# -----------------------------------------------------------------------------
# Utility: Sanitize Conversation History
# -----------------------------------------------------------------------------
def _sanitize_history(history: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Ensure history contains valid role/content message objects."""
    sanitized: List[Dict[str, str]] = []

    for item in history:
        if not isinstance(item, dict):
            continue

        role = str(item.get("role", "")).strip().lower()
        content = str(item.get("content", "")).strip()

        if role not in {"user", "assistant"}:
            continue

        if not content:
            continue

        sanitized.append({"role": role, "content": content})

    return sanitized


# -----------------------------------------------------------------------------
# Main Chat Endpoint
# -----------------------------------------------------------------------------
@app.post("/respond", response_model=ChatResponse)
def respond(request: ChatRequest) -> ChatResponse:
    """
    Generate a context-aware assistant reply using conversation memory.
    Includes:
    - History sanitization
    - Memory size control
    - Structured extraction (internal only)
    - Guardrails enforcement
    """

    latest_message = request.message.strip()
    if not latest_message:
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    try:
        # ----------------------------
        # Sanitize and Limit Memory
        # ----------------------------
        history = _sanitize_history(request.history or [])

        MAX_HISTORY = 10  # Prevent token overflow
        history = history[-MAX_HISTORY:]

        # Build context messages
        context_messages = history + [{"role": "user", "content": latest_message}]

        # Flatten conversation into context blob
        context_blob = "\n".join(
            f"{msg['role']}: {msg['content']}" for msg in context_messages
        )

        # ----------------------------
        # Structured Extraction
        # ----------------------------
        extracted_raw = extract_structured_data(context_blob)
        structured = StructuredExtraction.model_validate(extracted_raw)

        # ----------------------------
        # Generate Reply
        # ----------------------------
        generated_reply = generate_reply(
            history=history,
            latest_message=latest_message,
            extracted_data=structured.model_dump(),
        )

        # ----------------------------
        # Apply Guardrails
        # ----------------------------
        cleaned_reply = apply_guardrails(
            reply_text=generated_reply,
            context_blob=context_blob,
            extracted_data=structured.model_dump(),
        )

        return ChatResponse(reply=cleaned_reply)

    except LLMServiceError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected server error: {exc}",
        ) from exc
