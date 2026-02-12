"""OpenRouter LLM client and prompt functions for extraction and contextual reply generation."""

import json
import os
from typing import Any, Dict, List

import requests
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
TIMEOUT_SECONDS = 45


class LLMServiceError(RuntimeError):
    """Raised when an OpenRouter call fails or returns invalid content."""


def _call_openrouter(messages: List[Dict[str, str]]) -> str:
    """Send a chat completion request to OpenRouter and return message content."""
    if not OPENROUTER_API_KEY:
        raise LLMServiceError("OPENROUTER_API_KEY is not configured.")

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": OPENROUTER_MODEL,
        "temperature": 0.2,
        "messages": messages,
    }

    try:
        response = requests.post(
            OPENROUTER_URL,
            headers=headers,
            json=payload,
            timeout=TIMEOUT_SECONDS,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        raise LLMServiceError(f"OpenRouter request failed: {exc}") from exc

    data = response.json()
    choices = data.get("choices") or []
    if not choices:
        raise LLMServiceError("OpenRouter returned no choices.")

    message = choices[0].get("message", {})
    content = message.get("content", "")
    if not isinstance(content, str) or not content.strip():
        raise LLMServiceError("OpenRouter returned empty content.")

    return content.strip()


def _parse_json_object(text: str) -> Dict[str, Any]:
    """Parse strict JSON; if wrapped in markdown, extract and parse the JSON object."""
    raw = text.strip()

    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = raw[start : end + 1]
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError as exc:
            raise LLMServiceError(f"Failed to parse extraction JSON: {exc}") from exc

    raise LLMServiceError("LLM extraction response is not valid JSON.")


def _format_history(history: List[Dict[str, str]]) -> str:
    """Convert chat history into a compact transcript for prompts."""
    lines: List[str] = []
    for item in history:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role", "")).strip().lower()
        content = str(item.get("content", "")).strip()
        if role not in {"user", "assistant"} or not content:
            continue
        speaker = "User" if role == "user" else "Assistant"
        lines.append(f"{speaker}: {content}")

    return "\n".join(lines) if lines else "No prior conversation."


def extract_structured_data(input_text: str) -> Dict[str, Any]:
    """Extract internal structured fields as strict JSON from provided context."""
    messages = [
        {
            "role": "system",
            "content": (
                "You extract structured fields from customer messages. "
                "Return ONLY strict JSON and no extra text."
            ),
        },
        {
            "role": "user",
            "content": (
                "Return ONLY JSON with keys: issue_type, location, trigger, urgency, "
                "missing_information. If info is missing, set value to 'Not Available'. "
                "If conflicting, mention conflict under missing_information as separate item. "
                "missing_information must be an array of strings.\n\n"
                f"Input text: \"{input_text}\""
            ),
        },
    ]

    content = _call_openrouter(messages=messages)
    return _parse_json_object(content)


def generate_reply(
    history: List[Dict[str, str]], latest_message: str, extracted_data: Dict[str, Any]
) -> str:
    """Generate a professional context-aware reply using history and latest message."""
    transcript = _format_history(history)
    structured_json = json.dumps(extracted_data, ensure_ascii=True)

    messages = [
        {
            "role": "system",
            "content": (
                "You are a professional support assistant. Maintain topic continuity across turns, "
                "avoid repeating previously asked questions, and respond naturally in client-friendly "
                "language. Do not invent facts, completed actions, or guarantees. Keep follow-up "
                "questions specific and minimal."
            ),
        },
        {
            "role": "user",
            "content": (
                "Use the conversation and latest message to craft the next assistant reply.\n"
                "Requirements:\n"
                "1) Acknowledge the latest user message naturally.\n"
                "2) Continue from prior context; do not reset the conversation.\n"
                "3) Do not repeat questions already asked unless absolutely necessary.\n"
                "4) Ask only relevant follow-up questions still needed.\n"
                "5) Provide safe next steps without guarantees.\n"
                "6) Avoid technical jargon and keep it concise.\n"
                "7) Return only assistant reply text.\n\n"
                f"Conversation history:\n{transcript}\n\n"
                f"Latest user message: {latest_message}\n\n"
                f"Internal structured extraction (not for display): {structured_json}"
            ),
        },
    ]

    return _call_openrouter(messages=messages)
