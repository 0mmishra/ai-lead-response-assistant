"""Streamlit UI for chatting with the AI Lead Response Assistant backend."""

import os
from typing import Any, Dict, List

import requests
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000/respond")


def send_to_backend(history: List[Dict[str, str]], message: str) -> Dict[str, Any]:
    """Send conversation history and latest message to FastAPI."""
    response = requests.post(
        BACKEND_URL,
        json={"history": history, "message": message},
        timeout=45,
    )
    response.raise_for_status()
    return response.json()


def _normalize_history(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Normalize session messages to role/content objects before backend call."""
    normalized: List[Dict[str, str]] = []
    for item in messages:
        role = str(item.get("role", "")).strip().lower()
        content = str(item.get("content", "")).strip()
        if role in {"user", "assistant"} and content:
            normalized.append({"role": role, "content": content})
    return normalized


def main() -> None:
    """Run the Streamlit chatbot with memory-aware backend integration."""
    st.set_page_config(page_title="AI Lead Response Assistant", page_icon="🤖", layout="centered")
    st.title("AI Lead Response Assistant")
    st.write("Chat with a professional assistant that remembers your conversation.")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for chat in st.session_state.messages:
        with st.chat_message(chat["role"]):
            st.markdown(chat["content"])

    user_input = st.chat_input("Type your message...")
    if not user_input:
        return

    history = _normalize_history(st.session_state.messages)

    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Generating response..."):
            try:
                payload = send_to_backend(history=history, message=user_input)
                assistant_text = str(payload.get("reply", "No reply generated.")).strip()
                if not assistant_text:
                    assistant_text = "I can help with that. Could you share one more detail?"
                st.markdown(assistant_text)
                st.session_state.messages.append(
                    {"role": "assistant", "content": assistant_text}
                )
            except requests.HTTPError as exc:
                error_text = f"Backend HTTP error: {exc}"
                st.error(error_text)
                st.session_state.messages.append({"role": "assistant", "content": error_text})
            except requests.RequestException as exc:
                error_text = f"Backend connection error: {exc}"
                st.error(error_text)
                st.session_state.messages.append({"role": "assistant", "content": error_text})


if __name__ == "__main__":
    main()
