"""Rule-based guardrails for assistant response safety and claim control."""

import re
from typing import Dict


def _soften_guarantees(text: str) -> str:
    """Replace guarantee-like language with professional, non-absolute wording."""
    replacements = {
        r"\bi cannot guarantee an outcome until the team verifies the details\.?": (
            "An inspection may help confirm the exact cause."
        ),
        r"\bi can't guarantee an outcome until the team verifies the details\.?": (
            "An inspection may help confirm the exact cause."
        ),
        r"\bguarantee\b": "commit",
        r"\bdefinitely\b": "likely",
        r"\bfor sure\b": "as appropriate",
        r"\b100%\b": "to the best of our assessment",
        r"\bwill be fixed\b": "can be investigated and addressed",
        r"\bis fixed\b": "appears to be addressed",
    }

    sanitized = text
    for pattern, replacement in replacements.items():
        sanitized = re.sub(pattern, replacement, sanitized, flags=re.IGNORECASE)
    return sanitized


def _remove_unverified_resolution_claims(text: str, context_blob: str) -> str:
    """Drop resolution-claim sentences unless context supports the same claim."""
    risky_markers = [
        "already resolved",
        "issue has been fixed",
        "we fixed",
        "refund has been issued",
        "technician has been dispatched",
        "your case is closed",
    ]

    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    filtered = []
    context_lower = context_blob.lower()

    for sentence in sentences:
        lower_sentence = sentence.lower()
        is_risky = any(marker in lower_sentence for marker in risky_markers)
        if not is_risky:
            filtered.append(sentence)
            continue

        if any(marker in context_lower for marker in risky_markers):
            filtered.append(sentence)

    return " ".join(filtered).strip()


def apply_guardrails(reply_text: str, context_blob: str, extracted_data: Dict[str, object]) -> str:
    """Apply final safety cleanup for hallucination and no-guarantee behavior."""
    cleaned = (reply_text or "").strip()
    if not cleaned:
        return (
            "Thanks for sharing that. Based on what you described, an inspection may help "
            "confirm the exact cause and guide the next step."
        )

    combined_context = f"{context_blob}\n{extracted_data}"
    cleaned = _soften_guarantees(cleaned)
    cleaned = _remove_unverified_resolution_claims(cleaned, combined_context)

    if not cleaned:
        cleaned = (
            "Thanks for the update. An inspection may help confirm the exact cause and "
            "the most suitable next step."
        )

    return cleaned
