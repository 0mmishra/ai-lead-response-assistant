"""Pydantic schemas for chat request/response and internal extraction."""

from typing import Dict, List, Union

from pydantic import BaseModel, Field, field_validator


class ChatRequest(BaseModel):
    """Input payload for the /respond endpoint with conversation memory."""

    history: List[Dict[str, str]] = Field(
        default_factory=list,
        description="Prior messages as {'role': 'user|assistant', 'content': '...'} objects.",
    )
    message: str = Field(..., min_length=1, description="Latest user message.")

    @field_validator("history", mode="before")
    @classmethod
    def normalize_history(cls, value: Union[List[Dict[str, str]], None]) -> List[Dict[str, str]]:
        """Normalize missing history to an empty list for robust request handling."""
        if value is None:
            return []
        if not isinstance(value, list):
            return []
        return value


class StructuredExtraction(BaseModel):
    """Normalized extraction result used internally by the backend."""

    issue_type: str = Field(..., description="Type/category of issue.")
    location: str = Field(..., description="Location referenced by the customer.")
    trigger: str = Field(..., description="What appears to have caused the issue.")
    urgency: str = Field(..., description="Urgency level inferred from the inquiry.")
    missing_information: List[str] = Field(
        default_factory=lambda: ["Not Available"],
        description="List of details that are not present or are conflicting.",
    )

    @field_validator("issue_type", "location", "trigger", "urgency", mode="before")
    @classmethod
    def normalize_text_fields(cls, value: Union[str, None]) -> str:
        """Coerce empty text fields into a consistent placeholder."""
        if value is None:
            return "Not Available"
        text = str(value).strip()
        return text if text else "Not Available"

    @field_validator("missing_information", mode="before")
    @classmethod
    def normalize_missing_information(cls, value: Union[str, List[str], None]) -> List[str]:
        """Normalize missing info into a non-empty list of strings."""
        if value is None:
            return ["Not Available"]
        if isinstance(value, str):
            cleaned = value.strip()
            return [cleaned] if cleaned else ["Not Available"]
        if isinstance(value, list):
            cleaned_items = [str(item).strip() for item in value if str(item).strip()]
            return cleaned_items if cleaned_items else ["Not Available"]
        return ["Not Available"]


class ChatResponse(BaseModel):
    """Chatbot-style API response."""

    reply: str
