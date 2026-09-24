from pydantic import BaseModel, Field
from typing import Literal


# === Request/Response Schemas ===
# Each Pydantic model below defines a strict data contract for an API endpoint.
# Validation (min_length, max_length, literal choices) is enforced automatically
# by FastAPI before any business logic runs, preventing invalid payloads from
# reaching the AI provider layer.


class TextRequest(BaseModel):
    """Holds the raw text input shared by summarization, classification, and extraction endpoints.

    min_length=1 prevents empty-string requests from reaching the model.
    max_length=100000 guards against oversized payloads that would exhaust the provider's context window.
    """
    text: str = Field(
        min_length=1, max_length=100000, description="Text to be processed"
    )


class ClassificationResult(BaseModel):
    """Strict output schema returned by the /classify endpoint.

    Literal types constrain the AI response to a closed set of valid labels,
    so downstream consumers never see unexpected category or sentiment strings.
    """
    category: Literal[
        "complaint",
        "question",
        "feedback",
        "request",
        "other",
    ] = Field(description="Text category")

    sentiment: Literal[
        "positive",
        "neutral",
        "negative",
    ] = Field(description="Text sentiment")


class Entity(BaseModel):
    """Single named entity within an ExtractionResult list."""
    text: str = Field(description="Entity text")
    type: str = Field(description="Entity type (person, organization, location, etc.)")


class ExtractionResult(BaseModel):
    """Container for the list of named entities extracted by the /extract endpoint."""
    entities: list[Entity] = Field(description="List of found entities")


class RewriteRequest(BaseModel):
    """Input for the /rewrite endpoint. The tone parameter has a sensible default
    so clients can omit it without breaking the request contract.
    """
    text: str = Field(min_length=1, max_length=100000, description="Original text")
    tone: Literal[
        "professional",
        "casual",
        "academic",
    ] = Field(default="professional", description="Desired writing tone")


class TranslateRequest(BaseModel):
    """Input for the /translate endpoint. target_language accepts any ISO code string;
    validation of the language code itself is delegated to the AI provider.
    """
    text: str = Field(min_length=1, max_length=100000, description="Original text")
    target_language: str = Field(min_length=2, max_length=50, description="Target language (ISO code)")
