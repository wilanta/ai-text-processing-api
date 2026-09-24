from pydantic import BaseModel, Field
from typing import Literal


# === Request/Response Schemas ===
# These schemas define the data structures for API requests and responses
# using Pydantic for automatic type validation.


class TextRequest(BaseModel):
    """Text request for various processing operations (summarize, classify, extract)."""
    text: str = Field(
        min_length=1, max_length=100000, description="Text to be processed"
    )


class ClassificationResult(BaseModel):
    """Text classification result: category and sentiment."""
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
    """A single named entity extracted from text."""
    text: str = Field(description="Entity text")
    type: str = Field(description="Entity type (person, organization, location, etc.)")


class ExtractionResult(BaseModel):
    """Entity extraction result: list of entities found in text."""
    entities: list[Entity] = Field(description="List of found entities")


class RewriteRequest(BaseModel):
    """Request to rewrite text in a specific tone."""
    text: str = Field(min_length=1, max_length=100000, description="Original text")
    tone: Literal[
        "professional",
        "casual",
        "academic",
    ] = Field(default="professional", description="Desired writing tone")


class TranslateRequest(BaseModel):
    """Request to translate text to a target language."""
    text: str = Field(min_length=1, max_length=100000, description="Original text")
    target_language: str = Field(min_length=2, max_length=50, description="Target language (ISO code)")
