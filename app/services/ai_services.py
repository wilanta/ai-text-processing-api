import os
import json

from dotenv import load_dotenv

from app.schemas import ClassificationResult, ExtractionResult

from app.providers.ollama import OllamaProvider

provider = OllamaProvider()

# Load environment variables from .env file
load_dotenv()

# Ollama connection configuration (local AI model)
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:1.7b")

AI_TIMEOUT = float(os.getenv("AI_TIMEOUT", "12O"))


async def summarize_text(text: str) -> str:
    """Summarize text into a concise version while preserving important information.

    Args:
        text: Source text to be summarized.

    Returns:
        Text summary as a string.
    """
    prompt = f"""
    You are a text summarization assistant.

Summarize the following text clearly and concisely.
Preserve the important information.
Do not add information that is not present in the original text.

Text:
{text}

Summary:
    """
    return await provider.generate(prompt)


def classify_text(text: str) -> ClassificationResult:
    """Classify text into a category and sentiment.

    Args:
        text: Text to be classified.

    Returns:
        ClassificationResult containing category and sentiment.
    """
    prompt = f"""
Classify the following text.

Determine:

1. Category:
- complaint
- question
- feedback
- request
- other

2. Sentiment:
- positive
- neutral
- negative

Return only data matching the requested schema.

Text:
{text}
    """
    result = provider.generate_structured(
        prompt,
        ClassificationResult.model_json_schema(),
    )
    parsed = json.loads(result["response"])
    return ClassificationResult.model_validate(parsed)


async def extract_entities(text: str) -> ExtractionResult:
    """Extract named entities (person names, organizations, locations, etc.) from text.

    Args:
        text: Source text for entity extraction.

    Returns:
        ExtractionResult containing the list of found entities.
    """
    prompt = f"""
Extract important named entities from the following text.

Entities may include:
- person
- organization
- location
- product
- date
- technology

Do not invent entities.

Text:
{text}
"""
    result = provider.generate_structured(
        prompt,
        ExtractionResult.model_json_schema(),
    )
    parsed = json.loads(result["response"])
    return await ExtractionResult.model_validate(parsed)


async def rewrite_text(text: str, tone: str) -> str:
    """Rewrite text in the requested tone without changing the original meaning.

    Args:
        text: Original text to be rewritten.
        tone: Writing tone ("professional", "casual", or "academic").

    Returns:
        Rewritten text.
    """
    prompt = f"""
Rewrite the following text using a {tone} tone.

Rules:
- Preserve the original meaning.
- Do not introduce new information.
- Return only the rewritten text.

Text:
{text}

Rewritten text:
"""
    return await provider.generate(prompt)


async def translate_text(text: str, target_language: str) -> str:
    """Translate text to the target language while preserving meaning and tone.

    Args:
        text: Original text to be translated.
        target_language: Target language (e.g., "indonesian", "japanese").

    Returns:
        Translated text.
    """
    prompt = f"""
Translate the following text into {target_language}.

Rules:
- Preserve the original meaning.
- Preserve the original tone where possible.
- Return only the translated text.

Text:
{text}

Translation:
"""
    return await provider.generate(prompt)
