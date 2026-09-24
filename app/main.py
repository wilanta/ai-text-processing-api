"""Application entry point for the AI text processing API.

Exposes a REST interface over FastAPI. Each route maps one-to-one to a function
in app.services.ai_services; this file handles only request binding, response
formatting, and global exception translation.
"""

# pylint: disable=import-error
from fastapi import FastAPI, Request  # pyright: ignore[reportMissingImports]
from fastapi.responses import JSONResponse

from app.schemas import TextRequest, RewriteRequest, TranslateRequest
from app.services.ai_services import (
    summarize_text,
    classify_text,
    extract_entities,
    rewrite_text,
    translate_text,
)
from app.exceptions import (
    AIConnectionError,
    AITimeoutError,
    AIResponseError,
)

# Initialize the FastAPI application with documentation metadata.
# The version field is incremented on each breaking API change.
app = FastAPI(
    title="AI Proccesing API",
    description="AI-powered text processing API",
    version="10.0.0",
)


# Register exception handlers so that AI-service failures return structured
# JSON with the correct HTTP status instead of leaking internal tracebacks.
@app.exception_handler(AIConnectionError)
async def ai_connection_handler(
    request: Request,
    exc: AIConnectionError,
):
    return JSONResponse(
        status_code=503,
        content={
            "error": "ai_service_unavailable",
            "detail": str(exc),
        },
    )


@app.exception_handler(AITimeoutError)
async def ai_timeout_handler(
    request: Request,
    exc: AITimeoutError,
):
    return JSONResponse(
        status_code=504,
        content={
            "error": "ai_timeout",
            "detail": str(exc),
        },
    )


@app.exception_handler(AIResponseError)
async def ai_response_handler(
    request: Request,
    exc: AIResponseError,
):
    return JSONResponse(
        status_code=502,
        content={
            "error": "invalid_ai_response",
            "detail": str(exc),
        },
    )


# Root endpoint — basic application info
@app.get("/")
def root():
    return {"message": "AI text proccesing is running"}


# Health check endpoint — verifies the service is running
@app.get("/health")
def health():
    return {"status": "healthy"}


# Summarize text using AI model
@app.post("/summarize")
async def summarize(request: TextRequest):
    summary = await summarize_text(request.text)
    return {"summary": summary}


# Classify text: category + sentiment
@app.post("/classify")
async def classify(request: TextRequest):
    return await classify_text(request.text)


# Extract named entities from text
@app.post("/extract")
async def extract(request: TextRequest):
    return await extract_entities(request.text)


# Rewrite text in the requested tone
@app.post("/rewrite")
async def rewrite(request: RewriteRequest):
    result = await rewrite_text(request.text, request.tone)
    return {"rewritten_text": result}


# Translate text to target language
@app.post("/translate")
async def translate(request: TranslateRequest):
    result = await translate_text(request.text, request.target_language)
    return {"translation": result, "target_language": request.target_language}
