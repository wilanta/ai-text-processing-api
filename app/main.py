"""Application entry point for the AI text processing API."""

# pylint: disable=import-error
from fastapi import FastAPI  # pyright: ignore[reportMissingImports]

from app.schemas import TextRequest, RewriteRequest, TranslateRequest
from app.services.ai_services import (
    summarize_text,
    classify_text,
    extract_entities,
    rewrite_text,
    translate_text,
)

# Initialize the FastAPI application with documentation metadata
app = FastAPI(
    title="AI Proccesing API",
    description="AI-powered text processing API",
    version="10.0.0",
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
def summarize(request: TextRequest):
    summary = summarize_text(request.text)
    return {"summary": summary}


# Classify text: category + sentiment
@app.post("/classify")
def classify(request: TextRequest):
    return classify_text(request.text)


# Extract named entities from text
@app.post("/extract")
def extract(request: TextRequest):
    return extract_entities(request.text)


# Rewrite text in the requested tone
@app.post("/rewrite")
def rewrite(request: RewriteRequest):
    result = rewrite_text(request.text, request.tone)
    return {"rewritten_text": result}


# Translate text to target language
@app.post("/translate")
def translate(request: TranslateRequest):
    result = translate_text(request.text, request.target_language)
    return {"translation": result, "target_language": request.target_language}
