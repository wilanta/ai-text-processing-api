import pytest
from pydantic import ValidationError

from app.schemas import TextRequest


def test_valid_text_request():
    """A normal payload passes validation and stores the text unchanged."""
    request = TextRequest(text="Hello world")
    assert request.text == "Hello world"


def test_empty_text_rejected():
    """Empty string violates min_length=1 and must raise ValidationError — this prevents useless calls to the AI provider."""
    with pytest.raises(ValidationError):
        TextRequest(text="")
