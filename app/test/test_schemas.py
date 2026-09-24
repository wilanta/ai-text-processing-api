import pytest
from pydantic import ValidationError

from app.schemas import TextRequest


def test_valid_text_request():

    request = TextRequest(text="Hello world")

    assert request.text == "Hello world"


def test_empty_text_rejected():

    with pytest.raises(ValidationError):
        TextRequest(text="")
