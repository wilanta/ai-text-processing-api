from fastapi.testclient import TestClient

from app.main import app

# unittest.mock patching lets us inject fake responses without a running Ollama instance.
from unittest.mock import AsyncMock, patch

client = TestClient(app)


def test_health():
    """Verify the health endpoint returns 200 with the expected body — smoke test that the app started."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_summarize():
    """Verify /summarize calls the AI provider and returns the generated text wrapped in {"summary": ...}.

    Patches provider.generate so no real model call is made during the test.
    """
    with patch(
        "app.services.ai_services.provider.generate",
        new=AsyncMock(return_value="Short summary."),
    ):
        response = client.post(
            "/summarize",
            json={"text": "A very long text."},
        )

    assert response.status_code == 200
    assert response.json() == {"summary": "Short summary."}
