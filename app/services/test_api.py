from fastapi.testclient import TestClient

from app.main import app

from unittest.mock import AsyncMock, patch

client = TestClient(app)


def test_health():

    response = client.get("/health")

    assert response.status_code == 200

    assert response.json() == {"status": "healthy"}


def test_summarize():
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
