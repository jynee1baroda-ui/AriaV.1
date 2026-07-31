"""API tests (Phase 2, Milestone 1).

Uses FastAPI's TestClient, which drives the real ASGI app over HTTP
without opening a port - fast, deterministic, no server needed.
"""

import unittest

from fastapi.testclient import TestClient

from app.main import app


class TestHealthEndpoint(unittest.TestCase):
    """GET /api/v1/health must report a healthy, identified service."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.client = TestClient(app)

    def test_health_returns_200_and_identity(self) -> None:
        response = self.client.get("/api/v1/health")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["assistant"], "Aria")
        self.assertIsInstance(payload["version"], str)

    def test_unknown_route_returns_404(self) -> None:
        response = self.client.get("/api/v1/nonexistent")
        self.assertEqual(response.status_code, 404)


if __name__ == "__main__":
    unittest.main()
