import os
import sys
from typing import Optional

import httpx
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

# Keep imports compatible with the project's app-level module layout.
APP_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "app"))
CONFIG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "configs"))
os.environ.setdefault("FOAP_CONFIG_DIR", CONFIG_DIR)
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

import utils  # noqa: E402


def _create_test_app() -> FastAPI:
    app = FastAPI()

    @app.post("/proxy")
    async def proxy(request: Request):
        response = await utils.handle_request(request, "v1/chat/completions")
        return utils.process_completion_response(response)

    return app


class _FakeAsyncClient:
    next_response: Optional[httpx.Response] = None

    def __init__(self, *args, **kwargs):
        self.closed = False

    def build_request(self, method, url, **kwargs):
        return httpx.Request(method, url, headers=kwargs.get("headers"), json=kwargs.get("json"))

    async def send(self, request, stream=False):
        assert self.next_response is not None
        return self.next_response

    async def post(self, url, json=None, headers=None):
        assert self.next_response is not None
        return self.next_response

    async def aclose(self):
        self.closed = True


def _patch_auth_and_model(monkeypatch):
    monkeypatch.setattr(utils, "can_request", lambda model, token: True)
    monkeypatch.setattr(
        utils.models,
        "get_model_data",
        lambda model, api_path: {
            "provider": "test",
            "target_model_name": "target-model",
            "target_base_url": "https://backend.local",
            "api_key": "backend-token",
            "request_timeout": 5,
        },
    )


def test_streaming_upstream_error_keeps_error_status_and_payload(monkeypatch):
    _patch_auth_and_model(monkeypatch)
    monkeypatch.setattr(utils.httpx, "AsyncClient", _FakeAsyncClient)

    _FakeAsyncClient.next_response = httpx.Response(
        status_code=401,
        headers={"Content-Type": "application/json"},
        json={"error": {"message": "invalid key"}},
        request=httpx.Request("POST", "https://backend.local/v1/chat/completions"),
    )

    client = TestClient(_create_test_app())
    response = client.post(
        "/proxy",
        headers={"Authorization": "Bearer client-token"},
        json={"model": "gpt-4o", "stream": True, "messages": [{"role": "user", "content": "hi"}]},
    )

    assert response.status_code == 401
    assert response.json() == {"error": {"message": "invalid key"}}


def test_streaming_success_preserves_status_and_headers(monkeypatch):
    _patch_auth_and_model(monkeypatch)
    monkeypatch.setattr(utils.httpx, "AsyncClient", _FakeAsyncClient)

    _FakeAsyncClient.next_response = httpx.Response(
        status_code=200,
        headers={"Content-Type": "text/event-stream", "x-request-id": "req_123"},
        content=b"data: {\"id\":\"abc\"}\n\n",
        request=httpx.Request("POST", "https://backend.local/v1/chat/completions"),
    )

    client = TestClient(_create_test_app())
    response = client.post(
        "/proxy",
        headers={"Authorization": "Bearer client-token"},
        json={"model": "gpt-4o", "stream": True, "messages": [{"role": "user", "content": "hi"}]},
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.headers.get("x-request-id") == "req_123"
    assert "data:" in response.text


