from __future__ import annotations

import importlib.util
from pathlib import Path

import httpx

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "smoke.py"
SPEC = importlib.util.spec_from_file_location("rotostream_smoke", SCRIPT)
assert SPEC and SPEC.loader
smoke = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(smoke)


def test_session_login_keeps_cookie_and_sends_password_only_to_signin() -> None:
    requests: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path == "/auth/login":
            return httpx.Response(
                200,
                json={"ok": True},
                headers={"set-cookie": "rotostream_session=test-session; Path=/; Secure; HttpOnly; SameSite=Lax"},
            )
        return httpx.Response(200, json={"status": "ok"})

    base_url = "https://rotostream.example"
    username = "editor"
    password = "never-log-this-password"
    with httpx.Client(base_url=base_url, transport=httpx.MockTransport(handle)) as client:
        response = smoke.login_editor_session(client, base_url, username, password)
        assert response.status_code == 200
        assert client.get("/api/health").status_code == 200
        assert client.cookies.get("rotostream_session") == "test-session"

    assert requests[0].url.path == "/auth/login"
    assert requests[0].headers["origin"] == base_url
    assert requests[0].read().decode() == '{"username":"editor","password":"never-log-this-password"}'
    assert requests[1].url.path == "/api/health"
    assert requests[1].headers["cookie"] == "rotostream_session=test-session"
