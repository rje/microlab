from __future__ import annotations

import re


def _csrf_from_login(client) -> str:
    html = client.get("/login").get_data(as_text=True)
    match = re.search(r'name="csrf_token" value="([^"]+)"', html)
    assert match, "login form must include a csrf token"
    return match.group(1)


def test_api_state_requires_auth(client):
    response = client.get("/api/state")
    assert response.status_code == 401


def test_login_success_grants_state(client):
    token = _csrf_from_login(client)
    response = client.post(
        "/login",
        data={"password": "test-password-123", "csrf_token": token},
        follow_redirects=False,
    )
    assert response.status_code == 302
    state = client.get("/api/state")
    assert state.status_code == 200
    assert state.get_json()["phases"][0]["id"] == "phase-0"


def test_login_rejects_wrong_password(client):
    token = _csrf_from_login(client)
    response = client.post(
        "/login",
        data={"password": "nope", "csrf_token": token},
    )
    assert response.status_code == 401
    assert client.get("/api/state").status_code == 401


def test_login_rejects_missing_csrf(client):
    response = client.post("/login", data={"password": "test-password-123"})
    assert response.status_code == 400


def test_logout_clears_session(auth_client):
    assert auth_client.get("/api/state").status_code == 200
    assert auth_client.post("/logout").status_code in (200, 302)
    assert auth_client.get("/api/state").status_code == 401


def test_markdown_route_requires_auth(client):
    assert client.get("/api/markdown?path=plans/note.md").status_code == 401


def test_markdown_route_returns_document_when_authed(auth_client):
    response = auth_client.get("/api/markdown?path=plans/note.md")
    assert response.status_code == 200
    assert response.get_json()["title"] == "Note"


def test_markdown_route_rejects_unpublished_path(auth_client):
    assert auth_client.get("/api/markdown?path=environment.yml").status_code == 400


def test_spa_index_requires_auth(client):
    response = client.get("/")
    assert response.status_code == 302
    assert "/login" in response.headers["Location"]


def test_spa_index_served_when_authed(auth_client):
    response = auth_client.get("/")
    assert response.status_code == 200
    assert b"Console" in response.data


def test_login_next_rejects_open_redirect(client):
    token = _csrf_from_login(client)
    response = client.post(
        "/login",
        data={"password": "test-password-123", "csrf_token": token, "next": "//evil.com"},
        follow_redirects=False,
    )
    assert response.status_code == 302
    assert "evil.com" not in response.headers["Location"]
