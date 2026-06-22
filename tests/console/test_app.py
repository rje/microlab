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


def test_login_next_rejects_crlf(client):
    token = _csrf_from_login(client)
    response = client.post(
        "/login",
        data={
            "password": "test-password-123",
            "csrf_token": token,
            "next": "/x\r\nSet-Cookie: a=b",
        },
        follow_redirects=False,
    )
    assert response.status_code == 302
    location = response.headers["Location"]
    assert "\r" not in location and "\n" not in location
    assert "a=b" not in location


def _login(client):
    token = _csrf_from_login(client)
    client.post("/login", data={"password": "test-password-123", "csrf_token": token})
    return client.get("/api/state").get_json()["csrfToken"]


def _seed_mmlu(project_root):
    import json

    (project_root / "papers" / "manifest.json").write_text(
        json.dumps(
            [
                {
                    "topic": "evaluation",
                    "title": "Measuring Massive Multitask Language Understanding",
                    "authors": "H",
                    "year": 2020,
                    "source_url": "https://arxiv.org/abs/2009.03300",
                    "pdf_url": "https://arxiv.org/pdf/2009.03300",
                    "filename": "mmlu.pdf",
                }
            ]
        ),
        encoding="utf-8",
    )


def test_state_includes_progress_and_csrf(auth_client):
    body = auth_client.get("/api/state").get_json()
    assert "csrfToken" in body and body["csrfToken"]
    assert isinstance(body["papers"], list)


def test_progress_requires_csrf(auth_client):
    resp = auth_client.post("/api/papers/mmlu/progress", json={"readState": "mapped"})
    assert resp.status_code == 403


def test_progress_round_trip(client, project_root):
    _seed_mmlu(project_root)
    csrf = _login(client)
    resp = client.post(
        "/api/papers/mmlu/progress",
        json={"readState": "mapped", "depth": "understand"},
        headers={"X-CSRF-Token": csrf},
    )
    assert resp.status_code == 200, resp.get_data(as_text=True)
    papers = client.get("/api/state").get_json()["papers"]
    mmlu = next(p for p in papers if p["id"] == "mmlu")
    assert mmlu["progress"] == {"readState": "mapped", "depth": "understand"}


def test_progress_rejects_unknown_paper(client):
    csrf = _login(client)
    resp = client.post(
        "/api/papers/not-a-paper/progress",
        json={"readState": "mapped"},
        headers={"X-CSRF-Token": csrf},
    )
    assert resp.status_code == 404


def test_progress_rejects_bad_read_state(client, project_root):
    _seed_mmlu(project_root)
    csrf = _login(client)
    resp = client.post(
        "/api/papers/mmlu/progress",
        json={"readState": "banana"},
        headers={"X-CSRF-Token": csrf},
    )
    assert resp.status_code == 400


def test_notes_round_trip_endpoint(client, project_root):
    _seed_mmlu(project_root)
    csrf = _login(client)
    assert client.get("/api/papers/mmlu/notes").get_json()["content"] == ""
    resp = client.post(
        "/api/papers/mmlu/notes",
        json={"content": "RoPE rotates Q/K."},
        headers={"X-CSRF-Token": csrf},
    )
    assert resp.status_code == 200
    assert "RoPE rotates" in client.get("/api/papers/mmlu/notes").get_json()["content"]


def test_overview_404_when_unknown_paper(auth_client):
    assert auth_client.get("/api/papers/not-a-paper/overview").status_code == 404


def test_overview_404_when_absent(client, project_root):
    _seed_mmlu(project_root)
    _login(client)
    assert client.get("/api/papers/mmlu/overview").status_code == 404


def test_overview_returns_json_when_present(client, project_root):
    import json

    _seed_mmlu(project_root)
    overview = project_root / "content" / "papers" / "mmlu" / "overview.json"
    overview.parent.mkdir(parents=True, exist_ok=True)
    overview.write_text(
        json.dumps({"paperId": "mmlu", "tldr": "hi", "sections": []}),
        encoding="utf-8",
    )
    _login(client)
    body = client.get("/api/papers/mmlu/overview").get_json()
    assert body["paperId"] == "mmlu"
    assert body["tldr"] == "hi"


def test_public_routes_need_no_auth(client, project_root):
    _seed_mmlu(project_root)
    assert client.get("/public").status_code == 200
    lib = client.get("/public/api/library")
    assert lib.status_code == 200
    assert "phases" in lib.get_json()


def test_public_library_has_no_private_data(client, project_root):
    _seed_mmlu(project_root)
    blob = client.get("/public/api/library").get_data(as_text=True)
    assert "progress" not in blob and "notes" not in blob


def test_private_routes_still_gated_without_session(client, project_root):
    _seed_mmlu(project_root)
    assert client.get("/api/state").status_code == 401
    assert client.get("/api/papers/mmlu/notes").status_code == 401
    assert client.get("/api/papers/mmlu/overview").status_code == 401
    # SPA root redirects to login when unauthenticated
    assert client.get("/").status_code == 302


def test_public_pdf_serves_and_404s(client, project_root):
    _seed_mmlu(project_root)
    pdf = project_root / "papers" / "evaluation" / "mmlu.pdf"
    pdf.parent.mkdir(parents=True, exist_ok=True)
    pdf.write_bytes(b"%PDF-1.4 test")
    assert client.get("/public/pdf/mmlu").status_code == 200
    assert client.get("/public/pdf/not-a-paper").status_code == 404


def test_static_assets_served_without_auth(client, project_root):
    assets = project_root / "site" / "dist" / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    (assets / "x.js").write_text("console.log(1)", encoding="utf-8")
    resp = client.get("/assets/x.js")
    assert resp.status_code == 200
    assert client.get("/assets/../../secret.txt").status_code in (400, 404)


def _seed_phase_task(project_root):
    import json

    task = {"id": "t1", "title": "Task one", "status": "queued", "why": "w", "links": []}
    (project_root / "site" / "content" / "phases.json").write_text(
        json.dumps([
            {"id": "phase-0", "title": "Phase 0", "status": "current", "goal": "g",
             "tasks": [task], "readingPaperIds": []}
        ]),
        encoding="utf-8",
    )


def test_task_status_override_in_state(client, project_root):
    _seed_phase_task(project_root)
    csrf = _login(client)
    resp = client.post(
        "/api/phases/phase-0/tasks/t1/status",
        json={"status": "done"},
        headers={"X-CSRF-Token": csrf},
    )
    assert resp.status_code == 200, resp.get_data(as_text=True)
    state = client.get("/api/state").get_json()
    task = state["phases"][0]["tasks"][0]
    assert task["status"] == "done"


def test_task_status_requires_csrf(auth_client, project_root):
    _seed_phase_task(project_root)
    resp = auth_client.post("/api/phases/phase-0/tasks/t1/status", json={"status": "done"})
    assert resp.status_code == 403


def test_task_status_rejects_unknown_phase_or_task(client, project_root):
    _seed_phase_task(project_root)
    csrf = _login(client)
    assert client.post("/api/phases/nope/tasks/t1/status", json={"status": "done"},
                       headers={"X-CSRF-Token": csrf}).status_code == 404
    assert client.post("/api/phases/phase-0/tasks/nope/status", json={"status": "done"},
                       headers={"X-CSRF-Token": csrf}).status_code == 404


def test_task_status_rejects_bad_status(client, project_root):
    _seed_phase_task(project_root)
    csrf = _login(client)
    assert client.post("/api/phases/phase-0/tasks/t1/status", json={"status": "banana"},
                       headers={"X-CSRF-Token": csrf}).status_code == 400


def _seed_cards(project_root):
    import json
    _seed_mmlu(project_root)
    cards = project_root / "content" / "papers" / "mmlu" / "cards.json"
    cards.parent.mkdir(parents=True, exist_ok=True)
    cards.write_text(json.dumps({"paperId": "mmlu", "cards": [
        {"id": "mmlu#1", "question": "Q1?", "answer": "A1"},
        {"id": "mmlu#2", "question": "Q2?", "answer": "A2"},
    ]}), encoding="utf-8")


def test_recall_due_lists_new_cards(client, project_root):
    _seed_cards(project_root)
    _login(client)
    body = client.get("/api/recall/due").get_json()
    assert body["total"] == 2
    assert {c["id"] for c in body["cards"]} == {"mmlu#1", "mmlu#2"}
    assert all(c["status"] == "new" for c in body["cards"])


def test_recall_due_requires_auth(client):
    assert client.get("/api/recall/due").status_code == 401


def test_recall_review_round_trip(client, project_root):
    _seed_cards(project_root)
    csrf = _login(client)
    resp = client.post("/api/recall/review",
                       json={"cardId": "mmlu#1", "paperId": "mmlu", "grade": 4},
                       headers={"X-CSRF-Token": csrf})
    assert resp.status_code == 200, resp.get_data(as_text=True)
    # mmlu#1 now scheduled in the future -> only mmlu#2 remains due
    body = client.get("/api/recall/due").get_json()
    assert {c["id"] for c in body["cards"]} == {"mmlu#2"}


def test_recall_review_requires_csrf(auth_client, project_root):
    _seed_cards(project_root)
    resp = auth_client.post(
        "/api/recall/review", json={"cardId": "mmlu#1", "paperId": "mmlu", "grade": 4}
    )
    assert resp.status_code == 403


def test_recall_review_unknown_card_404(client, project_root):
    _seed_cards(project_root)
    csrf = _login(client)
    assert client.post("/api/recall/review",
                       json={"cardId": "nope#9", "paperId": "mmlu", "grade": 4},
                       headers={"X-CSRF-Token": csrf}).status_code == 404


def test_recall_review_bad_grade_400(client, project_root):
    _seed_cards(project_root)
    csrf = _login(client)
    assert client.post("/api/recall/review",
                       json={"cardId": "mmlu#1", "paperId": "mmlu", "grade": 99},
                       headers={"X-CSRF-Token": csrf}).status_code == 400
