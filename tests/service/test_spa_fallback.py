"""The "frontend not built" answer must be reachable in the case it names.

The defect these tests lock down: the service used to fall back from
``web/dist`` to ``web/`` — the SOURCE tree, which is tracked and therefore
present in every clone. The message below the fallback could then never fire,
and a visitor with an unbuilt checkout got a blank page with no diagnosis while
the API worked perfectly.

A message that cannot be reached is worse than no message: it reads, to whoever
maintains the file, as a case already handled.

No models and no network here — the SPA route only looks at the filesystem, so
the tests point it at a temporary directory and check both answers. Checking
only the unbuilt case would prove nothing: a route that always returns the hint
would pass it just as well, which is why the built case is asserted too.
"""
from __future__ import annotations

import pathlib
import sys

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from service import main  # noqa: E402


@pytest.fixture
def client():
    return TestClient(main.app)


def test_unbuilt_frontend_answers_with_instructions(client, tmp_path, monkeypatch):
    """No ``index.html`` under the served root -> a page that says what to run."""
    monkeypatch.setattr(main, "_web_root", tmp_path)

    response = client.get("/")

    assert response.status_code == 503, (
        "an unbuilt UI is unavailable, and saying so with 200 OK hides it from "
        "anything that checks status rather than reading the body"
    )
    assert "npm run build" in response.text, (
        "the instruction must be IN the answer: this is the whole defect - the "
        "text existed but could not be reached"
    )
    assert "text/html" in response.headers["content-type"], (
        "the visitor is in a browser; JSON here is a diagnosis nobody reads"
    )


def test_api_still_answers_when_the_ui_is_not_built(client, tmp_path, monkeypatch):
    """The missing build must not look like a broken service."""
    monkeypatch.setattr(main, "_web_root", tmp_path)

    assert client.get("/health").status_code == 200


def test_built_frontend_is_served_and_says_nothing_about_building(
        client, tmp_path, monkeypatch):
    """The control. Without it the suite would pass on a route that always hints."""
    (tmp_path / "index.html").write_text("<p>BUILT", encoding="utf-8")
    monkeypatch.setattr(main, "_web_root", tmp_path)

    response = client.get("/")

    assert response.status_code == 200
    assert "BUILT" in response.text
    assert "npm run build" not in response.text


def test_unknown_path_falls_back_to_the_built_index(client, tmp_path, monkeypatch):
    """Client-side routing still works: unknown paths get index.html, not 404."""
    (tmp_path / "index.html").write_text("<p>BUILT", encoding="utf-8")
    monkeypatch.setattr(main, "_web_root", tmp_path)

    assert "BUILT" in client.get("/documents/42").text


def test_only_the_build_output_is_ever_served():
    """Guards the fix itself, not its symptom.

    The served root must be ``web/dist``. If a future change makes the service
    fall back to ``web/`` again "so that something shows up", every test above
    keeps passing — they point at a temporary directory — and the original
    defect returns silently. This is the one assertion that would notice.
    """
    assert main._web_root.name == "dist", (
        f"served root is {main._web_root}, expected .../web/dist. The source "
        f"tree is never a servable artifact: vite builds to dist/, the "
        f"Dockerfile copies it there, and in development vite serves the SPA "
        f"itself."
    )


def test_index_sets_no_cache_and_static_asset_is_served_without_traversal(client, tmp_path, monkeypatch):
    dist = tmp_path / "dist"
    (dist / "assets").mkdir(parents=True)
    (dist / "index.html").write_text("<p>BUILT</p>", encoding="utf-8")
    (dist / "assets" / "app.js").write_text("console.log('ok');", encoding="utf-8")
    secret = tmp_path / "secret.txt"
    secret.write_text("TOP_SECRET", encoding="utf-8")
    monkeypatch.setattr(main, "_web_root", dist)

    idx = client.get("/")
    assert idx.status_code == 200
    assert idx.headers.get("cache-control") == "no-cache, must-revalidate"

    asset = client.get("/assets/app.js")
    assert asset.status_code == 200 and "console.log" in asset.text

    # Path traversal outside dist/ must never serve parent files
    trav = client.get("/../secret.txt")
    assert "TOP_SECRET" not in trav.text
    trav_enc = client.get("/%2e%2e/secret.txt")
    assert "TOP_SECRET" not in trav_enc.text
