"""اختبارات HTTP: كل نقطةٍ يجب أن تُرجع ما يكفي لعرضها بلا أخطاء."""

from __future__ import annotations

import pytest

fastapi = pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from mishkat.api.app import create_app  # noqa: E402


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(create_app())


def test_health(client: TestClient) -> None:
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_stats_shape(client: TestClient) -> None:
    r = client.get("/api/stats")
    assert r.status_code == 200
    body = r.json()
    for key in ("corpus", "index", "space", "hnsw", "graph", "inference"):
        assert key in body


def test_search_basic(client: TestClient) -> None:
    r = client.get("/api/search", params={"q": "ابن الهيثم والضوء", "k": 3})
    assert r.status_code == 200
    body = r.json()
    assert body["results"]
    assert body["results"][0]["doc_id"] in {"opt-01", "opt-02"}
    assert "trace" in body and body["trace"]["terms"]


def test_search_empty_query(client: TestClient) -> None:
    r = client.get("/api/search", params={"q": ""})
    assert r.status_code == 200
    assert r.json()["results"] == []


def test_search_modes(client: TestClient) -> None:
    for mode in ("hybrid", "lexical", "semantic"):
        r = client.get("/api/search", params={"q": "النجوم", "mode": mode})
        assert r.status_code == 200


def test_search_invalid_mode(client: TestClient) -> None:
    r = client.get("/api/search", params={"q": "x", "mode": "bogus"})
    assert r.status_code == 422


def test_doc_detail(client: TestClient) -> None:
    r = client.get("/api/doc/opt-01")
    assert r.status_code == 200
    body = r.json()
    assert body["document"]["id"] == "opt-01"
    assert body["graph"]["out"]


def test_doc_missing(client: TestClient) -> None:
    assert client.get("/api/doc/does-not-exist").status_code == 404


def test_domains(client: TestClient) -> None:
    body = client.get("/api/domains").json()
    assert body["domains"]
    assert all("icon" in d for d in body["domains"])


def test_graph_overview(client: TestClient) -> None:
    body = client.get("/api/graph/overview", params={"limit": 50}).json()
    assert body["nodes"]
    assert body["edges"]


def test_graph_query_ok(client: TestClient) -> None:
    r = client.post("/api/graph/query", json={"query": "MATCH (?x) -[ألّف]-> (?y) LIMIT 3"})
    assert r.status_code == 200
    assert r.json()["count"] >= 1
    assert r.json()["labels"]


def test_graph_query_invalid(client: TestClient) -> None:
    r = client.post("/api/graph/query", json={"query": "SELECT nope"})
    assert r.status_code == 400


def test_graph_ego(client: TestClient) -> None:
    body = client.get("/api/graph/ego/doc:opt-01").json()
    assert len(body["nodes"]) > 1


def test_word_neighbors(client: TestClient) -> None:
    body = client.get("/api/words/مناعة/neighbors", params={"k": 5}).json()
    assert "neighbors" in body


def test_space_map(client: TestClient) -> None:
    body = client.get("/api/space/map").json()
    assert len(body["points"]) >= 50
    assert -1.01 <= body["points"][0]["x"] <= 1.01


def test_clusters(client: TestClient) -> None:
    body = client.get("/api/clusters", params={"k": 4}).json()
    assert body["clusters"]


def test_analyze(client: TestClient) -> None:
    body = client.post("/api/analyze", json={"text": "العلم نور والجهل ظلام"}).json()
    assert body["counts"]["tokens"] == 4
    assert body["tokens"][0]["root"]


def test_analyze_requires_text(client: TestClient) -> None:
    assert client.post("/api/analyze", json={"text": ""}).status_code == 422


def test_word_card(client: TestClient) -> None:
    body = client.get("/api/word/المدارس").json()
    assert body["analysis"]["root"] == "درس"
    assert body["candidates"]


def test_studio_served(client: TestClient) -> None:
    r = client.get("/")
    assert r.status_code == 200
    assert "مِشكاة" in r.text


def test_assets_served(client: TestClient) -> None:
    assert client.get("/assets/style.css").status_code == 200
    assert client.get("/assets/app.js").status_code == 200


def test_unknown_path_404(client: TestClient) -> None:
    assert client.get("/nope/nope").status_code == 404


def test_lm_status(client: TestClient) -> None:
    body = client.get("/api/lm/status").json()
    assert "trained" in body
