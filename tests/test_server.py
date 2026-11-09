"""Offline tests for the web server: the model and the rules are faked."""
from fastapi.testclient import TestClient

import server
from app.corpus import Rule
from app.ratelimit import RateLimiter

FAKE_RULES = [Rule("194", "Winning."), Rule("194.3", "The Victory Score is 8 points by default."), Rule("195", "Other.")]


def make_client(monkeypatch, per_minute=6, per_day=300):
    monkeypatch.setattr(server, "ask", lambda q, mode: {"answer": "You need 8 points [rule 194.3].",
                                                         "citations": ["194.3"]})
    monkeypatch.setattr(server, "get_rules", lambda: FAKE_RULES)
    monkeypatch.setattr(server, "limiter", RateLimiter(per_minute, per_day))
    return TestClient(server.app)


def test_index_page_is_served(monkeypatch):
    res = make_client(monkeypatch).get("/")
    assert res.status_code == 200 and "Riftbound rules lookup" in res.text


def test_ask_returns_cited_rule_text(monkeypatch):
    res = make_client(monkeypatch).post("/ask", json={"question": "How do I win?", "mode": "rag"})
    assert res.status_code == 200
    assert res.json()["sources"] == [{"rule": "194.3", "text": "194.3. The Victory Score is 8 points by default."}]


def test_per_minute_limit(monkeypatch):
    client = make_client(monkeypatch, per_minute=2)
    codes = [client.post("/ask", json={"question": "How do I win?"}).status_code for _ in range(3)]
    assert codes == [200, 200, 429]


def test_daily_limit_applies_to_everyone():
    limiter = RateLimiter(per_minute=100, per_day=2)
    assert limiter.check("a", now=0) is None and limiter.check("b", now=1) is None
    assert "daily" in limiter.check("c", now=2)


def test_model_failure_gives_clear_error(monkeypatch):
    client = make_client(monkeypatch)
    def boom(q, mode):
        raise RuntimeError("proxy down")
    monkeypatch.setattr(server, "ask", boom)
    res = client.post("/ask", json={"question": "How do I win?"})
    assert res.status_code == 502 and "Try again" in res.json()["detail"]


def test_invalid_question_rejected(monkeypatch):
    assert make_client(monkeypatch).post("/ask", json={"question": "x"}).status_code == 422
