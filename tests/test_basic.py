"""Offline tests: no LLM calls and no Riot data needed (a small invented rules fixture is used)."""
from pathlib import Path

import pytest

import app.agent as agent_mod
from app.config import REFUSAL
from app.corpus import parse_rules_text
from app.rag import answer_rag, extract_citations
from app.retriever import build_retriever, chunk_by_section
from app.schemas import parse_agent_step
from app.tools import make_tools, run_tool
from eval.metrics import first_relevant_rank, score_answer

FIXTURE = (Path(__file__).parent / "fixtures" / "mini_rules_layout.txt").read_text()


@pytest.fixture(scope="module")
def rules():
    return parse_rules_text(FIXTURE)


@pytest.fixture(scope="module")
def retriever(rules):
    return build_retriever("section", "bm25", rules)


def test_parser_keeps_numbers_and_joins_continuation_lines(rules):
    by_id = {r.id: r.text for r in rules}
    assert by_id["102"] == "Each player shuffles their deck and draws 5 cards."
    assert by_id["202.1"].endswith("This continuation line belongs to rule 202.1.")
    assert "Last Updated" not in " ".join(by_id.values())


def test_section_chunks_use_headings(rules):
    titles = {c.title for c in chunk_by_section(rules)}
    assert {"Setup", "Points"} <= titles


def test_retriever_finds_winning_rule(retriever):
    chunk, _ = retriever.search("how many points to win", k=1)[0]
    assert "202" in chunk.rules


def test_min_score_refuses_without_calling_the_model(retriever):
    assert answer_rag("booster pack price", retriever, min_score=5.0)["answer"] == REFUSAL


def test_lookup_rule_validates_input(rules, retriever):
    tools = make_tools(retriever, rules, 0.0)
    assert "202.1" in run_tool(tools, "lookup_rule", "202")
    assert run_tool(tools, "lookup_rule", "999").startswith("Error")
    assert run_tool(tools, "lookup_rule", "; rm -rf /").startswith("Error")
    assert run_tool(tools, "delete_files", "x").startswith("Error: unknown tool")


def test_parse_agent_step_handles_fences_and_rejects_bad_actions():
    step = parse_agent_step('```json\n{"thought": "t", "action": "lookup_rule", "action_input": "202"}\n```')
    assert step.action == "lookup_rule"
    with pytest.raises(ValueError):
        parse_agent_step('{"action": "run_shell", "action_input": "ls"}')
    with pytest.raises(ValueError):
        parse_agent_step("The answer is 10 points.")


def test_citations_are_filtered_to_real_rules():
    assert extract_citations("Win at 10 [rule 202] and [rule 999.1] [202.1]", {"202"}) == ["202", "202.1"]


def test_agent_loop_with_fake_model(monkeypatch, rules, retriever):
    replies = iter([
        "not json",  # invalid -> the agent asks again
        '{"thought": "search", "action": "search_rules", "action_input": "points to win"}',
        '{"thought": "read it", "action": "lookup_rule", "action_input": "202"}',
        '{"thought": "done", "action": "final_answer", "action_input": "You win at 10 points [rule 202]."}',
    ])
    monkeypatch.setattr(agent_mod, "chat", lambda messages: next(replies))
    out = agent_mod.run_agent("How do I win?", retriever, rules)
    assert out["answer"].startswith("You win at 10 points")
    assert out["citations"] == ["202"] and out["steps"] == 4


def test_agent_stops_at_step_limit(monkeypatch, rules, retriever):
    monkeypatch.setattr(agent_mod, "chat", lambda m: '{"thought": "", "action": "search_rules", "action_input": "x"}')
    out = agent_mod.run_agent("loop forever", retriever, rules, max_steps=3)
    assert out["steps"] == 3 and "step limit" in out["answer"]


def test_scoring_rules():
    assert score_answer({"type": "single", "must_include": ["40"]}, "At least 40 cards [rule 103.3].")
    assert not score_answer({"type": "single", "must_include": ["40"]}, REFUSAL)
    assert score_answer({"type": "unanswerable", "must_include": []}, REFUSAL)
    assert first_relevant_rank([{"101"}, {"202", "203"}], ["202"]) == 2
