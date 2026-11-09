"""Tools the agent may call. Only names in the allow-list can run, and inputs are checked first."""
import re

from app.config import TOP_K
from app.corpus import Rule
from app.rag import format_context
from app.retriever import BM25Retriever

RULE_ID = re.compile(r"^\d{3}(\.[0-9a-z]+)*$")
MAX_OBSERVATION_CHARS = 3000


def make_tools(retriever: BM25Retriever, rules: list[Rule], min_score: float) -> dict:
    by_id = {r.id: r for r in rules}

    def search_rules(query: str) -> str:
        hits = retriever.search(query, k=TOP_K, min_score=min_score)
        return format_context(hits) if hits else "No relevant rules found."

    def lookup_rule(number: str) -> str:
        """Return a rule and all of its sub-rules, e.g. "469" or "103.3"."""
        number = number.strip().strip("[]").removeprefix("rule").strip().rstrip(".")
        if not RULE_ID.match(number):
            return "Error: give a rule number such as 469 or 103.3"
        if number not in by_id:
            return f"Error: rule {number} does not exist"
        found = [r for r in rules if r.id == number or r.id.startswith(number + ".")]
        return "\n".join(f"{r.id}. {r.text}" for r in found)

    return {"search_rules": search_rules, "lookup_rule": lookup_rule}


def run_tool(tools: dict, name: str, arg: str) -> str:
    if name not in tools:  # allow-list check
        return f"Error: unknown tool '{name}'"
    return tools[name](arg)[:MAX_OBSERVATION_CHARS]  # cap what goes back to the model
