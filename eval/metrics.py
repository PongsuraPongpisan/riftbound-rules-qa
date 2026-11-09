"""Scoring helpers shared by the evaluation scripts."""
import json
import re
from pathlib import Path

from app.config import REFUSAL

QUESTIONS_PATH = Path(__file__).resolve().parent / "questions.jsonl"


def load_questions(path: Path = QUESTIONS_PATH) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def normalize(text: str) -> str:
    return re.sub(r"(?<=\d),(?=\d)", "", text.lower())  # "1,500" -> "1500"


def is_refusal(answer: str) -> bool:
    a = normalize(answer)
    return any(p in a for p in [REFUSAL.lower().rstrip("."), "could not find", "not in the core rules"])


def contains(answer: str, alternatives: str) -> bool:
    """True if any 'a|b|c' alternative appears in the answer as a whole word/number."""
    a = normalize(answer)
    return any(re.search(rf"(?<![\w.]){re.escape(alt.lower())}(?![\w])", a) for alt in alternatives.split("|"))


def score_answer(q: dict, answer: str) -> bool:
    if q["type"] == "unanswerable":
        return is_refusal(answer)
    return not is_refusal(answer) and all(contains(answer, m) for m in q["must_include"])


# ---------- retrieval metrics (a chunk is relevant if it contains one of the gold rules) ----------

def first_relevant_rank(retrieved: list[set[str]], gold_rules: list[str]):
    for rank, rules in enumerate(retrieved, 1):
        if rules & set(gold_rules):
            return rank
    return None


def recall_at_k(ranks: list, k: int) -> float:
    return sum(1 for r in ranks if r is not None and r <= k) / len(ranks)


def mrr(ranks: list) -> float:
    return sum(1 / r for r in ranks if r is not None) / len(ranks)
