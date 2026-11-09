"""Retrieval-augmented answering with rule-number citations, plus a no-retrieval baseline."""
import re

from app.config import MIN_SCORE, REFUSAL, TOP_K
from app.llm import chat
from app.retriever import BM25Retriever

RAG_SYSTEM = f"""You answer questions about the rules of Riftbound, the League of Legends trading card game,
using ONLY the Core Rules excerpts provided.
Rules:
- Answer in 1-3 short sentences.
- Cite the rule number for each fact in square brackets, e.g. [rule 103.3] or [rule 116].
- If the excerpts do not contain the answer, reply exactly: {REFUSAL}
- Do not use outside knowledge, and do not guess."""

BASELINE_SYSTEM = f"""You answer questions about the rules of Riftbound, the League of Legends trading card game.
Answer in 1-3 short sentences. If you do not know the answer, reply exactly: {REFUSAL}"""

CITATION_RE = re.compile(r"\[(?:rules?\s*)?(\d{3}(?:\.[0-9a-z]+)*)\]", re.IGNORECASE)


def extract_citations(answer: str, valid_tops: set[str]) -> list[str]:
    """Model output is untrusted: keep only citations whose rule number exists in the Core Rules."""
    return [c for c in dict.fromkeys(CITATION_RE.findall(answer)) if c.split(".")[0] in valid_tops]


def format_context(hits) -> str:
    return "\n\n".join(f"### {c.title}\n{c.text}" if c.title else c.text for c, _ in hits)


def valid_tops(retriever: BM25Retriever) -> set[str]:
    return {t for c in retriever.chunks for t in c.rules}


def answer_rag(question: str, retriever: BM25Retriever, k: int = TOP_K, min_score: float = MIN_SCORE) -> dict:
    hits = retriever.search(question, k=k, min_score=min_score)
    if not hits:  # nothing passed the score floor: refuse without calling the model
        return {"answer": REFUSAL, "citations": [], "retrieved": []}
    messages = [
        {"role": "system", "content": RAG_SYSTEM},
        {"role": "user", "content": f"Core Rules excerpts:\n\n{format_context(hits)}\n\nQuestion: {question}"},
    ]
    answer = chat(messages).strip()
    return {
        "answer": answer,
        "citations": extract_citations(answer, valid_tops(retriever)),
        "retrieved": [{"id": c.id, "rules": sorted(c.rules), "score": round(s, 2)} for c, s in hits],
    }


def answer_no_retrieval(question: str) -> dict:
    """Baseline: the same model with no access to the rules."""
    messages = [{"role": "system", "content": BASELINE_SYSTEM}, {"role": "user", "content": question}]
    return {"answer": chat(messages).strip(), "citations": [], "retrieved": []}
