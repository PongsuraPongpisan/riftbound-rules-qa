"""One entry point used by both the CLI and the API, so they behave identically."""
from functools import lru_cache

from app.agent import run_agent
from app.config import CHUNKING
from app.corpus import load_rules
from app.rag import answer_no_retrieval, answer_rag
from app.retriever import build_retriever


@lru_cache(maxsize=1)
def get_rules():
    return load_rules()


@lru_cache(maxsize=1)
def get_retriever():
    return build_retriever(chunking=CHUNKING, kind="bm25", rules=get_rules())


def ask(question: str, mode: str = "rag", verbose: bool = False) -> dict:
    if mode == "rag":
        return answer_rag(question, get_retriever())
    if mode == "agent":
        return run_agent(question, get_retriever(), get_rules(), verbose=verbose)
    if mode == "none":
        return answer_no_retrieval(question)
    raise ValueError(f"unknown mode: {mode}")
