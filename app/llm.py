"""Thin wrapper around the class proxy. Every LLM call in the project goes through chat()."""
from openai import OpenAI

from app.config import MODEL

_client = None


def get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI()  # reads OPENAI_API_KEY and OPENAI_BASE_URL from the environment
    return _client


def chat(messages: list[dict], temperature: float = 0.0, max_tokens: int = 512) -> str:
    """Send messages to the model and return the reply text. temperature=0 for repeatable evals."""
    resp = get_client().chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content or ""
