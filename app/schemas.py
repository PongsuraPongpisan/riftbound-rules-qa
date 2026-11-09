"""Pydantic schemas: every structured value that crosses a boundary is validated here."""
import json
import re
from typing import Literal

from pydantic import BaseModel, Field


class AgentStep(BaseModel):
    """One step the agent model must produce, as JSON."""
    thought: str = ""
    action: Literal["search_rules", "lookup_rule", "final_answer"]
    action_input: str = Field(min_length=1, max_length=500)


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)
    mode: Literal["rag", "agent", "none"] = "rag"


class RuleSource(BaseModel):
    rule: str
    text: str


class AskResponse(BaseModel):
    answer: str
    citations: list[str]
    sources: list[RuleSource] = []
    mode: str


def parse_agent_step(raw: str) -> AgentStep:
    """Extract the first JSON object from the model's reply (it may add ```json fences) and validate it.

    Raises ValueError (or pydantic.ValidationError, a subclass) if the reply is not a valid step.
    """
    text = re.sub(r"```(?:json)?", "", raw).strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("no JSON object found in the reply")
    return AgentStep.model_validate(json.loads(text[start:end + 1]))
