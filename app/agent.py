"""A small tool-using agent loop.

The model replies with one JSON object per step (validated by AgentStep). We run the chosen tool,
feed the result back as an observation, and stop on final_answer or after AGENT_MAX_STEPS steps.
"""
from pydantic import ValidationError

from app.config import AGENT_MAX_STEPS, MIN_SCORE, REFUSAL
from app.corpus import Rule
from app.llm import chat
from app.rag import extract_citations, valid_tops
from app.retriever import BM25Retriever
from app.schemas import parse_agent_step
from app.tools import make_tools, run_tool

AGENT_SYSTEM = f"""You answer questions about the rules of Riftbound, the League of Legends trading card game.
You can use tools to read the official Core Rules.

Tools:
- search_rules: input is a short search query. Returns matching rules with their numbers.
- lookup_rule: input is a rule number such as 469 or 103.3. Returns that rule and its sub-rules.
  Use it when a rule says "See rule X" and you need rule X.
- final_answer: input is your answer for the user (1-3 sentences), citing rule numbers like [rule 116].

Rules:
- Always use search_rules before answering. Use only facts from the rules you read.
- If the rules do not contain the answer, your final answer must be exactly: {REFUSAL}

Reply with ONE JSON object and nothing else, in this format:
{{"thought": "what you will do next", "action": "search_rules" | "lookup_rule" | "final_answer", "action_input": "..."}}"""


def run_agent(question: str, retriever: BM25Retriever, rules: list[Rule], max_steps: int = AGENT_MAX_STEPS,
              min_score: float = MIN_SCORE, verbose: bool = False) -> dict:
    tools = make_tools(retriever, rules, min_score)
    messages = [{"role": "system", "content": AGENT_SYSTEM}, {"role": "user", "content": question}]
    trace = []

    for step_no in range(1, max_steps + 1):
        raw = chat(messages)
        messages.append({"role": "assistant", "content": raw})
        try:
            step = parse_agent_step(raw)
        except (ValueError, ValidationError) as e:
            trace.append({"step": step_no, "error": f"invalid step: {str(e)[:200]}"})
            messages.append({"role": "user", "content":
                             "Your reply was not a valid JSON object in the required format. "
                             "Reply with exactly one JSON object with keys thought, action, action_input."})
            continue

        if verbose:
            print(f"[step {step_no}] {step.action}: {step.action_input}")

        if step.action == "final_answer":  # stop condition 1: the model is done
            trace.append({"step": step_no, "action": "final_answer"})
            return {"answer": step.action_input.strip(),
                    "citations": extract_citations(step.action_input, valid_tops(retriever)),
                    "steps": step_no, "trace": trace}

        observation = run_tool(tools, step.action, step.action_input)
        trace.append({"step": step_no, "action": step.action, "input": step.action_input,
                      "observation": observation[:200]})
        if verbose:
            print(f"    -> {observation[:150]!r}")
        messages.append({"role": "user", "content": f"Observation:\n{observation}"})

    # stop condition 2: step limit reached
    return {"answer": "Sorry, I couldn't finish answering within the step limit.",
            "citations": [], "steps": max_steps, "trace": trace}
