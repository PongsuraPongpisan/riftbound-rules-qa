"""Turn the Core Rules PDF into a list of numbered rules.

The PDF lists every rule as a number at the start of a line ("103.3.b.  Your Main Deck can ...")
followed by indented continuation lines. We keep that structure: one record per rule number.
"""
import json
import re
from dataclasses import dataclass
from pathlib import Path

from app.config import RULES_JSONL

RULE_LINE = re.compile(r"^\s{0,3}(\d{3}\.(?:[0-9a-z]+\.)*)(?:\s+(.*))?$")


@dataclass
class Rule:
    id: str     # "103.3.b"
    text: str

    @property
    def top(self) -> str:
        """Top-level rule number, e.g. "103" for "103.3.b"."""
        return self.id.split(".")[0]


def parse_rules_text(text: str) -> list[Rule]:
    rules: list[Rule] = []
    for line in text.split("\n"):
        m = RULE_LINE.match(line)
        if m:
            rules.append(Rule(m.group(1).rstrip("."), (m.group(2) or "").strip()))
        elif rules and line.strip():
            rules[-1].text += " " + line.strip()
    for r in rules:
        r.text = re.sub(r"\s+", " ", r.text).strip()
    return rules


def parse_pdf(pdf_path: Path) -> list[Rule]:
    from pypdf import PdfReader  # imported here so the rest of the app does not need pypdf
    reader = PdfReader(str(pdf_path))
    text = "\n".join(page.extract_text(extraction_mode="layout") for page in reader.pages)
    return parse_rules_text(text)


def save_rules(rules: list[Rule], path: Path = RULES_JSONL) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps({"id": r.id, "text": r.text}, ensure_ascii=False) for r in rules) + "\n",
                    encoding="utf-8")


def load_rules(path: Path = RULES_JSONL) -> list[Rule]:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `python -m scripts.fetch_rules` first (see README).")
    return [Rule(**json.loads(line)) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def is_heading(rule: Rule, has_children: bool) -> bool:
    """Headings are short titles like "193. Winning" with no sub-rules and no full stop."""
    return not has_children and len(rule.text.split()) <= 5 and not rule.text.endswith(".")
