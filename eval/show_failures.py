"""Print every wrong answer from each saved run: python -m eval.show_failures"""
import json
from pathlib import Path

for f in sorted(Path("results").glob("answers_*.jsonl")):
    print(f"\n===== {f.stem} =====")
    for line in f.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if not r["correct"]:
            print(f"{r['id']} [{r['type']}] {r['question']}")
            print(f"   expected: {r['must_include']}  gold rules: {r['gold_rules']}")
            print(f"   answer:   {r['answer'][:200]}\n")