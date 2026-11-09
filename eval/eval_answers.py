"""End-to-end answer evaluation (calls the LLM through the class proxy).

    python -m eval.eval_answers --mode none          # baseline: no retrieval
    python -m eval.eval_answers --mode rag           # RAG, min_score from config (5.0)
    python -m eval.eval_answers --mode rag --min-score 0   # RAG with the naive default floor
    python -m eval.eval_answers --mode agent         # tool-using agent
    python -m eval.eval_answers --summary            # print a table of every saved run

Metrics (all on eval/questions.jsonl):
  answerable accuracy  - answer contains every required fact and is not a refusal
  refusal accuracy     - unanswerable questions answered with the refusal sentence
  citation accuracy    - answerable questions that cite one of the gold rule numbers
  overall accuracy     - all questions, each scored by the rule for its type
"""
import argparse
import json
import time
from pathlib import Path

from app.agent import run_agent
from app.config import CHUNKING, MIN_SCORE
from app.corpus import load_rules
from app.rag import answer_no_retrieval, answer_rag
from app.retriever import build_retriever
from eval.metrics import load_questions, score_answer

RESULTS = Path(__file__).resolve().parent.parent / "results"


def summarize(rows: list[dict]) -> dict:
    ans = [r for r in rows if r["type"] != "unanswerable"]
    una = [r for r in rows if r["type"] == "unanswerable"]
    hard = [r for r in rows if r["type"] == "hard"]
    pct = lambda xs, key: sum(r[key] for r in xs) / len(xs) if xs else 0.0
    return {
        "n": len(rows),
        "overall_acc": pct(rows, "correct"),
        "answerable_acc": pct(ans, "correct"),
        "hard_acc": pct(hard, "correct"),
        "refusal_acc": pct(una, "correct"),
        "citation_acc": pct(ans, "cites_gold"),
        "avg_seconds": sum(r["seconds"] for r in rows) / len(rows),
    }


def run(mode: str, min_score: float, limit: int | None) -> None:
    qs = load_questions()[:limit]
    rules = load_rules()
    retriever = build_retriever(CHUNKING, "bm25", rules)
    rows = []
    for q in qs:
        t0 = time.time()
        if mode == "none":
            out = answer_no_retrieval(q["question"])
        elif mode == "rag":
            out = answer_rag(q["question"], retriever, min_score=min_score)
        else:
            out = run_agent(q["question"], retriever, rules, min_score=min_score)
        correct = score_answer(q, out["answer"])
        cites_gold = any(c.split(".")[0] in q["gold_rules"] for c in out["citations"])
        rows.append({**q, "answer": out["answer"], "citations": out["citations"], "correct": correct,
                     "cites_gold": cites_gold, "seconds": round(time.time() - t0, 2),
                     "steps": out.get("steps")})
        print(f"{q['id']} {'OK ' if correct else 'BAD'} {out['answer'][:90]!r}")

    name = f"{mode}" if mode == "none" else f"{mode}_min{min_score:g}"
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"answers_{name}.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    s = summarize(rows)
    print(json.dumps(s, indent=2))
    print(f"Saved results/answers_{name}.jsonl")


def print_summary() -> None:
    files = sorted(RESULTS.glob("answers_*.jsonl"))
    if not files:
        print("No runs yet. Run with --mode first.")
        return
    lines = ["| run | n | overall | answerable | hard | refusal | citation | sec/q |", "|---|---|---|---|---|---|---|---|"]
    for f in files:
        rows = [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
        s = summarize(rows)
        lines.append(f"| {f.stem.removeprefix('answers_')} | {s['n']} | {s['overall_acc']:.3f} | {s['answerable_acc']:.3f} | "
                     f"{s['hard_acc']:.3f} | {s['refusal_acc']:.3f} | {s['citation_acc']:.3f} | {s['avg_seconds']:.1f} |")
    table = "\n".join(lines)
    print(table)
    (RESULTS / "answer_results.md").write_text("# Answer results\n\n" + table + "\n")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["none", "rag", "agent"])
    p.add_argument("--min-score", type=float, default=MIN_SCORE)
    p.add_argument("--limit", type=int, default=None, help="only run the first N questions (quick test)")
    p.add_argument("--summary", action="store_true")
    a = p.parse_args()
    if a.summary:
        print_summary()
    elif a.mode:
        run(a.mode, a.min_score, a.limit)
    else:
        p.error("give --mode or --summary")


if __name__ == "__main__":
    main()
