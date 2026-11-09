"""Retrieval evaluation. Needs no LLM, so it is fully deterministic and reproducible.

    python -m eval.eval_retrieval

1) Compares chunking strategies and retrievers on recall@1, recall@3, recall@5 and MRR.
2) Sweeps the min_score floor for the chosen setup: how many answerable questions keep a gold
   chunk versus how many unanswerable questions are blocked before reaching the LLM.
"""
import json
from pathlib import Path

from app.config import CHUNKING, TOP_K
from app.corpus import load_rules
from app.retriever import build_retriever
from eval.metrics import first_relevant_rank, load_questions, mrr, recall_at_k

RESULTS = Path(__file__).resolve().parent.parent / "results"
CONFIGS = [
    ("overlap + section (baseline)", "section", "overlap"),
    ("bm25 + fixed 200 words (naive)", "fixed", "bm25"),
    ("bm25 + rule", "rule", "bm25"),
    ("bm25 + section", "section", "bm25"),
]
THRESHOLDS = [0, 2, 4, 5, 6, 8, 10, 12, 14]


def main() -> None:
    rules = load_rules()
    qs = load_questions()
    answerable = [q for q in qs if q["type"] != "unanswerable"]
    unanswerable = [q for q in qs if q["type"] == "unanswerable"]
    lines = [f"# Retrieval results\n\n{len(answerable)} answerable questions, {len(unanswerable)} unanswerable. "
             "A retrieved chunk counts as relevant if it contains one of the question's gold rule numbers.\n",
             "## 1. Retriever / chunking comparison\n",
             "| config | chunks | recall@1 | recall@3 | recall@5 | MRR |", "|---|---|---|---|---|---|"]
    table = {}
    for name, chunking, kind in CONFIGS:
        r = build_retriever(chunking, kind, rules)
        ranks = [first_relevant_rank([c.rules for c, _ in r.search(q["question"], k=10)], q["gold_rules"])
                 for q in answerable]
        row = {"chunks": len(r.chunks), "recall@1": recall_at_k(ranks, 1), "recall@3": recall_at_k(ranks, 3),
               "recall@5": recall_at_k(ranks, 5), "mrr": mrr(ranks)}
        table[name] = row
        lines.append(f"| {name} | {row['chunks']} | {row['recall@1']:.3f} | {row['recall@3']:.3f} | "
                     f"{row['recall@5']:.3f} | {row['mrr']:.3f} |")
        misses = [q["id"] for q, rk in zip(answerable, ranks) if rk is None or rk > 3]
        print(f"{name:32s} R@1={row['recall@1']:.3f} R@3={row['recall@3']:.3f} R@5={row['recall@5']:.3f} "
              f"MRR={row['mrr']:.3f}  not-in-top3={misses}")

    r = build_retriever(CHUNKING, "bm25", rules)
    lines += [f"\n## 2. min_score sweep (bm25 + {CHUNKING}, top-{TOP_K})\n",
              f"kept = answerable questions with a gold chunk still in the top-{TOP_K} after the floor; "
              "blocked = unanswerable questions with no chunk above the floor.\n",
              "| min_score | kept (answerable) | blocked (unanswerable) |", "|---|---|---|"]
    sweep = []
    for t in THRESHOLDS:
        kept = sum(bool(first_relevant_rank([c.rules for c, _ in r.search(q["question"], TOP_K, t)], q["gold_rules"]))
                   for q in answerable) / len(answerable)
        blocked = sum(not r.search(q["question"], TOP_K, t) for q in unanswerable) / len(unanswerable)
        sweep.append({"min_score": t, "kept": kept, "blocked": blocked})
        lines.append(f"| {t} | {kept:.3f} | {blocked:.3f} |")

    print(f"\nmin_score sweep (bm25 + {CHUNKING}, top-{TOP_K}):")
    for s in sweep:
        print(f"  {s['min_score']:>4}: kept={s['kept']:.3f} blocked={s['blocked']:.3f}")
    print("\nTop score for each unanswerable question:")
    for q in unanswerable:
        hits = r.search(q["question"], 1)
        print(f"  {q['id']} {hits[0][1] if hits else 0:.2f}  {q['question']}")
    gold_scores = []
    for q in answerable:
        s = [sc for c, sc in r.search(q["question"], TOP_K) if c.rules & set(q["gold_rules"])]
        if s:
            gold_scores.append((max(s), q["id"]))
    print("Lowest gold-chunk scores among answerable questions:", [(round(s, 2), i) for s, i in sorted(gold_scores)[:5]])

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "retrieval_results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (RESULTS / "retrieval_results.json").write_text(json.dumps({"configs": table, "sweep": sweep}, indent=2))
    print(f"\nSaved to {RESULTS / 'retrieval_results.md'}")


if __name__ == "__main__":
    main()
