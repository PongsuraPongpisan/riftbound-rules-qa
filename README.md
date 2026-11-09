# Riftbound Rules Assistant

Ask questions about the rules of **Riftbound**, Riot's League of Legends trading card game, and get short answers that cite the official rule numbers, e.g. *"A Main Deck needs at least 40 cards [rule 103.3]."* If the Core Rules don't cover a question, the assistant says so instead of guessing.

**Who it's for:** players learning the game, and anyone settling a rules question at the table. The official Core Rules are a 142-page document with about 2,500 numbered rules, which is too long to search by hand mid-game.

**Why it's a good RAG test:** Riftbound launched in late 2025, so the model has little or no knowledge of it. The no-retrieval baseline shows how much the retrieval actually adds.

All LLM calls go to the class proxy: `OpenAI()` with `model="gemma-4-E4B-it"`.

## Course capability families covered

| # | Family | Where |
| --- | --- | --- |
| 1 | Structured output with schema validation | `app/schemas.py`: every agent step is JSON validated by Pydantic, and the API request and response are validated too |
| 2 | RAG with citations | `app/corpus.py` (PDF → numbered rules), `app/retriever.py` (section chunks + BM25), `app/rag.py` (rule-number citations, `min_score` floor) |
| 3 | Retrieval and answer evaluation | `eval/`: 37 labelled questions, recall@k, MRR, accuracy, refusal and citation accuracy, baselines |
| 4 | Agent loop with tools | `app/agent.py`, `app/tools.py`: allow-listed `search_rules` and `lookup_rule` tools (the agent can follow "See rule X" references), stop on `final_answer` or after 5 steps |
| 6 | Served interface | `server.py`: FastAPI `POST /ask` |

## Setup

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
export OPENAI_API_KEY=...        # your class key
export OPENAI_BASE_URL=...       # the class proxy URL

# Download and parse the official Core Rules (version 2026-10-09)
python -m scripts.fetch_rules
# If the download fails, download the PDF from https://playriftbound.com/en-us/rules-hub/ and run:
python -m scripts.fetch_rules --pdf path/to/RiftboundCoreRules.pdf
```

The rules text belongs to Riot Games, so the PDF and the parsed rules are **not committed** (see `.gitignore`). The fetch script recreates them. The script pins the URL and SHA-256 of the exact version the evaluation was written against, and warns if your copy differs.

No key or URL is stored in the repo; `OpenAI()` reads both from the environment (see `.env.example`).

## Run

```bash
# Command line
python cli.py "How many cards are in a Rune Deck?"
python cli.py --mode agent "What happens when I conquer a battlefield one point away from winning?"
python cli.py                     # interactive

# REST API: then open http://127.0.0.1:8000/docs
uvicorn server:app --reload
curl -X POST localhost:8000/ask -H "Content-Type: application/json" \
     -d '{"question": "How many points do I need to win?", "mode": "rag"}'

# Offline tests (no API key and no Riot data needed: they use a small invented rules fixture)
python -m pytest -q
```

Modes: `rag` (default), `agent` (tool loop), `none` (baseline, no retrieval).

## How it works

```
PDF ──pypdf──► 2,494 numbered rules ──► section chunks (≤350 words, heading attached) ──► BM25 index
                                                                                         │
question ─────────────────────────────► top-3 chunks with score ≥ min_score ◄────────────┘
                                          │ none pass → refuse without calling the LLM
                                          ▼
                              Gemma (temperature 0) answers from the excerpts and cites [rule N]
                                          ▼
                              citations checked against real rule numbers
```

In agent mode, Gemma instead chooses step by step between `search_rules`, `lookup_rule` and `final_answer`. Each step is a JSON object validated against a schema, and the loop stops at the final answer or after 5 steps.

## Evaluation

The test set is `eval/questions.jsonl`, with 37 hand-written questions:

- 30 single-fact questions
- 2 hard questions, phrased far from the rule text or needing a cross-reference
- 5 questions the Core Rules don't answer (prices, release dates, the meta, card stats, prizes), where the correct behaviour is to refuse

Each answerable question is labelled with its gold rule number(s) and the facts the answer must contain. All questions are checked against the 2026-10-09 rules.

```bash
python -m eval.eval_retrieval                          # no LLM needed, fully deterministic
python -m eval.eval_answers --mode none                # baseline: same model, no retrieval
python -m eval.eval_answers --mode rag --min-score 0   # RAG without the score floor
python -m eval.eval_answers --mode rag                 # RAG, min_score 5.0
python -m eval.eval_answers --mode agent               # agent
python -m eval.eval_answers --summary                  # writes results/answer_results.md
```

### Retrieval results (from `results/retrieval_results.md`)

These are the 32 answerable questions. A chunk counts as relevant if it contains a gold rule.

| config | recall@1 | recall@3 | recall@5 | MRR |
| --- | --- | --- | --- | --- |
| word-overlap + section chunks (baseline) | 0.500 | 0.625 | 0.750 | 0.598 |
| BM25 + fixed 200-word chunks (naive default) | 0.750 | 0.844 | 0.875 | 0.812 |
| BM25 + one chunk per rule number | 0.625 | 0.844 | 0.906 | 0.742 |
| **BM25 + section chunks (used)** | **0.812** | **0.875** | **0.906** | **0.857** |

**Measured improvement over the naive default.** Chunking by the rulebook's own sections, with the section heading attached, raises recall@1 from 0.750 to 0.812 and MRR from 0.812 to 0.857, compared with fixed 200-word windows.

Smaller one-rule chunks do worse at the top (recall@1 of 0.625). Many rules are a single line like "116. Players each draw 4.", which has too few words for BM25 to match.

**The `min_score` floor helps only a little here (honest result).** 5.0 is the highest floor that loses no answerable question, but it blocks only 1 of 5 unanswerable questions. "How much does a Riftbound booster pack cost?" scores 14.3 because rule 702.1 mentions booster packs, and "How much Might does Jinx, Rebel have?" scores 18.5 because rule 103.3 uses Jinx as an example. So refusing those has to come from the LLM.

**Retrieval misses (top-3)** come from vocabulary mismatch. "Opening hand" never appears in rule 116 ("Players each draw 4"), and "below zero" doesn't match rule 194.4 ("cannot have less than 0 points").

### Answer results

Results on all 37 questions (temperature 0):

| run | overall | answerable | hard | refusal | citation | sec/q |
| --- | --- | --- | --- | --- | --- | --- |
| none (baseline, no retrieval) | 0.297 | 0.219 | 0.000 | 0.800 | 0.000 | 0.4 |
| rag, min_score 0 | 0.838 | 0.812 | 1.000 | 1.000 | 0.812 | 0.7 |
| **rag, min_score 5 (default)** | **0.838** | **0.812** | **1.000** | **1.000** | **0.812** | **0.6** |
| agent, min_score 5 | 0.811 | 0.781 | 0.500 | 1.000 | 0.719 | 2.3 |

- **Retrieval is what matters:** answerable accuracy rises from 0.219 without the rules to 0.812 with RAG. Without the rules, the model confidently invents answers (7-card hands, 4 copies per card, runes that produce "Mana").
- **The model refuses well on its own:** RAG refused all 5 unanswerable questions even with no score floor, so the min_score floor did not change accuracy here.
- **The agent did not beat plain RAG** on this set. Most questions need only one rule, and the agent was about 4× slower. With 37 questions, one question is worth about 3 points, so the gap between them is small.
- **RAG's 6 misses:** 3 are retrieval misses from wording mismatches (q04, q07, q10). 1 retrieved the right rule number but not the sentence with the answer (q01). In 2 the right text was retrieved but the model misread it (q19, q23). Full list: `results/failures.txt`.

## Project structure

```
app/
  config.py      env-based settings + the pinned rules version
  corpus.py      PDF → numbered rules (parser), load/save
  retriever.py   chunking strategies, BM25, the word-overlap baseline
  rag.py         RAG answer with rule citations, and the no-retrieval baseline
  schemas.py     Pydantic models and agent-step JSON parsing
  tools.py       allow-listed tools: search_rules, lookup_rule
  agent.py       agent loop with stop conditions
  llm.py         the only place the OpenAI client is created and called
  service.py     single entry point used by the CLI and the API
scripts/fetch_rules.py   download + verify + parse the Core Rules
cli.py, server.py
eval/            questions.jsonl, metrics, retrieval eval, answer eval
results/         saved evaluation outputs
tests/           offline unit tests (with an invented rules fixture)
```

## Safety choices

- **Config:** API settings come from the environment only.
- **Agent output:** each step must parse as JSON and pass the `AgentStep` schema. On failure, the model is asked again; it is never executed blindly.
- **Tools:** only allow-listed tool names can run. `lookup_rule` accepts only a valid rule-number pattern and only rules that exist. Tool output is length-capped before it goes back to the model.
- **Citations:** a citation is kept only if the rule number exists, since model output is untrusted.
- **API input:** request length and mode are validated.

## Limitations

- **Rules version:** the rules change with every set, and the eval is tied to the 2026-10-09 version.
- **Keyword-only retrieval:** BM25 misses paraphrases ("opening hand"). Embeddings or query rewriting would help.
- **Rules only:** card text and stats are not covered, so card-specific questions should be refused.
- **Scoring:** answers are scored by string-matching required facts, so check the saved `.jsonl` files by hand.
- **Model size:** Gemma E4B is small, so complex timing and interaction questions are likely to fail.
