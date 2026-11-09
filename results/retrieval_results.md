# Retrieval results

32 answerable questions, 5 unanswerable. A retrieved chunk counts as relevant if it contains one of the question's gold rule numbers.

## 1. Retriever / chunking comparison

| config | chunks | recall@1 | recall@3 | recall@5 | MRR |
|---|---|---|---|---|---|
| overlap + section (baseline) | 225 | 0.500 | 0.625 | 0.750 | 0.598 |
| bm25 + fixed 200 words (naive) | 395 | 0.750 | 0.844 | 0.875 | 0.812 |
| bm25 + rule | 432 | 0.625 | 0.844 | 0.906 | 0.742 |
| bm25 + section | 225 | 0.812 | 0.875 | 0.906 | 0.857 |

## 2. min_score sweep (bm25 + section, top-3)

kept = answerable questions with a gold chunk still in the top-3 after the floor; blocked = unanswerable questions with no chunk above the floor.

| min_score | kept (answerable) | blocked (unanswerable) |
|---|---|---|
| 0 | 0.875 | 0.000 |
| 2 | 0.875 | 0.000 |
| 4 | 0.875 | 0.000 |
| 5 | 0.875 | 0.200 |
| 6 | 0.844 | 0.200 |
| 8 | 0.781 | 0.400 |
| 10 | 0.688 | 0.600 |
| 12 | 0.594 | 0.600 |
| 14 | 0.406 | 0.600 |
