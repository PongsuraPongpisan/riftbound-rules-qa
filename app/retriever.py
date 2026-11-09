"""Chunk the numbered rules and search them with BM25 (plus a naive word-overlap baseline)."""
import math
import re
from collections import Counter
from dataclasses import dataclass, field

from app.corpus import Rule, is_heading, load_rules

STOPWORDS = set(
    "a an the and or of to in on for at by with is are was were be been it its this that i my me we our "
    "you your do does did can how what when where who which much many there their they as from if per "
    "after before about any into than then so will would".split()
)
MAX_WORDS = 350  # chunks longer than this are split at sub-rule boundaries


@dataclass
class Chunk:
    id: str                 # internal id, e.g. "section:110" or "fixed:42"
    title: str              # section heading, e.g. "Setup Process"
    text: str               # rule lines as "116. Players each draw 4."
    rules: set[str] = field(default_factory=set)  # top-level rule numbers covered, e.g. {"116", "117"}


def tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOPWORDS]


def _with_headings(rules: list[Rule]) -> list[tuple[str, Rule]]:
    """Attach the current section heading to every rule and drop the heading-only rules."""
    tops_with_children = {r.top for r in rules if "." in r.id}
    out, heading = [], ""
    for r in rules:
        if "." not in r.id and is_heading(r, r.top in tops_with_children):
            heading = r.text
            continue
        out.append((heading, r))
    return out


def _pack(lines: list[tuple[str, Rule]], prefix: str) -> list[Chunk]:
    """Pack consecutive rules into chunks of at most MAX_WORDS words."""
    chunks, buf, words = [], [], 0
    for heading, r in lines:
        n = len(r.text.split()) + 1
        if buf and words + n > MAX_WORDS:
            chunks.append(buf); buf, words = [], 0
        buf.append((heading, r)); words += n
    if buf:
        chunks.append(buf)
    return [Chunk(f"{prefix}:{b[0][1].id}", b[0][0], "\n".join(f"{r.id}. {r.text}" for _, r in b),
                  {r.top for _, r in b}) for b in chunks]


def chunk_by_section(rules: list[Rule]) -> list[Chunk]:
    """One chunk per heading section (e.g. all of 'Setup Process'), split if too long."""
    groups: dict[str, list] = {}
    for heading, r in _with_headings(rules):
        groups.setdefault(heading, []).append((heading, r))
    return [c for g in groups.values() for c in _pack(g, "section")]


def chunk_by_rule(rules: list[Rule]) -> list[Chunk]:
    """One chunk per top-level rule number (e.g. 116 with all its sub-rules), split if too long."""
    groups: dict[str, list] = {}
    for heading, r in _with_headings(rules):
        groups.setdefault(r.top, []).append((heading, r))
    return [c for g in groups.values() for c in _pack(g, "rule")]


def chunk_fixed(rules: list[Rule], size: int = 200, overlap: int = 40) -> list[Chunk]:
    """Naive default: fixed word windows that ignore the rule structure."""
    words, owner = [], []
    for r in rules:
        for w in f"{r.id}. {r.text}".split():
            words.append(w); owner.append(r.top)
    chunks = []
    for i, start in enumerate(range(0, len(words), size - overlap)):
        end = min(start + size, len(words))
        chunks.append(Chunk(f"fixed:{i}", "", " ".join(words[start:end]), set(owner[start:end])))
        if end == len(words):
            break
    return chunks


CHUNKERS = {"section": chunk_by_section, "rule": chunk_by_rule, "fixed": chunk_fixed}


class BM25Retriever:
    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75):
        self.chunks = chunks
        self.k1, self.b = k1, b
        self.tfs = [Counter(tokenize(f"{c.title} {c.text}")) for c in chunks]
        self.lens = [sum(tf.values()) for tf in self.tfs]
        self.avgdl = sum(self.lens) / len(self.lens)
        df = Counter(term for tf in self.tfs for term in tf)
        n = len(chunks)
        self.idf = {t: math.log(1 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}

    def score(self, query: str) -> list[float]:
        q = tokenize(query)
        scores = []
        for tf, dl in zip(self.tfs, self.lens):
            s = 0.0
            for t in q:
                if t in tf:
                    f = tf[t]
                    s += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
            scores.append(s)
        return scores

    def search(self, query: str, k: int = 3, min_score: float = 0.0) -> list[tuple[Chunk, float]]:
        ranked = sorted(zip(self.chunks, self.score(query)), key=lambda x: x[1], reverse=True)
        return [(c, s) for c, s in ranked[:k] if s > 0 and s >= min_score]


class OverlapRetriever(BM25Retriever):
    """Baseline: score = number of distinct query words in the chunk (no IDF, no length normalisation)."""

    def score(self, query: str) -> list[float]:
        q = set(tokenize(query))
        return [float(len(q & tf.keys())) for tf in self.tfs]


def build_retriever(chunking: str = "section", kind: str = "bm25", rules: list[Rule] | None = None) -> BM25Retriever:
    chunks = CHUNKERS[chunking](rules if rules is not None else load_rules())
    return BM25Retriever(chunks) if kind == "bm25" else OverlapRetriever(chunks)
