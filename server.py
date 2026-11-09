"""Web app + REST API.

    uvicorn server:app --reload      then open http://127.0.0.1:8000  (API docs at /docs)
"""
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse

from app.config import DAILY_LIMIT, RATE_LIMIT_PER_MINUTE
from app.ratelimit import RateLimiter
from app.schemas import AskRequest, AskResponse, RuleSource
from app.service import ask, get_rules

app = FastAPI(title="Riftbound Rules Assistant")
limiter = RateLimiter(RATE_LIMIT_PER_MINUTE, DAILY_LIMIT)
INDEX = Path(__file__).resolve().parent / "static" / "index.html"
MAX_SOURCE_CHARS = 1500


def client_id(request: Request) -> str:
    """Hosting platforms put the visitor's real IP first in X-Forwarded-For."""
    forwarded = request.headers.get("x-forwarded-for", "")
    return forwarded.split(",")[0].strip() or (request.client.host if request.client else "unknown")


def rule_sources(citations: list[str]) -> list[RuleSource]:
    """The text of each cited rule and its direct sub-rules, so readers can check the answer."""
    rules = get_rules()
    out = []
    for c in citations:
        depth = c.count(".")
        lines = [f"{r.id}. {r.text}" for r in rules
                 if r.id == c or (r.id.startswith(c + ".") and r.id.count(".") == depth + 1)]
        text = ""
        for line in lines:  # trim on a line boundary, never mid-rule
            if text and len(text) + len(line) > MAX_SOURCE_CHARS:
                text += "\n…"
                break
            text += ("\n" if text else "") + line
        if text:
            out.append(RuleSource(rule=c, text=text))
    return out


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(INDEX)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask_endpoint(req: AskRequest, request: Request) -> AskResponse:
    limited = limiter.check(client_id(request))
    if limited:
        raise HTTPException(status_code=429, detail=limited)
    try:
        result = ask(req.question, mode=req.mode)
    except Exception as e:  # the model service is down, slow or refused the request
        raise HTTPException(status_code=502, detail="The model service didn't respond. Try again in a minute.") from e
    return AskResponse(answer=result["answer"], citations=result["citations"],
                       sources=rule_sources(result["citations"]), mode=req.mode)
