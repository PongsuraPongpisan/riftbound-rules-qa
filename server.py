"""Minimal REST API.  Run: uvicorn server:app --reload   then open http://127.0.0.1:8000/docs"""
from fastapi import FastAPI

from app.schemas import AskRequest, AskResponse
from app.service import ask

app = FastAPI(title="Riftbound Rules Assistant")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask_endpoint(req: AskRequest) -> AskResponse:
    result = ask(req.question, mode=req.mode)
    return AskResponse(answer=result["answer"], citations=result["citations"], mode=req.mode)
