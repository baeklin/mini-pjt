"""HTTP API: POST /query 로 질문을 받아 answer, contexts, trace 를 돌려준다."""

from fastapi import FastAPI
from pydantic import BaseModel, Field

from agent import run_query

app = FastAPI(title="AI 육아도우미")


class QueryRequest(BaseModel):
    """POST /query 요청 본문."""

    question: str = Field(description="사용자 질의")
    thread_id: str | None = Field(default=None, description="같은 대화를 이어 갈 때 쓰는 대화 ID (선택)")


class Context(BaseModel):
    """답변 근거로 쓴 가이드 문서 조각."""

    doc_id: str
    text: str


class TraceStep(BaseModel):
    """처리 단계 하나의 기록."""

    step: str
    input: str
    output: str


class QueryResponse(BaseModel):
    """POST /query 응답 본문 (제출 규약: answer, contexts, trace 세 키)."""

    answer: str
    contexts: list[Context]
    trace: list[TraceStep]


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> dict:
    """질문 하나를 에이전트로 처리해 answer, contexts, trace 를 돌려준다."""
    return run_query(request.question, request.thread_id)
