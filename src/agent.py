"""메인 에이전트 그래프: 입력 가드레일 → ReAct 에이전트(도구 3개) → 출력 가드레일 순서로 질문에 답한다."""

import json
import sys
import uuid
from datetime import date
from functools import lru_cache

from langchain.agents import create_agent
from langchain.agents.middleware import ModelFallbackMiddleware
from langchain.agents.structured_output import ToolStrategy
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import BaseModel, Field

from guardrails import check_input, check_output, mask_pii
from llm import build_models
from tools import (
    calc_checkup_schedule,
    calc_vaccine_schedule,
    get_age_months,
    get_baby_records,
    load_profile,
    retrieve_guide,
)

TOOLS = [retrieve_guide, get_baby_records, calc_vaccine_schedule, calc_checkup_schedule]
TOOL_NAMES = {t.name for t in TOOLS}
# trace 에 남길 도구 출력의 최대 길이
TRACE_OUTPUT_LIMIT = 500

SYSTEM_PROMPT = """너는 생후 0~24개월 첫째 아이를 키우는 초보 부모를 돕는 'AI 육아도우미'다.

[답변 원칙]
1. 반드시 도구로 조회한 결과만 근거로 답한다. 도구 결과에 없는 내용은 지어내지 않는다.
2. 육아 정보 질문은 retrieve_guide 로 가이드 문서를 검색한 뒤 답한다.
   우리 아이의 월령·기록이 필요한 질문은 get_baby_records 를 먼저 부른다.
   예방접종 일정 질문은 get_baby_records 로 생년월일을 확인한 뒤 calc_vaccine_schedule 을 부른다.
   영유아 건강검진 질문에는 retrieve_guide 로 차수별 일반 시기를 확인하고, get_baby_records 로 생년월일을
   확인한 뒤 calc_checkup_schedule 로 우리 아이의 다음 검진 차수·시작일도 함께 안내한다.
   검진 마감일은 계산하지 말고 국민건강보험공단 '영유아 검진일자 조회'로 확인하도록 안내한다.
3. 가이드 문서로 답할 때는 답변 끝에 "(출처: 문서 제목)" 형식으로 출처를 적는다.
4. 검색 결과에 질문에 대한 내용이 없으면 "제가 가진 육아 가이드 문서에는 해당 정보가 없습니다"라고 말하고
   소아청소년과 상담을 안내하는 것으로 답을 끝낸다. 사이트·기관·앱 이름, 일반 상식 조언, 학회 권고처럼
   도구 결과에 없는 내용은 도움이 될 것 같아도 덧붙이지 않는다. 비슷한 주제의 다른 내용으로 대신 답하지 않는다.
5. 질병을 진단하거나 약 이름·용량을 안내하지 않는다. 그런 질문에는 의사나 약사와 상의하도록 안내한다.
6. 발달 질문에는 개인차가 있다는 점을 함께 말하고, 걱정되면 소아청소년과 상담을 권한다. 단정하지 않는다.
7. 날짜를 계산해 답할 때는 기준 날짜를 함께 적는다.
8. 정중한 존댓말로, 핵심부터 간결하게 답한다. 이모지는 쓰지 않는다.
"""


class AgentAnswer(BaseModel):
    """에이전트의 최종 답변 형식 (구조화 출력)."""

    answer: str = Field(description="사용자에게 보여줄 최종 답변")
    sources: list[str] = Field(default_factory=list, description="근거로 쓴 가이드 문서 제목 목록. 없으면 빈 목록")


@lru_cache(maxsize=1)
def build_agent():
    """Bedrock 모델과 도메인 도구로 ReAct 에이전트 그래프를 한 번만 만든다.

    대체 모델이 설정되어 있으면 ModelFallbackMiddleware 로 1순위 모델 실패 시 순서대로 넘긴다.
    """
    primary, *fallbacks = build_models()
    middleware = [ModelFallbackMiddleware(*fallbacks)] if fallbacks else []
    return create_agent(
        model=primary,
        tools=TOOLS,
        system_prompt=SYSTEM_PROMPT,
        response_format=ToolStrategy(AgentAnswer),
        middleware=middleware,
        checkpointer=InMemorySaver(),
    )


def _truncate(text: str, limit: int = TRACE_OUTPUT_LIMIT) -> str:
    """trace 에 넣을 긴 문자열을 잘라 낸다."""
    return text if len(text) <= limit else text[:limit] + "...(생략)"


def _collect_turn(messages: list) -> tuple[list[dict], list[dict], AIMessage | None]:
    """이번 질문 이후의 메시지에서 사용 모델·도구 호출 trace, contexts, 마지막 AI 메시지를 모은다."""
    last_human = max(i for i, m in enumerate(messages) if isinstance(m, HumanMessage))
    turn = messages[last_human + 1 :]
    used_models = list(
        dict.fromkeys(
            f"{m.response_metadata.get('model_name', 'unknown')}@{m.response_metadata.get('region', '?')}"
            for m in turn
            if isinstance(m, AIMessage)
        )
    )
    tool_args = {
        call["id"]: call["args"]
        for m in turn
        if isinstance(m, AIMessage)
        for call in m.tool_calls
    }
    trace = [{"step": "model", "input": "agent", "output": ", ".join(used_models)}]
    contexts, seen = [], set()
    for m in turn:
        if not isinstance(m, ToolMessage) or m.name not in TOOL_NAMES:
            continue
        trace.append(
            {
                "step": f"tool:{m.name}",
                "input": json.dumps(tool_args.get(m.tool_call_id, {}), ensure_ascii=False),
                "output": mask_pii(_truncate(str(m.content))),
            }
        )
        # 문서 검색은 찾은 문서 조각을, 그 밖의 도구(기록 조회·계산)는 도구 결과 자체를 답변의 근거로 남긴다
        if m.artifact:
            tool_contexts = m.artifact
        else:
            tool_contexts = [{"doc_id": f"tool:{m.name}", "text": mask_pii(str(m.content))}]
        for ctx in tool_contexts:
            key = (ctx["doc_id"], ctx["text"])
            if key not in seen:
                seen.add(key)
                contexts.append(ctx)
    last_ai = next((m for m in reversed(turn) if isinstance(m, AIMessage)), None)
    return trace, contexts, last_ai


def run_query(question: str, thread_id: str | None = None) -> dict:
    """질문 하나를 처리해 answer, contexts, trace 세 키를 담은 dict 를 돌려준다.

    thread_id 가 같으면 이전 대화를 이어 가고, 없으면 새 대화로 처리한다.
    """
    trace = []
    masked = mask_pii(question)
    if masked != question:
        trace.append({"step": "guardrail:pii_mask", "input": "(개인정보 포함 질문)", "output": masked})

    profile_age = get_age_months(date.fromisoformat(load_profile()["birth_date"]))
    blocked = check_input(masked, profile_age)
    trace.append({"step": "guardrail:input", "input": masked, "output": blocked[0] if blocked else "pass"})
    if blocked:
        trace.append({"step": "answer", "input": masked, "output": blocked[1]})
        return {"answer": blocked[1], "contexts": [], "trace": trace}

    config = {"configurable": {"thread_id": thread_id or str(uuid.uuid4())}}
    result = build_agent().invoke({"messages": [{"role": "user", "content": masked}]}, config)
    tool_trace, contexts, last_ai = _collect_turn(result["messages"])
    trace.extend(tool_trace)

    structured = result.get("structured_response")
    draft = structured.answer if structured else (last_ai.text if last_ai else "")
    checked, verdict = check_output(draft)
    trace.append(
        {
            "step": "guardrail:output",
            "input": _truncate(draft),
            "output": f"violates: {verdict.reason}" if verdict.violates else "pass",
        }
    )
    answer = mask_pii(checked)
    trace.append({"step": "answer", "input": masked, "output": answer})
    return {"answer": answer, "contexts": contexts, "trace": trace}


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sample_questions = [
        "우리 아기 다음 예방접종 언제야?",
        "아기가 갑자기 몸을 떨면서 경련을 해요",
        "30개월 아이가 말이 늦은데 어떡하죠?",
    ]
    for q in sample_questions:
        response = run_query(q)
        print(f"질문: {q}")
        for step in response["trace"]:
            if step["step"].startswith("tool:"):
                print(f"  [도구 호출] {step['step'][5:]}({step['input']})")
            elif step["step"].startswith("guardrail:") and step["output"] != "pass":
                print(f"  [가드레일] {step['step'][10:]} → {step['output']}")
        print(f"  contexts: {[c['doc_id'] for c in response['contexts']]}")
        print(f"답변: {response['answer']}\n")
