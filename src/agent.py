"""메인 에이전트 그래프: 입력 가드레일 → ReAct 에이전트(도메인 도구) → 출력 가드레일 순서로 질문에 답한다.

소아과 예약(book_clinic)은 실행 직전에 멈추고, 같은 thread_id 로 사용자의 승인('예'/'아니오')을 받아 재개한다.
"""

import json
import re
import sys
import uuid
from datetime import date
from functools import lru_cache

from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware, ModelFallbackMiddleware
from langchain.agents.structured_output import ToolStrategy
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.types import Command
from langsmith import traceable
from pydantic import BaseModel, Field

from guardrails import check_input, check_output, mask_pii
from llm import build_models
from tools import (
    book_clinic,
    calc_checkup_schedule,
    calc_vaccine_schedule,
    find_clinic,
    get_age_months,
    get_baby_records,
    load_profile,
    retrieve_guide,
    search_clinics,
)

TOOLS = [
    retrieve_guide,
    get_baby_records,
    calc_vaccine_schedule,
    calc_checkup_schedule,
    search_clinics,
    book_clinic,
]
# 사람 승인 없이는 실행하지 않는 도구
APPROVAL_TOOLS = {"book_clinic"}
# 승인은 짧은 승인 표현으로만 이루어진 답만 인정한다. '예약 시간 바꿔줘'처럼 '예'로 시작하는 다른 말을
# 승인으로 잘못 읽지 않도록, 애매하면 승인하지 않고 다시 묻는다.
APPROVE_PATTERN = re.compile(
    r"^(예|네|응|넵|좋아요?|그래요?|승인(합니다|할게요)?)\s*[,.!~]?\s*"
    r"((예약|진행)?\s*해\s*(줘|주세요|요)?)?\s*[.!~]*$"
)
REJECT_WORDS = ("아니", "취소", "싫어", "거절", "하지 마", "안 할")
TOOL_NAMES = {t.name for t in TOOLS}
# trace 에 남길 도구 출력의 최대 길이
TRACE_OUTPUT_LIMIT = 500

SYSTEM_PROMPT = """너는 생후 0~24개월 첫째 아이를 키우는 초보 부모를 돕는 'AI 육아도우미'다.

[답변 원칙]
1. 반드시 도구로 조회한 결과만 근거로 답한다. 도구 결과에 없는 내용은 지어내지 않는다.
   아이 기록·계산 결과로 답할 때는 그 결과의 수치만 쓴다. 권장 수면 시간·정상 범위·평균 같은 일반 기준
   수치는 retrieve_guide 로 찾은 문서에 있을 때만 쓰고, 없으면 해석하려고 덧붙이지 않는다.
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
9. 소아과 예약 요청에는 search_clinics 로 예약 가능한 소아과와 시간을 먼저 조회한다.
   예약은 search_clinics 결과에 있는 소아과·시간으로만 할 수 있다. 목록에 없는 병원은 예약할 수 없다고 안내하고
   예약 가능한 후보를 알려 준다. 이렇게 목록에 없는 병원이라 후보만 안내하는 경우에는, 병원·시간을 고르면 예약 전에
   승인을 한 번 더 받는다고 안내하고 '바로 예약해 드리겠다'처럼 말하지 않는다. 목록에 있는 소아과로 예약을 요청받으면 사용자에게 병원·시간을 되묻지 말고,
   사용자가 정한 병원·시간이 있으면 그 값으로, 없으면 조건에 맞는 가장 이른 시간으로 바로 book_clinic 을 호출한다.
   book_clinic 은 실행 전에 시스템이 후보 목록과 함께 사용자 승인을 받으므로, 사용자는 그 단계에서 확인하거나
   거절할 수 있다. 사용자가 거절하면 예약하지 않았다고 안내한다.
"""


class AgentAnswer(BaseModel):
    """에이전트의 최종 답변 형식 (구조화 출력)."""

    answer: str = Field(description="사용자에게 보여줄 최종 답변")
    sources: list[str] = Field(default_factory=list, description="근거로 쓴 가이드 문서 제목 목록. 없으면 빈 목록")


@lru_cache(maxsize=1)
def build_agent():
    """Bedrock 모델과 도메인 도구로 ReAct 에이전트 그래프를 한 번만 만든다.

    대체 모델이 설정되어 있으면 ModelFallbackMiddleware 로 1순위 모델 실패 시 순서대로 넘긴다.
    HumanInTheLoopMiddleware 로 예약 도구는 실행 직전에 멈추고 사용자 승인을 기다린다.
    """
    primary, *fallbacks = build_models()
    middleware = [ModelFallbackMiddleware(*fallbacks)] if fallbacks else []
    middleware.append(
        HumanInTheLoopMiddleware(
            interrupt_on={name: {"allowed_decisions": ["approve", "reject"]} for name in APPROVAL_TOOLS}
        )
    )
    return create_agent(
        model=primary,
        tools=TOOLS,
        system_prompt=SYSTEM_PROMPT,
        response_format=ToolStrategy(AgentAnswer),
        middleware=middleware,
        # 대화 상태에 저장되는 구조화 출력(AgentAnswer)을 체크포인터가 다시 읽을 수 있도록 허용한다
        checkpointer=InMemorySaver(
            serde=JsonPlusSerializer(allowed_msgpack_modules=[(AgentAnswer.__module__, "AgentAnswer")])
        ),
    )


def _truncate(text: str, limit: int = TRACE_OUTPUT_LIMIT) -> str:
    """trace 에 넣을 긴 문자열을 잘라 낸다."""
    return text if len(text) <= limit else text[:limit] + "...(생략)"


def _collect_turn(turn: list, all_messages: list) -> tuple[list[dict], list[dict], AIMessage | None]:
    """이번 턴에 새로 생긴 메시지에서 사용 모델·도구 호출 trace, contexts, 마지막 AI 메시지를 모은다.

    도구 인자는 대화 전체에서 찾는다. 승인 후 재개된 예약처럼 도구 호출은 이전 턴에 만들어지고
    실행 결과만 이번 턴에 생기는 경우가 있기 때문이다.
    """
    used_models = list(
        dict.fromkeys(
            f"{m.response_metadata.get('model_name', 'unknown')}@{m.response_metadata.get('region', '?')}"
            for m in turn
            if isinstance(m, AIMessage)
        )
    )
    tool_args = {
        call["id"]: call["args"]
        for m in all_messages
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


def _pending_actions(agent, config: dict) -> list[dict]:
    """이 대화(thread)에서 사용자 승인을 기다리며 멈춘 도구 호출 목록을 돌려준다. 없으면 빈 목록."""
    interrupts = getattr(agent.get_state(config), "interrupts", ()) or ()
    return [action for item in interrupts for action in item.value.get("action_requests", [])]


def _parse_decision(text: str) -> str | None:
    """승인 대기 중 사용자 답을 'approve' 또는 'reject' 로 해석한다. 판단할 수 없으면 None (다시 확인)."""
    text = text.strip()
    if any(word in text for word in REJECT_WORDS):
        return "reject"
    if APPROVE_PATTERN.match(text):
        return "approve"
    return None


def _approval_message(actions: list[dict], turn: list) -> str:
    """예약 후보(이번 턴의 search_clinics 결과)와 승인 질문을 담은 안내 문구를 만든다."""
    lines = []
    for m in turn:
        if isinstance(m, ToolMessage) and m.name == "search_clinics":
            found = json.loads(m.content)
            if "예약_가능_소아과" not in found:  # 날짜를 해석하지 못한 조회 결과는 건너뛴다
                continue
            candidates = [f"- {c['이름']}: {', '.join(s[11:] for s in c['예약_가능_시간'])}"
                          for c in found["예약_가능_소아과"]]
            if candidates:
                lines += [f"{found['조회_날짜']} 예약 가능한 소아과입니다.", *candidates, ""]
    for action in actions:
        clinic = find_clinic(action["args"].get("clinic_id", "")) or {"name": action["args"].get("clinic_id")}
        lines.append(f"{clinic['name']} {action['args'].get('slot')} 예약을 진행할까요?")
    lines.append("예약하시려면 '예', 취소하시려면 '아니오'라고 답해 주세요. 다른 병원·시간을 원하시면 '아니오' 후 다시 요청해 주세요.")
    return "\n".join(lines)


def _booking_message(turn: list) -> str | None:
    """승인 후 실행된 book_clinic 결과로 예약 안내 문구를 만든다. 예약 결과가 없으면 None.

    예약 결과는 정확해야 하므로 LLM 이 쓰지 않고 도구 결과로만 만든다
    (LLM 이 예약 전 조회 목록을 다시 보여 주며 방금 예약한 시간을 '예약 가능'으로 적던 문제 방지).
    """
    for m in turn:
        if not (isinstance(m, ToolMessage) and m.name == "book_clinic"):
            continue
        try:
            result = json.loads(m.content)
        except (TypeError, ValueError):  # 거절된 호출은 JSON 이 아닌 안내 문장이다
            continue
        if result.get("예약_결과") == "완료":
            return "\n".join([
                "예약을 완료했습니다.",
                f"- 소아과: {result['소아과']}",
                f"- 일시: {result['시간']}",
                f"- 예약번호: {result['예약번호']}",
                f"- 연락처: {result['연락처']} (프로필에 등록된 번호)",
                "",
                result["안내"],
            ])
        return f"예약하지 못했습니다. {result.get('이유', '')}".strip()
    return None


def _mask_trace_inputs(inputs: dict) -> dict:
    """LangSmith 에 보내는 run_query 입력에서 개인정보를 가린다."""
    return {**inputs, "question": mask_pii(str(inputs.get("question", "")))}


@traceable(name="run_query", run_type="chain", process_inputs=_mask_trace_inputs)
def run_query(question: str, thread_id: str | None = None) -> dict:
    """질문 하나를 처리해 answer, contexts, trace 세 키를 담은 dict 를 돌려준다.

    thread_id 가 같으면 이전 대화를 이어 가고, 없으면 새 대화로 처리한다.
    같은 thread 에 승인 대기 중인 예약이 있으면 이번 질문을 '예'/'아니오' 답으로 보고 재개한다.
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

    agent = build_agent()
    config = {"configurable": {"thread_id": thread_id or str(uuid.uuid4())}}
    previous_count = len(agent.get_state(config).values.get("messages", []))
    pending = _pending_actions(agent, config)
    decision = None
    if pending:
        decision = _parse_decision(masked)
        if decision is None:
            answer = "먼저 예약 승인 여부를 알려 주세요.\n" + _approval_message(pending, [])
            trace.append({"step": "hitl:book_clinic", "input": masked, "output": "답을 판단할 수 없어 다시 확인"})
            trace.append({"step": "answer", "input": masked, "output": answer})
            return {"answer": answer, "contexts": [], "trace": trace}
        trace.append({"step": "hitl:book_clinic", "input": masked, "output": "승인" if decision == "approve" else "거절"})
        payload = Command(resume={"decisions": [{"type": decision} for _ in pending]})
        turn_start = previous_count
    else:
        payload = {"messages": [{"role": "user", "content": masked}]}
        turn_start = previous_count + 1  # 방금 추가된 사용자 메시지 다음부터

    result = agent.invoke(payload, config)
    turn = result["messages"][turn_start:]
    tool_trace, contexts, last_ai = _collect_turn(turn, result["messages"])
    trace.extend(tool_trace)

    waiting = _pending_actions(agent, config)
    if waiting:
        for action in waiting:
            trace.append({
                "step": f"tool:{action['name']}",
                "input": json.dumps(action["args"], ensure_ascii=False),
                "output": "승인 대기 (아직 실행되지 않음)",
            })
        answer = _approval_message(waiting, turn)
        trace.append({"step": "hitl:book_clinic", "input": masked, "output": "승인 요청"})
        trace.append({"step": "answer", "input": masked, "output": answer})
        return {"answer": answer, "contexts": contexts, "trace": trace}

    booked = _booking_message(turn) if decision == "approve" else None
    if booked:
        answer = mask_pii(booked)
        trace.append({"step": "answer", "input": masked, "output": answer})
        return {"answer": answer, "contexts": contexts, "trace": trace}

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
