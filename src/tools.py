"""도메인 도구: 육아 가이드 검색, 아이 기록 조회, 예방접종 일정 계산."""

import calendar
import json
import os
from datetime import date
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.tools import tool

from retriever import search

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
load_dotenv(ROOT / ".env")

# 표준 예방접종 일정 (data/guides/01_vaccine_schedule.md 와 같은 내용)
# (백신, 차수, 시작 월령, 끝 월령)
VACCINE_SCHEDULE = [
    ("B형간염(HepB)", "1차", 0, 0),
    ("결핵(BCG)", "1회", 0, 1),
    ("B형간염(HepB)", "2차", 1, 1),
    ("DTaP", "1차", 2, 2),
    ("폴리오(IPV)", "1차", 2, 2),
    ("Hib", "1차", 2, 2),
    ("폐렴구균(PCV)", "1차", 2, 2),
    ("로타바이러스(RV)", "1차", 2, 2),
    ("DTaP", "2차", 4, 4),
    ("폴리오(IPV)", "2차", 4, 4),
    ("Hib", "2차", 4, 4),
    ("폐렴구균(PCV)", "2차", 4, 4),
    ("로타바이러스(RV)", "2차", 4, 4),
    ("DTaP", "3차", 6, 6),
    ("폴리오(IPV)", "3차", 6, 6),
    ("Hib", "3차", 6, 6),
    ("폐렴구균(PCV)", "3차", 6, 6),
    ("로타바이러스(RV)", "3차", 6, 6),
    ("B형간염(HepB)", "3차", 6, 6),
    ("Hib", "4차", 12, 15),
    ("폐렴구균(PCV)", "4차", 12, 15),
    ("MMR", "1차", 12, 15),
    ("수두(VAR)", "1회", 12, 15),
    ("A형간염(HepA)", "1~2차", 12, 23),
    ("일본뇌염", "불활성화 1~2차 또는 약독화 1차", 12, 23),
    ("DTaP", "4차", 15, 18),
]


def get_today() -> date:
    """기준 날짜를 돌려준다. .env 의 TODAY 가 있으면 그 날짜를, 없으면 실제 오늘 날짜를 쓴다."""
    value = os.getenv("TODAY", "").strip()
    return date.fromisoformat(value) if value else date.today()


def add_months(base: date, months: int) -> date:
    """날짜에 개월 수를 더한다. 해당 월에 같은 날이 없으면 그 달의 마지막 날로 맞춘다."""
    year, month_index = divmod(base.month - 1 + months, 12)
    year += base.year
    month = month_index + 1
    day = min(base.day, calendar.monthrange(year, month)[1])
    return date(year, month, day)


def get_age_months(birth_date: date, today: date | None = None) -> int:
    """생년월일과 기준 날짜로 만 개월 수(월령)를 계산한다."""
    today = today or get_today()
    months = (today.year - birth_date.year) * 12 + (today.month - birth_date.month)
    if today.day < birth_date.day:
        months -= 1
    return months


def load_profile() -> dict:
    """더미 아이 프로필(data/baby_profile.json)을 읽는다."""
    return json.loads((DATA_DIR / "baby_profile.json").read_text(encoding="utf-8"))


def _summarize_records(records: list[dict]) -> dict:
    """일별 육아 기록에서 수면·수유·체온의 평균과 범위를 계산한다."""
    sleep = [r["sleep_hours"] for r in records]
    feeds = [r["breastfeeding_count"] for r in records]
    temps = [r["max_temp_c"] for r in records]
    return {
        "기간": f"{records[0]['date']} ~ {records[-1]['date']}",
        "수면_평균_시간": round(sum(sleep) / len(sleep), 1),
        "수면_최소_최대_시간": [min(sleep), max(sleep)],
        "모유수유_하루_평균_횟수": round(sum(feeds) / len(feeds), 1),
        "최고체온_범위_c": [min(temps), max(temps)],
    }


@tool(response_format="content_and_artifact")
def retrieve_guide(query: str) -> tuple[str, list[dict]]:
    """육아 가이드 문서에서 질문과 관련된 내용을 검색한다.

    문서 주제: 표준 예방접종 일정, 영유아 건강검진, 월령별 발달, 이유식, 안전한 수면 환경,
    예방접종 후 이상반응, 영아 발열과 응급 신호, 수유.
    결과의 [문서 제목 · 섹션] 을 답변의 출처로 인용한다. 결과에 없는 내용은 문서에 없는 것이다.
    """
    docs = search(query)
    content = "\n\n".join(
        f"[{d.metadata['title']} · {d.metadata['section']}] (doc_id: {d.metadata['doc_id']})\n"
        f"{d.page_content}"
        for d in docs
    )
    artifact = [{"doc_id": d.metadata["doc_id"], "text": d.page_content} for d in docs]
    return content, artifact


@tool
def get_baby_records() -> str:
    """우리 아이의 프로필과 최근 7일 육아 기록을 조회한다.

    생년월일, 기준 날짜의 월령, 수유 방식, 이유식 시작일, 알레르기, 예방접종·건강검진 이력,
    일별 수유·수면·체온 기록과 그 요약을 돌려준다. 보호자 연락처는 돌려주지 않는다.
    """
    profile = load_profile()
    records = json.loads((DATA_DIR / "baby_records.json").read_text(encoding="utf-8"))["records"]
    today = get_today()
    birth_date = date.fromisoformat(profile["birth_date"])
    result = {
        "기준_날짜": today.isoformat(),
        "이름": profile["name"],
        "생년월일": profile["birth_date"],
        "월령_개월": get_age_months(birth_date, today),
        "수유_방식": profile["feeding_type"],
        "이유식_시작일": profile["weaning_started"],
        "알레르기": profile["allergies"],
        "예방접종_이력": profile["vaccinations_done"],
        "건강검진_이력": profile["checkups_done"],
        "최근_7일_기록": records,
        "최근_7일_요약": _summarize_records(records),
    }
    return json.dumps(result, ensure_ascii=False)


@tool
def calc_vaccine_schedule(birth_date: str) -> str:
    """생년월일(YYYY-MM-DD)로 표준 예방접종 일정에서 다음에 맞을 접종 항목과 접종 기간을 계산한다.

    생년월일은 get_baby_records 로 먼저 확인한다. 표준 일정 기준으로 계산하며,
    실제로 맞은 접종은 get_baby_records 의 예방접종_이력 으로 확인한다.
    """
    birth = date.fromisoformat(birth_date)
    today = get_today()
    age_months = get_age_months(birth, today)
    candidates = [item for item in VACCINE_SCHEDULE if item[3] >= age_months]
    if not candidates:
        return json.dumps(
            {"기준_날짜": today.isoformat(), "월령_개월": age_months,
             "안내": "0~24개월 표준 접종 일정에 남은 접종이 없습니다."},
            ensure_ascii=False,
        )
    # 지금 접종 기간이 열려 있는 항목이 있으면 모두 보여주고, 없으면 가장 먼저 시작하는 묶음을 보여준다
    due_now = [item for item in candidates if item[2] <= age_months]
    if due_now:
        selected = due_now
    else:
        next_start = min(item[2] for item in candidates)
        selected = [item for item in candidates if item[2] == next_start]
    next_vaccines = [
        {
            "백신": vaccine,
            "차수": dose,
            "권장_월령": f"{start}개월" if start == end else f"{start}~{end}개월",
            "접종_기간": f"{add_months(birth, start).isoformat()} ~ {add_months(birth, end).isoformat()}",
        }
        for vaccine, dose, start, end in selected
    ]
    result = {
        "기준_날짜": today.isoformat(),
        "생년월일": birth_date,
        "월령_개월": age_months,
        "다음_접종": next_vaccines,
        "근거": "질병관리청 표준 예방접종 일정 (가이드 문서: 표준 예방접종 일정)",
    }
    return json.dumps(result, ensure_ascii=False)
