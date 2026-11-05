"""도메인 도구: 육아 가이드 검색, 아이 기록 조회, 예방접종·건강검진 일정 계산, 소아과 조회·예약."""

import calendar
import json
import os
from datetime import date, timedelta
from itertools import count
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


# 영유아 건강검진 일정 (data/guides/02_health_checkup.md 와 같은 내용, 생후 0~24개월 해당 차수)
# (차수, 단위, 시작, 끝, 권장 시기 표기, 구강검진 시기, 발달선별검사 포함 여부)
CHECKUP_SCHEDULE = [
    ("1차", "days", 14, 35, "생후 14~35일", None, False),
    ("2차", "months", 4, 6, "생후 4~6개월", None, False),
    ("3차", "months", 9, 12, "생후 9~12개월", None, True),
    ("4차", "months", 18, 24, "생후 18~24개월", "생후 18~29개월", True),
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


@tool
def calc_checkup_schedule(birth_date: str) -> str:
    """생년월일(YYYY-MM-DD)로 영유아 건강검진(생후 0~24개월 차수) 중 지금 받을 수 있거나 다음에 받을 차수와 시작일을 계산한다.

    생년월일은 get_baby_records 로 먼저 확인한다. 정확한 검진 마감일은 계산하지 않으며,
    국민건강보험공단 '영유아 검진일자 조회'로 확인하도록 안내한다.
    """
    birth = date.fromisoformat(birth_date)
    today = get_today()
    age = {"days": (today - birth).days, "months": get_age_months(birth, today)}

    def start_date(unit: str, start: int) -> date:
        """검진 차수의 시작일을 계산한다."""
        return birth + timedelta(days=start) if unit == "days" else add_months(birth, start)

    open_now = [c for c in CHECKUP_SCHEDULE if c[2] <= age[c[1]] <= c[3]]
    upcoming = [c for c in CHECKUP_SCHEDULE if age[c[1]] < c[2]]
    selected, status = (open_now, "지금 받을 수 있음") if open_now else (upcoming[:1], "예정")
    if not selected:
        return json.dumps(
            {"기준_날짜": today.isoformat(), "월령_개월": age["months"],
             "안내": "생후 0~24개월 건강검진 차수가 모두 지났습니다. 다음은 5차(생후 30~36개월)입니다."},
            ensure_ascii=False,
        )
    checkups = [
        {
            "차수": round_name,
            "권장_시기": period,
            "시작일": start_date(unit, start).isoformat(),
            "상태": status,
            "구강검진_시기": oral or "없음",
            "발달선별검사_포함": development,
        }
        for round_name, unit, start, _end, period, oral, development in selected
    ]
    result = {
        "기준_날짜": today.isoformat(),
        "생년월일": birth_date,
        "월령_개월": age["months"],
        "다음_검진": checkups,
        "마감일_안내": "정확한 검진 기간(마감일)은 국민건강보험공단 '영유아 검진일자 조회'에서 확인",
        "근거": "국민건강보험공단 영유아 건강검진 안내 (가이드 문서: 영유아 건강검진 시기)",
    }
    return json.dumps(result, ensure_ascii=False)


# 시연용 예약 저장소 (메모리). 서버를 다시 시작하면 사라진다.
_BOOKINGS: dict[str, dict] = {}
_BOOKED_SLOTS: set[tuple[str, str]] = set()
_BOOKING_NUMBERS = count(1)
TIME_RANGES = {"오전": (0, 12), "오후": (12, 18), "저녁": (18, 24), "전체": (0, 24)}


def load_clinics() -> list[dict]:
    """더미 소아과 목록(data/clinics.json)을 읽는다."""
    return json.loads((DATA_DIR / "clinics.json").read_text(encoding="utf-8"))["clinics"]


def find_clinic(clinic_id: str) -> dict | None:
    """clinic_id 로 소아과 정보를 찾는다. 없으면 None 을 돌려준다."""
    return next((c for c in load_clinics() if c["clinic_id"] == clinic_id), None)


WEEKDAYS = "월화수목금토일"


def format_day(day: date) -> str:
    """날짜를 'YYYY-MM-DD(요일)' 형식으로 만든다. LLM 이 요일을 직접 계산하지 않도록 도구 결과에 넣는다."""
    return f"{day.isoformat()}({WEEKDAYS[day.weekday()]})"


def _resolve_date(value: str) -> date:
    """'오늘', '내일', '모레' 또는 YYYY-MM-DD 를 기준 날짜 기준의 날짜로 바꾼다."""
    offsets = {"오늘": 0, "내일": 1, "모레": 2}
    value = value.strip()
    return get_today() + timedelta(days=offsets[value]) if value in offsets else date.fromisoformat(value)


@tool
def search_clinics(day: str = "내일", time_of_day: str = "전체") -> str:
    """예약 가능한 소아과와 시간을 조회한다. 예약 가능한 소아과는 이 목록에 있는 곳뿐이다.

    day 는 '오늘', '내일', '모레' 또는 YYYY-MM-DD, time_of_day 는 '오전', '오후', '저녁', '전체' 중 하나다.
    결과의 clinic_id 와 시간(YYYY-MM-DD HH:MM)을 book_clinic 에 그대로 넘긴다.
    """
    try:
        target = _resolve_date(day)
    except ValueError:
        return json.dumps(
            {"안내": f"'{day}' 날짜를 이해하지 못했습니다. '오늘', '내일', '모레' 또는 YYYY-MM-DD 형식으로 알려 주세요."},
            ensure_ascii=False,
        )
    start_hour, end_hour = TIME_RANGES.get(time_of_day, TIME_RANGES["전체"])
    clinics = []
    for clinic in load_clinics():
        slots = [
            slot for slot in clinic["available_slots"]
            if slot.startswith(target.isoformat())
            and start_hour <= int(slot[11:13]) < end_hour
            and (clinic["clinic_id"], slot) not in _BOOKED_SLOTS
        ]
        if slots:
            clinics.append({
                "clinic_id": clinic["clinic_id"],
                "이름": clinic["name"],
                "지역": clinic["area"],
                "진료시간": clinic["open_hours"],
                "예약_가능_시간": slots,
            })
    result = {
        "기준_날짜": format_day(get_today()),
        "조회_날짜": format_day(target),
        "시간대": time_of_day,
        "예약_가능_소아과": clinics,
        "안내": "이 목록에 없는 병원은 예약할 수 없다",
    }
    return json.dumps(result, ensure_ascii=False)


@tool
def book_clinic(clinic_id: str, slot: str) -> str:
    """소아과를 예약한다. 사용자 승인을 받은 뒤에만 실행된다.

    clinic_id 와 slot(YYYY-MM-DD HH:MM)은 search_clinics 결과에 있는 값만 쓸 수 있다.
    보호자 연락처는 프로필에 등록된 번호를 쓴다.
    """
    clinic = find_clinic(clinic_id)
    if clinic is None or slot not in clinic["available_slots"] or (clinic_id, slot) in _BOOKED_SLOTS:
        return json.dumps(
            {"예약_결과": "실패", "이유": "예약 가능한 소아과·시간 목록에 없습니다. search_clinics 로 다시 조회하세요."},
            ensure_ascii=False,
        )
    booking_id = f"BK-{next(_BOOKING_NUMBERS):04d}"
    phone = load_profile()["guardian"]["phone"]
    _BOOKED_SLOTS.add((clinic_id, slot))
    _BOOKINGS[booking_id] = {"clinic_id": clinic_id, "slot": slot}
    result = {
        "예약_결과": "완료",
        "예약번호": booking_id,
        "소아과": clinic["name"],
        "시간": f"{format_day(date.fromisoformat(slot[:10]))} {slot[11:]}",
        "연락처": f"{phone[:3]}-****-{phone[-4:]}",
        "안내": "시연용 예약입니다. 실제 병원에 예약되지 않습니다.",
    }
    return json.dumps(result, ensure_ascii=False)
