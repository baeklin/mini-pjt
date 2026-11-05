"""입력·출력 가드레일: 응급 신호, 프롬프트 인젝션, 지원 범위 밖 질문, 개인정보 마스킹, 진단·약 용량 출력 검사."""

import re
from functools import lru_cache

from pydantic import BaseModel, Field

from llm import build_structured_model

GO_TO_ER = "지금 바로 119에 전화하거나 가까운 응급실로 가세요."

# (이름, 정규식, 안내 문구). 위에서부터 먼저 걸리는 것을 쓴다.
EMERGENCY_RULES = [
    ("경련", re.compile(r"경련|발작"), f"경련은 응급 상황입니다. {GO_TO_ER}"),
    (
        "호흡곤란",
        re.compile(r"호흡\s*곤란|무호흡|갈비뼈|숨(을|이)?\s*(못|안\s*쉬|가쁘|가빠|헐떡|차|쉬기\s*힘)"),
        f"숨쉬기 힘들어하거나 숨 쉴 때 갈비뼈 사이가 들어가는 것은 응급 신호입니다. {GO_TO_ER}",
    ),
    (
        "의식저하",
        # 증상 표현만 잡는다. '깨지 않고 통잠', '안 깨워도', '의식주', '수유 간격이 늘어져요' 같은 일상 표현은 제외
        re.compile(
            r"의식(이|을)?\s*(없|흐려|흐리|잃|저하|떨어|희미)|정신을\s*잃|반응(이|을)?\s*없"
            r"|(깨워도|흔들어도|불러도)\s*(잘\s*)?(안\s*(깨|일어나)|깨지\s*않|일어나지\s*않|반응)"
            r"|(아기|아이|애)(가|는)?\s*(안\s*깨(?![고워우])|깨지\s*않(?!고)|깨어나지\s*않)"
            r"|잘\s*깨지\s*않(?!고)|안\s*깨어나|깨어나지\s*않"
            r"|축\s*(처|늘어)|(아기|아이|애)(가|는)?\s*(축\s*)?늘어(져|졌|지)"
            r"|입술.{0,4}(보라|파랗|파래)"
        ),
        f"의식이 흐려지거나 처지는 것은 응급 신호입니다. {GO_TO_ER}",
    ),
]
INFANT_FEVER_MESSAGE = (
    "생후 3개월 미만 아기의 38℃ 이상 발열은 응급 신호입니다. "
    "해열제를 먹이기 전에 지금 바로 응급실로 가거나 119에 전화하세요."
)
INJECTION_PATTERNS = [
    re.compile(r"(이전|앞의|위의|기존)\s*(지시|명령|규칙|설정).{0,8}(무시|잊)"),
    re.compile(r"시스템\s*프롬프트|system\s*prompt", re.IGNORECASE),
    re.compile(r"ignore\s+(all\s+)?(previous|prior|above)", re.IGNORECASE),
    re.compile(r"개발자\s*모드|developer\s*mode|\bDAN\b", re.IGNORECASE),
]
INJECTION_MESSAGE = (
    "해당 요청에는 응할 수 없습니다. "
    "아기의 접종·발달·이유식·수면 등 육아 관련 질문을 도와드릴 수 있습니다."
)
OUT_OF_RANGE_MESSAGE = (
    "AI 육아도우미는 생후 0~24개월 아기를 대상으로 하여 해당 월령의 질문에는 답변드리기 어렵습니다. "
    "소아청소년과 상담을 권해 드립니다."
)
OUTPUT_REFUSAL_MESSAGE = (
    "약 이름이나 용량, 질병 진단은 아이의 나이와 체중, 상태에 따라 달라 안내해 드릴 수 없습니다. "
    "의사나 약사와 상의해 주세요."
)

TEMPERATURE_PATTERN = re.compile(r"(?<![\d.])(\d{2}(?:\.\d)?)\s*(?:도|℃|°C|°)")
# '열이 37.9에서 38.4로'처럼 단위 없이 쓴 체온: '열·체온'이 나온 질문의 두 자리 숫자를 모두 후보로 본다
FEVER_CONTEXT = re.compile(r"열(?!흘)|체온")
UNITLESS_NUMBER_PATTERN = re.compile(
    r"(?<![\d.])(\d{2}(?:\.\d)?)(?![\d.])(?!\s*(분|시간|일|개월|달|주|살|세|회|번|ml|mL|g|kg|cm|%|kcal))"
)
# 아기 체온이 아닌 온도: '목욕물 38도', '방 온도 24도' 등
NON_BODY_TEMPERATURE_BEFORE = re.compile(r"(물|방|실내|온도|목욕|분유|우유|이유식)\D{0,4}$")
# 'N살/N세'가 아이 나이가 아닌 경우: '엄마 나이 35세', '형이 4살' 등 ('3세끼'는 패턴 쪽에서 제외)
NON_CHILD_AGE_BEFORE = re.compile(r"(엄마|아빠|부모|할머니|할아버지|어른|남편|아내|형|누나|언니|오빠)\D{0,6}$")
KOREAN_NUMBERS = {"한": 1, "두": 2, "세": 3, "네": 4, "다섯": 5, "여섯": 6, "일곱": 7, "여덟": 8, "아홉": 9, "열": 10}
PHONE_PATTERN = re.compile(r"(01[016789])[-.\s]?(\d{3,4})[-.\s]?(\d{4})")
RRN_PATTERN = re.compile(r"\d{6}[-\s]?[1-4]\d{6}")


def extract_age_months(question: str) -> float | None:
    """질문에 적힌 아기 월령(개월)을 찾는다.

    '신생아', '생후 N일'·'N일 된', '생후 N주'·'N주 된', 'N개월'·'N달', '한·두·세 달' 을 인식한다.
    """
    if "신생아" in question:
        return 0
    if match := re.search(r"생후\s*(\d+)\s*일|(\d+)\s*일\s*(된|째|차)", question):
        return int(match.group(1) or match.group(2)) / 30
    if match := re.search(r"생후\s*(\d+)\s*주|(\d+)\s*주\s*(된|째|차)", question):
        return int(match.group(1) or match.group(2)) / 4.345
    if match := re.search(r"(\d+)\s*(개월|달)", question):
        return int(match.group(1))
    if match := re.search(r"(한|두|세|네|다섯|여섯|일곱|여덟|아홉|열)\s*달", question):
        return KOREAN_NUMBERS[match.group(1)]
    return None


def _body_temperatures(question: str) -> list[float]:
    """질문에서 아기 체온으로 볼 수 있는 숫자(34~43)를 모두 찾는다.

    단위('도·℃')가 붙은 숫자와, '열·체온'이 나온 질문의 단위 없는 숫자를 본다.
    '40분' 같은 다른 단위의 숫자와 '목욕물·방 온도' 처럼 아기 체온이 아닌 온도는 뺀다.
    """
    matches = list(TEMPERATURE_PATTERN.finditer(question))
    if FEVER_CONTEXT.search(question):
        matches += list(UNITLESS_NUMBER_PATTERN.finditer(question))
    values = []
    for match in matches:
        if NON_BODY_TEMPERATURE_BEFORE.search(question[: match.start()]):
            continue
        value = float(match.group(1))
        if 34.0 <= value <= 43.0:
            values.append(value)
    return values


def check_emergency(question: str, profile_age_months: int) -> tuple[str, str] | None:
    """응급 신호를 규칙으로 찾는다. 걸리면 (규칙 이름, 안내 문구)를, 아니면 None 을 돌려준다.

    3개월 미만 발열은 질문에 적힌 월령을 먼저 쓰고, 없으면 프로필 월령으로 판단한다.
    """
    for name, pattern, message in EMERGENCY_RULES:
        if pattern.search(question):
            return name, message
    temperatures = _body_temperatures(question)
    age = extract_age_months(question)
    age = profile_age_months if age is None else age
    if any(t >= 38.0 for t in temperatures) and age < 3:
        return "3개월미만발열", INFANT_FEVER_MESSAGE
    return None


def detect_injection(question: str) -> bool:
    """시스템 프롬프트 노출·지시 무시 같은 프롬프트 인젝션 시도인지 판단한다."""
    return any(pattern.search(question) for pattern in INJECTION_PATTERNS)


def check_out_of_range(question: str) -> bool:
    """질문이 0~24개월 지원 범위를 벗어난 아이에 관한 것인지 판단한다 (25개월 이상, 3살 이상).

    보호자·형제 나이('엄마 나이 35세', '형이 4살')와 '3세끼' 같은 표현은 아이 나이로 보지 않는다.
    """
    if match := re.search(r"(\d+)\s*(개월|달)", question):
        if int(match.group(1)) > 24:
            return True
    ages = [(m.start(), int(m.group(1))) for m in re.finditer(r"(\d+)\s*(살|세)(?![끼트대])", question)]
    ages += [
        (m.start(), KOREAN_NUMBERS[m.group(1)])
        for m in re.finditer(r"(?<![가-힣])(세|네|다섯|여섯|일곱|여덟|아홉|열)\s*살", question)
    ]
    for start, age in ages:
        if NON_CHILD_AGE_BEFORE.search(question[:start]):
            continue
        if age >= 3:
            return True
    return False


def check_input(question: str, profile_age_months: int) -> tuple[str, str] | None:
    """입력 가드레일을 우선순위(응급 → 인젝션 → 범위 밖)대로 실행한다.

    막아야 하면 (가드레일 이름, 고정 안내 문구)를, 통과하면 None 을 돌려준다.
    """
    if emergency := check_emergency(question, profile_age_months):
        return f"emergency:{emergency[0]}", emergency[1]
    if detect_injection(question):
        return "injection", INJECTION_MESSAGE
    if check_out_of_range(question):
        return "out_of_range", OUT_OF_RANGE_MESSAGE
    return None


def mask_pii(text: str) -> str:
    """휴대폰 번호는 가운데 자리를, 주민등록번호는 전체를 * 로 가린다.

    주민등록번호를 먼저 가린다. 2001년생 등은 앞자리가 '01x'라 휴대폰 번호로 먼저 잡히면 뒷자리가 남기 때문이다.
    """
    text = RRN_PATTERN.sub("******-*******", text)
    return PHONE_PATTERN.sub(lambda m: f"{m.group(1)}-****-{m.group(3)}", text)


class OutputVerdict(BaseModel):
    """출력 가드레일 판정 결과."""

    violates: bool = Field(description="약 이름·용량 안내 또는 질병 진단이 들어 있으면 true")
    reason: str = Field(description="판정 이유 한 문장")


OUTPUT_JUDGE_PROMPT = """너는 육아 상담 답변의 안전 검사관이다. 아래 답변이 다음 중 하나라도 하는지 판정해라.
1. 특정 약 이름(예: 해열제 성분명·제품명)을 먹이라고 권하거나, 약 용량(ml, mg, 몇 알 등)을 안내한다
2. 아이의 질병이나 장애를 진단한다 (예: "~병입니다", "~장애로 보입니다", "~일 가능성이 높습니다")

다음은 위반이 아니다:
- 약 용량·진단을 안내할 수 없다고 거절하거나 의사·약사·소아과 상담을 권하는 문장
- 이유식 양, 수유량(mL), 체온, 방 온도 같은 일반 육아 정보
- "상담이 필요한 신호" 처럼 진료를 권하는 안내

답변:
{answer}"""


@lru_cache(maxsize=1)
def _judge_model():
    """출력 판정에 쓸 구조화 출력 LLM(대체 모델 포함)을 한 번만 만든다."""
    return build_structured_model(OutputVerdict)


def check_output(answer: str) -> tuple[str, OutputVerdict]:
    """최종 답변에 약 이름·용량이나 질병 진단이 들어 있는지 LLM 으로 판정한다.

    위반이면 거절 문구로 바꾼 답변을, 아니면 원래 답변을 판정 결과와 함께 돌려준다.
    """
    verdict = _judge_model().invoke(OUTPUT_JUDGE_PROMPT.format(answer=answer))
    return (OUTPUT_REFUSAL_MESSAGE if verdict.violates else answer), verdict
