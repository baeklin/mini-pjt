"""인-아웃 세트 채점 스크립트: test_queries.csv 를 에이전트로 실행해 통과율과 RAGAS 지표를 round{N}_report.md 로 저장한다.

실행: python evaluation/run_eval.py --round 1
"""

import argparse
import csv
import math
import os
import sys
import time
import types
import warnings
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = ROOT / "evaluation"
sys.path.insert(0, str(ROOT / "src"))

# ragas 0.4.3 은 langchain-community 0.4 에서 없어진 vertexai 모듈을 import 한다.
# Vertex AI 모델 여부 확인에만 쓰이므로, Bedrock 만 쓰는 이 프로젝트에서는 빈 클래스로 대신한다.
_vertexai_stub = types.ModuleType("langchain_community.chat_models.vertexai")
_vertexai_stub.ChatVertexAI = type("ChatVertexAI", (), {})
sys.modules.setdefault(_vertexai_stub.__name__, _vertexai_stub)

from agent import run_query  # noqa: E402
from botocore.config import Config  # noqa: E402
from llm import RegionalChatBedrockConverse, build_models, model_ids  # noqa: E402
from tools import get_today  # noqa: E402

# 응급 문항: 답변에 이 단어 중 하나가 있어야 통과 (SERVICE.md 5번)
EMERGENCY_IDS = {"11", "12", "18"}
EMERGENCY_WORDS = ("응급실", "119")


def split_items(value: str) -> list[str]:
    """세미콜론으로 구분된 CSV 필드를 항목 목록으로 나눈다."""
    return [item.strip() for item in value.split(";") if item.strip()]


def load_cases() -> list[dict]:
    """test_queries.csv 와 ragas_references.csv 를 id 로 합쳐 문항 목록을 만든다."""
    with open(EVAL_DIR / "test_queries.csv", encoding="utf-8") as f:
        cases = list(csv.DictReader(f))
    with open(EVAL_DIR / "ragas_references.csv", encoding="utf-8") as f:
        references = {row["id"]: row["reference"] for row in csv.DictReader(f)}
    for case in cases:
        case["reference"] = references.get(case["id"], "")
    return cases


def grade(case: dict, response: dict) -> dict:
    """한 문항의 응답을 자동 판정한다: ① 기대 도구 호출 ② forbidden 문자열 없음 ③ 응급 문항은 응급실·119 안내."""
    answer = response["answer"]
    called = [s["step"][5:] for s in response["trace"] if s["step"].startswith("tool:")]
    expected = split_items(case["expected_tools"])
    missing_tools = [t for t in expected if t not in called]
    forbidden_hits = [item for item in split_items(case["forbidden"]) if item in answer]
    emergency_ok = None
    if case["id"] in EMERGENCY_IDS:
        emergency_ok = any(word in answer for word in EMERGENCY_WORDS)
    auto_pass = not missing_tools and not forbidden_hits and emergency_ok is not False
    model = next((s["output"] for s in response["trace"] if s["step"] == "model"), "-")
    return {
        "called": called,
        "missing_tools": missing_tools,
        "forbidden_hits": forbidden_hits,
        "emergency_ok": emergency_ok,
        "auto_pass": auto_pass,
        "model": model,
    }


def run_cases(cases: list[dict]) -> list[dict]:
    """문항마다 새 대화로 에이전트를 실행하고 자동 판정 결과를 붙인다."""
    results = []
    for case in cases:
        started = time.time()
        try:
            response = run_query(case["input"])
            result = {**case, "response": response, **grade(case, response)}
        except Exception as error:  # 한 문항 실패가 전체 평가를 멈추지 않게 한다
            response = {"answer": f"(오류) {type(error).__name__}: {error}", "contexts": [], "trace": []}
            result = {**case, "response": response, "called": [], "missing_tools": [], "forbidden_hits": [],
                      "emergency_ok": None, "auto_pass": False, "model": "-"}
        result["seconds"] = round(time.time() - started, 1)
        results.append(result)
        print(f"[{case['id']:>2}] {'PASS' if result['auto_pass'] else 'FAIL'} ({result['seconds']}초) {case['input'][:30]}")
    return results


def run_ragas(results: list[dict]) -> tuple[dict, dict]:
    """retrieve_guide 를 쓰는 문항을 RAGAS 4지표로 채점해 (평균, 문항별 점수) 를 돌려준다.

    평균은 SERVICE.md 5번 계산 범위(negative 제외)만으로 낸다. negative 문항 점수는 참고용으로 문항별 점수에만 남긴다.
    """
    from langchain_aws import BedrockEmbeddings
    from ragas import EvaluationDataset, evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness
    from ragas.run_config import RunConfig

    targets = [
        r for r in results
        if "retrieve_guide" in split_items(r["expected_tools"]) and r["response"]["contexts"]
    ]
    if not targets:
        return {}, {}
    dataset = EvaluationDataset.from_list([
        {
            "user_input": r["input"],
            "response": r["response"]["answer"],
            "retrieved_contexts": [c["text"] for c in r["response"]["contexts"]],
            "reference": r["reference"],
        }
        for r in targets
    ])
    embeddings = BedrockEmbeddings(
        model_id=os.environ["BEDROCK_EMBED_MODEL_ID"], region_name=os.environ["AWS_REGION"]
    )
    metrics = [faithfulness, answer_relevancy, context_recall, context_precision]
    judge = _pick_working_model()
    scores = evaluate(
        dataset,
        metrics=metrics,
        llm=LangchainLLMWrapper(judge),
        embeddings=LangchainEmbeddingsWrapper(embeddings),
        # 동시 호출을 줄이고(기본 16) 재시도 대기를 늘려 쓰로틀링을 피한다
        run_config=RunConfig(max_workers=2, timeout=600, max_wait=120),
        show_progress=False,
    ).to_pandas()
    names = [m.name for m in metrics]
    per_case = {
        r["id"]: {name: round(float(scores.iloc[i][name]), 3) for name in names}
        for i, r in enumerate(targets)
    }
    in_scope = [r["id"] for r in targets if r["category"] != "negative"]
    averages = {name: _mean([per_case[i][name] for i in in_scope]) for name in names}
    averages_all = {name: _mean([per_case[r["id"]][name] for r in targets]) for name in names}
    failed = sum(math.isnan(v) for case in per_case.values() for v in case.values())
    judge_name = f"{judge.model_id}@{judge.region_name}"
    return {"scope": averages, "all": averages_all, "failed": failed, "judge": judge_name}, per_case


def _pick_working_model():
    """대체 모델 목록에서 지금 호출되는 첫 모델을 RAGAS 채점용으로 고른다.

    RAGAS 는 모델 폴백을 지원하지 않아, 1순위 모델이 쓰로틀링 중이면 점수가 모두 계산 실패(nan)가 된다.
    또 llm.py 는 대체 모델이 있으면 재시도를 1회로 줄이므로, 채점용으로는 적응형 재시도 8회로 다시 만든다.
    """
    for model in build_models():
        try:
            model.invoke("ping")
            return RegionalChatBedrockConverse(
                model=model.model_id,
                region_name=model.region_name,
                temperature=0,
                config=Config(retries={"max_attempts": 8, "mode": "adaptive"}),
            )
        except Exception as error:  # 쓰로틀링·권한 오류면 다음 모델을 시도한다
            print(f"RAGAS 채점 모델 건너뜀: {model.model_id}@{model.region_name} ({type(error).__name__})")
    raise RuntimeError("RAGAS 채점에 쓸 수 있는 모델이 없습니다.")


def _mean(values: list[float]) -> float:
    """계산에 실패한 값(nan)을 뺀 평균을 소수 셋째 자리로 반올림한다. 남는 값이 없으면 nan 을 돌려준다."""
    valid = [v for v in values if not math.isnan(v)]
    return round(sum(valid) / len(valid), 3) if valid else float("nan")


def _rate(passed: int, total: int) -> str:
    """통과 수와 비율을 'n / m (xx%)' 형식으로 만든다."""
    return f"{passed} / {total} ({passed / total:.0%})" if total else "-"


def build_report(round_no: int, results: list[dict], averages: dict, per_case: dict) -> str:
    """채점 결과를 마크다운 리포트 문자열로 만든다."""
    categories = ["positive", "negative", "edge", "guardrail"]
    total_pass = sum(r["auto_pass"] for r in results)
    emergency = [r for r in results if r["id"] in EMERGENCY_IDS]
    lines = [
        f"# {round_no}차 자체 평가 리포트",
        "",
        f"- 실행 시각: {datetime.now():%Y-%m-%d %H:%M}",
        f"- 기준 날짜(TODAY): {get_today().isoformat()}",
        f"- 1순위 모델: {'@'.join(model_ids()[0])} (대체 모델 {len(model_ids()) - 1}개)",
        "- 자동 판정 = ① expected_tools 모두 호출 ② forbidden 문자열 없음 ③ 응급 문항(11·12·18)은 '응급실' 또는 '119' 포함",
        "- expected_traits 와 서술형 forbidden 은 아래 문항별 상세에서 사람이 확인한다 (SERVICE.md 5번)",
        "",
        "## 요약",
        "",
        "| 구분 | 자동 판정 통과 |",
        "|---|---|",
    ]
    for category in categories:
        group = [r for r in results if r["category"] == category]
        lines.append(f"| {category} | {_rate(sum(r['auto_pass'] for r in group), len(group))} |")
    lines += [
        f"| **전체** | **{_rate(total_pass, len(results))}** |",
        f"| 응급 문항 응급실·119 안내 | {_rate(sum(bool(r['emergency_ok']) for r in emergency), len(emergency))} |",
        "",
        "## RAGAS",
        "",
        "- 평가 범위: 문서를 근거로 답해야 하는 positive·edge 문항 (SERVICE.md 5번)",
        "- 참고: negative 문항을 포함한 retrieve_guide 사용 문항 전체 (negative 는 올바른 답이 '정보 없음'이라 faithfulness 로 잴 수 없음)",
        "",
    ]
    if averages:
        lines += ["| 지표 | 평가 범위 평균 | 참고: negative 포함 평균 |", "|---|---|---|"]
        lines += [f"| {k} | {v} | {averages['all'][k]} |" for k, v in averages["scope"].items()]
        lines += ["", f"- 채점 모델: {averages['judge']}", f"- 계산 실패(nan) 값: {averages['failed']}건 (평균에서 제외)"]
    else:
        lines.append("계산하지 않음")
    lines += [
        "",
        "## 문항별 결과",
        "",
        "| id | 구분 | 자동 | 호출 도구 | 누락 도구 | forbidden 일치 | 응답 모델 | 시간(초) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        lines.append(
            f"| {r['id']} | {r['category']} | {'PASS' if r['auto_pass'] else 'FAIL'} | "
            f"{', '.join(r['called']) or '-'} | {', '.join(r['missing_tools']) or '-'} | "
            f"{', '.join(r['forbidden_hits']) or '-'} | {r['model']} | {r['seconds']} |"
        )
    lines += ["", "## 문항별 상세 (사람 확인)", ""]
    for r in results:
        lines += [
            f"### {r['id']}. [{r['category']}] {r['input']}",
            "",
            f"- 자동 판정: {'PASS' if r['auto_pass'] else 'FAIL'}",
        ]
        if r["id"] in per_case:
            lines.append(f"- RAGAS: {per_case[r['id']]}")
        lines += [
            f"- contexts: {', '.join(c['doc_id'] for c in r['response']['contexts']) or '-'}",
            "",
            "**답변**",
            "",
            *[f"> {line}" for line in r["response"]["answer"].splitlines()],
            "",
            "**expected_traits (사람 확인)**",
            "",
            *[f"- [ ] {trait}" for trait in split_items(r["expected_traits"])],
            "",
            f"**forbidden**: {r['forbidden'] or '-'}",
            "",
            "- [ ] 서술형 forbidden 항목이 답변에 없음 (사람 확인)",
            "- 최종 판정: ",
            "",
        ]
    return "\n".join(lines)


def main() -> None:
    """명령행 인자를 읽어 평가를 실행하고 리포트를 저장한다."""
    parser = argparse.ArgumentParser(description="인-아웃 세트 자체 평가")
    parser.add_argument("--round", type=int, required=True, help="평가 회차 (1, 2, ...)")
    parser.add_argument("--ids", default="", help="일부 문항만 실행할 때 쉼표로 구분한 id (리포트는 저장하지 않음)")
    parser.add_argument("--skip-ragas", action="store_true", help="RAGAS 계산을 건너뛴다")
    parser.add_argument("--force", action="store_true", help="같은 회차 리포트가 있어도 덮어쓴다")
    args = parser.parse_args()

    sys.stdout.reconfigure(encoding="utf-8")
    warnings.filterwarnings("ignore")
    report_path = EVAL_DIR / f"round{args.round}_report.md"
    if not args.ids and report_path.exists() and not args.force:
        sys.exit(f"{report_path.name} 이 이미 있습니다. 사람 확인 내용이 지워질 수 있으니 덮어쓰려면 --force 를 붙이세요.")

    cases = load_cases()
    if args.ids:
        wanted = {i.strip() for i in args.ids.split(",")}
        cases = [c for c in cases if c["id"] in wanted]
    results = run_cases(cases)
    averages, per_case = ({}, {}) if args.skip_ragas else run_ragas(results)

    total_pass = sum(r["auto_pass"] for r in results)
    print(f"\n자동 판정 통과: {_rate(total_pass, len(results))}")
    if averages:
        print(f"RAGAS 평가 범위: {averages['scope']}")
        print(f"RAGAS 참고(negative 포함): {averages['all']}")
        print(f"RAGAS 채점 모델: {averages['judge']} | 계산 실패: {averages['failed']}건")
    if args.ids:
        print("(일부 문항 실행이라 리포트는 저장하지 않음)")
        return
    report_path.write_text(build_report(args.round, results, averages, per_case), encoding="utf-8")
    print(f"리포트 저장: {report_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
