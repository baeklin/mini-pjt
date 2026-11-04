"""Bedrock LLM 생성: 1순위 모델과, 쓰로틀링 등으로 실패할 때 순서대로 쓸 대체 모델을 만든다."""

import os
from pathlib import Path

from botocore.config import Config
from dotenv import load_dotenv
from langchain_aws import ChatBedrockConverse
from langchain_core.runnables import Runnable

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

# 대체 모델이 있을 때는 같은 모델 재시도를 1회로 줄여 빨리 다음 모델로 넘어간다
FAST_FAIL_CONFIG = Config(retries={"max_attempts": 2, "mode": "standard"})


def model_ids() -> list[str]:
    """.env 의 BEDROCK_MODEL_ID(1순위)와 BEDROCK_FALLBACK_MODEL_IDS(쉼표 구분 대체 목록)를 순서대로 돌려준다."""
    fallbacks = os.getenv("BEDROCK_FALLBACK_MODEL_IDS", "")
    ids = [os.environ["BEDROCK_MODEL_ID"]] + [m.strip() for m in fallbacks.split(",") if m.strip()]
    return list(dict.fromkeys(ids))


def build_models() -> list[ChatBedrockConverse]:
    """1순위 모델과 대체 모델 목록을 temperature=0 으로 만든다. 대체 모델이 없으면 기본 재시도 설정을 쓴다."""
    ids = model_ids()
    extra = {"config": FAST_FAIL_CONFIG} if len(ids) > 1 else {}
    return [
        ChatBedrockConverse(model=model_id, region_name=os.environ["AWS_REGION"], temperature=0, **extra)
        for model_id in ids
    ]


def build_structured_model(schema: type) -> Runnable:
    """구조화 출력 모델을 만든다. 1순위 모델이 실패하면 대체 모델로 순서대로 다시 시도한다."""
    primary, *fallbacks = [model.with_structured_output(schema) for model in build_models()]
    return primary.with_fallbacks(fallbacks) if fallbacks else primary
