"""Bedrock LLM 생성: 1순위 모델과, 쓰로틀링 등으로 실패할 때 순서대로 쓸 대체 모델을 만든다.

모델은 '모델ID' 또는 '모델ID@리전' 으로 적는다. 리전을 생략하면 AWS_REGION 을 쓴다.
"""

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


class RegionalChatBedrockConverse(ChatBedrockConverse):
    """응답 메타데이터에 호출한 리전(region)을 함께 남기는 ChatBedrockConverse."""

    def _generate(self, *args, **kwargs):
        """모델을 호출하고, 응답 메시지의 response_metadata 에 리전을 추가한다."""
        result = super()._generate(*args, **kwargs)
        for generation in result.generations:
            generation.message.response_metadata["region"] = self.region_name
        return result


def _parse_entry(entry: str) -> tuple[str, str]:
    """'모델ID@리전' 을 (모델ID, 리전) 으로 나눈다. 리전이 없으면 AWS_REGION 을 쓴다."""
    model_id, _, region = entry.strip().partition("@")
    return model_id.strip(), (region.strip() or os.environ["AWS_REGION"])


def model_ids() -> list[tuple[str, str]]:
    """BEDROCK_MODEL_ID(1순위)와 BEDROCK_FALLBACK_MODEL_IDS(쉼표 구분 대체 목록)를 (모델ID, 리전) 순서 목록으로 돌려준다."""
    entries = [os.environ["BEDROCK_MODEL_ID"]] + os.getenv("BEDROCK_FALLBACK_MODEL_IDS", "").split(",")
    return list(dict.fromkeys(_parse_entry(e) for e in entries if e.strip()))


def build_models() -> list[ChatBedrockConverse]:
    """1순위 모델과 대체 모델 목록을 각자의 리전과 temperature=0 으로 만든다. 대체 모델이 없으면 기본 재시도 설정을 쓴다."""
    entries = model_ids()
    extra = {"config": FAST_FAIL_CONFIG} if len(entries) > 1 else {}
    return [
        RegionalChatBedrockConverse(model=model_id, region_name=region, temperature=0, **extra)
        for model_id, region in entries
    ]


def build_structured_model(schema: type) -> Runnable:
    """구조화 출력 모델을 만든다. 1순위 모델이 실패하면 대체 모델로 순서대로 다시 시도한다."""
    primary, *fallbacks = [model.with_structured_output(schema) for model in build_models()]
    return primary.with_fallbacks(fallbacks) if fallbacks else primary
