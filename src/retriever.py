"""RAG 파이프라인: 육아 가이드 문서를 섹션 단위로 나눠 BM25 + 벡터 하이브리드 검색을 제공한다."""

import os
import re
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from langchain_aws import BedrockEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter

ROOT = Path(__file__).resolve().parent.parent
GUIDE_DIR = ROOT / "data" / "guides"
load_dotenv(ROOT / ".env")

# 각 검색기에서 후보를 몇 개씩 가져와 합칠지
CANDIDATES_PER_RETRIEVER = 8
# RRF 상수 (일반적으로 쓰는 값)
RRF_K = 60


def tokenize(text: str) -> list[str]:
    """한국어 BM25 용 토큰화. 조사가 붙은 단어도 맞도록 공백 단어와 글자 2-gram 을 함께 쓴다."""
    words = re.findall(r"[가-힣A-Za-z0-9.]+", text.lower())
    bigrams = [word[i : i + 2] for word in words for i in range(len(word) - 1)]
    return words + bigrams


def load_guides() -> list[Document]:
    """data/guides 의 마크다운 문서를 읽어 '##' 섹션 단위 Document 목록으로 만든다.

    metadata 에는 doc_id, title, section, source(첫 번째 출처 URL) 를 담는다.
    문서 맨 위의 메타 정보 블록(doc_id·출처 줄)은 검색 대상에서 뺀다.
    """
    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "title"), ("##", "section")]
    )
    documents = []
    for path in sorted(GUIDE_DIR.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        doc_id = re.search(r"^- doc_id:\s*(\S+)", text, re.MULTILINE).group(1)
        source = re.search(r"^- 출처:.*?(https?://\S+)", text, re.MULTILINE)
        for chunk in splitter.split_text(text):
            section = chunk.metadata.get("section")
            if not section:
                continue
            title = chunk.metadata.get("title", "")
            documents.append(
                Document(
                    page_content=f"{title} - {section}\n{chunk.page_content}",
                    metadata={
                        "doc_id": doc_id,
                        "title": title,
                        "section": section,
                        "source": source.group(1) if source else "",
                    },
                )
            )
    return documents


@lru_cache(maxsize=1)
def _build_index() -> tuple[BM25Retriever, FAISS]:
    """BM25 검색기와 FAISS 벡터 저장소를 한 번만 만들어 재사용한다."""
    documents = load_guides()
    bm25 = BM25Retriever.from_documents(
        documents, preprocess_func=tokenize, k=CANDIDATES_PER_RETRIEVER
    )
    embeddings = BedrockEmbeddings(
        model_id=os.environ["BEDROCK_EMBED_MODEL_ID"],
        region_name=os.environ["AWS_REGION"],
    )
    vectorstore = FAISS.from_documents(documents, embeddings)
    return bm25, vectorstore


def _reciprocal_rank_fusion(result_lists: list[list[Document]]) -> list[Document]:
    """여러 검색 결과 순위를 RRF 점수(1 / (RRF_K + 순위))로 합쳐 하나의 순위로 만든다."""
    scores: dict[tuple[str, str], float] = {}
    docs: dict[tuple[str, str], Document] = {}
    for results in result_lists:
        for rank, doc in enumerate(results, start=1):
            key = (doc.metadata["doc_id"], doc.metadata["section"])
            scores[key] = scores.get(key, 0.0) + 1.0 / (RRF_K + rank)
            docs[key] = doc
    ranked_keys = sorted(scores, key=scores.get, reverse=True)
    return [docs[key] for key in ranked_keys]


def search(query: str, k: int = 4) -> list[Document]:
    """질문과 관련된 가이드 문서 섹션을 BM25 + 벡터 하이브리드 검색으로 상위 k 개 반환한다."""
    bm25, vectorstore = _build_index()
    keyword_results = bm25.invoke(query)
    vector_results = vectorstore.similarity_search(query, k=CANDIDATES_PER_RETRIEVER)
    return _reciprocal_rank_fusion([keyword_results, vector_results])[:k]
